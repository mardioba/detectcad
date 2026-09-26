"""Rotas REST da API.

Todas as respostas JSON seguem o mesmo formato de erro::

    {"ok": false, "message": "..."}

Endpoints (conforme especificação do projeto)::

    GET    /api/status
    GET    /api/cameras            POST /api/cameras
    PUT    /api/cameras/{id}       DELETE /api/cameras/{id}
    GET    /api/count              GET  /api/history
    GET    /api/piles              GET  /api/models
    POST   /api/models/activate    POST /api/models/rollback
    GET    /api/dataset            POST /api/dataset/split
    GET    /api/training/status    POST /api/training/start
    GET    /api/calibration        POST /api/calibration
    GET    /api/events             GET  /api/snapshots
    GET    /api/health
    ...  mais: contagem manual, correções, importação de modelos, stream
"""

# ============================================================================
# ARQUIVO / MAPA  -  app/api/routes.py
# ============================================================================
# Camada REST do dashboard. Ela NÃO calcula nada: só lê o estado compartilhado
# (`state_store`), conversa com o banco via repositórios e delega o trabalho
# pesado aos serviços (contagem, dataset, treino, modelo).
#
# ORDEM DE LEITURA (por prefixo de rota):
#   1. helpers            _ok/_err, acessores _detector/_models/_cameras,
#                         _inference, _camera_id
#   2. /api/status        GET  - painel inteiro em 1 requisição (aba "today")
#      /api/health        GET  - liveness para monitor externo
#   3. /api/cameras       CRUD - lista, cria, atualiza, remove
#   4. /api/count         GET  - snapshot de contagem (números do topo)
#      /api/count/*       POST - corrigir e liberar a contagem manual
#      /api/piles         GET  - pilhas do banco
#      /api/history       GET  - histórico + série do gráfico
#      /api/corrections   GET  - correções manuais e sugestão de offset
#   5. /api/models        GET/POST - ativar, rollback, importar, registrar run
#   6. /api/dataset       GET/POST - stats, upload, split, auto-label
#      /api/annotation    GET  - dados da página de anotação
#   7. /api/training      GET/POST - status, start, stop, log
#   8. /api/calibration   GET/POST - ler, salvar, reset, medir altura da cadeira
#   9. /api/events        GET/DELETE - eventos do banco + eventos ao vivo
#  10. /api/snapshots     GET/POST/DELETE - listar, criar, apagar, servir JPEG
#  11. /api/stream.mjpg   GET  - stream MJPEG (o <img> do vídeo)
#      /api/frame.jpg     GET  - frame anotado (download/refresh)
#      /api/frame.raw.jpg GET  - frame sem overlay
#  12. /api/diagnostics   POST/GET - análise de ROI e imagem do recorte
#  13. /api/config        GET  - configuração efetiva (sem segredos)
#      /api/logs          GET  - cauda dos arquivos de log (aba Logs)
#  14. app_state          no fim do arquivo, injetado por main.py
#
# CONVENÇÕES QUE VALEM PARA O ARQUIVO INTEIRO:
#   * Resposta de sucesso = dict simples com {"ok": True, ...} (200 implícito).
#     Erro de negócio = `_err(...)`, que devolve JSONResponse com status 4xx/5xx
#     e o mesmo formato {"ok": False, "message": ...}. O frontend sempre lê
#     `message`, então os dois caminhos precisam ter essa chave.
#   * Só os dois handlers de upload (`async def`) são assíncronos, porque
#     `UploadFile.read()` é awaitable. Todo o resto é `def` síncrono: o FastAPI
#     joga no threadpool, então o event loop não trava com I/O de disco, SQLite
#     ou OpenCV. Regra prática: nada bloqueante dentro de `async def`.
#   * `mask_url(...)` SEMPRE que uma URL RTSP sai daqui. A senha do usuário
#     da câmera nunca vai para o navegador nem para o log.
# ============================================================================

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import select

from app.ai.model_manager import ModelManager, parse_version_from_name
from app.ai.yolo_detector import YoloDetector
from app.config import mask_url, settings
from app.counting.stack_counter import StackCounter
from app.database.database import session_scope
from app.database.models import Camera, CountCorrection, CountRecord
from app.database.repository import (
    CameraRepository,
    CorrectionRepository,
    CountRepository,
    EventRepository,
    ModelRepository,
    SnapshotRepository,
)
from app.schemas import CountStatus
from app.services.calibration_service import calibration_service
from app.services.counting_service import counting_service
from app.services.dataset_service import DatasetService, label_for
from app.services.state_store import state_store
from app.services.training_service import training_service

log = logging.getLogger("app")

router = APIRouter(prefix="/api", tags=["api"])


# --------------------------------------------------------------------- helpers
def _ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def _err(message: str, code: int = 400) -> JSONResponse:
    # 400 é o padrão porque quase todo erro aqui é "o pedido do cliente está
    # errado" (campo faltando, valor inválido, proportions que não somam 1).
    # Os handlers que querem outra semântica passam o código explícito
    # (404 = não encontrado, 409 = conflito de estado, 503 = dependência
    # indisponível). Usar JSONResponse em vez de HTTPException mantém o corpo
    # no mesmo formato {"ok": false, "message": ...} em 100% dos casos.
    return JSONResponse(status_code=code, content={"ok": False, "message": message})


# --- componentes acessados pela API -----------------------------------------
# A API precisa responder mesmo antes do bootstrap terminar (ou se a câmera
# falhou). Estes acessores entregam um objeto "vazio porém válido" nesse caso,
# em vez de estourar AttributeError e derrubar a página inteira.
_fallback: dict[str, Any] = {}


# Nota de thread-safety: este cache é um dict simples sem lock. É seguro
# porque (a) a construção do objeto é idempotente e (b) o GIL garante que
# `dict.__setitem__` é atômico - no pior caso duas threads constroem o
# fallback e uma das cópias é descartada, o que é inofensivo aqui.
# NÃO copiar esse padrão para estado que é lido e escrito pela API: nesse
# caso use `state_store`, que é protegido por RLock.
def _detector() -> YoloDetector:
    det = getattr(app_state, "detector", None)
    if det is None:
        # Detector sem modelo carregado: `info()` devolve loaded=False e os
        # endpoints continuam respondendo em vez de retornar 500.
        if "detector" not in _fallback:
            _fallback["detector"] = YoloDetector()
        det = _fallback["detector"]
    return det


def _models() -> ModelManager:
    mgr = getattr(app_state, "models", None)
    if mgr is None:
        if "models" not in _fallback:
            _fallback["models"] = ModelManager(_detector())
        mgr = _fallback["models"]
    return mgr


def _cameras() -> Any:
    cams = getattr(app_state, "cameras", None)
    if cams is None:
        if "cameras" not in _fallback:
            from app.camera.camera_manager import CameraManager

            _fallback["cameras"] = CameraManager()
        cams = _fallback["cameras"]
    return cams


def _inference() -> Any:
    # Diferente dos outros: aqui o fallback é None de verdade. Não existe
    # "worker de inferência de mentira" - quem chama já trata o None
    # (ex.: POST /api/snapshots responde 503 "Nenhum worker ativo").
    return getattr(app_state, "inference", None)


def _camera_id(camera_id: int | None = None) -> int:
    # A maior parte das rotas (calibração, diagnóstico) não recebe camera_id.
    # Quando o operador está na tela de uma câmera específica, o estado
    # global já sabe qual é - por isso o default vem do state_store.
    if camera_id is not None:
        return camera_id
    return state_store.get_status().get("camera_id", 1)


# ============================================================ STATUS / SAÚDE ===
@router.get("/status", response_model=None)
def api_status() -> dict[str, Any]:
    """Estado geral do sistema em um único objeto.

    É a requisição mais pesada e a mais chamada: o dashboard faz polling
    dela a cada poucos segundos para atualizar header, pilhas, histórico e
    abas. Por isso os limites são fixos e pequenos (120 pontos de histórico,
    25 eventos) - o gráfico não precisa de tudo, e o payload fica pequeno
    o bastante para não competir banda com o stream de vídeo.
    """
    store = state_store.get_status()
    detector = _detector()
    models = _models()
    cameras = _cameras()
    piles = state_store.get_piles()
    return _ok(
        system=store,
        camera=cameras.status(),
        piles=[p.as_dict() for p in piles],
        pile_count=len(piles),
        # only_stable=True: o número que a operação confia. Pilhas em
        # LOW_CONFIDENCE/UNKNOWN entram em "totals" (breakdown), não no total.
        total=state_store.total(only_stable=True),
        totals=state_store.totals_breakdown(),
        confidence=round(state_store.overall_confidence(), 4),
        model=detector.info(),
        model_active=models.get_active_info(),
        inference=_inference().status() if _inference() else None,
        history=state_store.get_history(limit=120),
        events=state_store.get_events(limit=25),
        training=training_service.status(),
    )


@router.get("/health", response_model=None)
def api_health() -> dict[str, Any]:
    """Liveness/ readiness simples, para monitor externo.

    Distingue "saudável" de "pronto": um sistema sem modelo e sem câmera
    ainda é saudável (está vivo e aceitando comandos), só não está pronto
    para contar. Monitorar `ready` para alarmar; `healthy` para reiniciar.
    """
    detector = _detector()
    store = state_store.get_status()
    model_ready = detector.is_loaded
    camera_state = store.get("camera_state", "OFFLINE")
    checks = {
        "database": True,
        "model_loaded": model_ready,
        "camera_online": camera_state == "ONLINE",
        "worker_running": bool(_inference() and _inference().status().get("running")),
    }
    # O sistema é "saudável" mesmo sem modelo/câmera: ele opera em modo de
    # preparação e mostra isso claramente. Só é "indisponível" se o banco ou
    # o worker principal estiverem quebrados.
    # Hoje `healthy` só olha o banco; os outros checks entram em `ready`.
    # Se algum dia a checagem do banco passar a ser real, é aqui que o
    # monitor externo começa a acusar restart.
    healthy = bool(checks["database"])
    return _ok(
        healthy=healthy,
        ready=bool(model_ready and checks["camera_online"]),
        mode=store.get("mode", "starting"),
        checks=checks,
        message=(
            "Operacional"
            if healthy
            else "Falha na inicialização - veja logs/app.log"
        ),
    )


# ================================================================== CÂMERAS ===
@router.get("/cameras", response_model=None)
def api_cameras() -> dict[str, Any]:
    """Lista as câmeras cadastradas (aba Câmera do dashboard)."""
    with session_scope() as session:
        rows = CameraRepository(session).list()
        out = [
            {
                "id": c.id,
                "name": c.name,
                # NUNCA expor a senha: mascarada aqui e no dashboard.
                # `mask_url` só mexe no pedaço user:pass@ da URL; host, porta
                # e caminho continuam visíveis porque ajudam a diagnosticar.
                "rtsp_url": mask_url(c.rtsp_url),
                "enabled": c.enabled,
                "is_default": c.is_default,
                "created_at": c.created_at.isoformat(timespec="seconds") if c.created_at else None,
            }
            for c in rows
        ]
    # `default_camera_id` separado porque o formulário do dashboard precisa
    # marcar qual é a câmera em uso, e isso não é um campo da tabela.
    return _ok(cameras=out, default_camera_id=_cameras().default_id())


@router.post("/cameras", response_model=None)
def api_create_camera(payload: dict[str, Any] = Body(...)) -> dict[str, Any] | JSONResponse:
    """Cadastra uma câmera e já tenta subir a captura em runtime."""
    # Validação feita à mão (e não com Pydantic) porque o corpo é livre:
    # o objetivo é apenas garantir que os dois campos obrigatórios existam
    # antes de gravar. O `str(...).strip()` evita nome/URL só com espaço.
    name = str(payload.get("name", "")).strip()
    url = str(payload.get("rtsp_url", "")).strip()
    if not name:
        return _err("Informe o nome da câmera.")
    if not url:
        return _err("Informe a URL RTSP em CAMERA_RTSP_URL (nunca no código).")
    with session_scope() as session:
        cam = CameraRepository(session).create(
            name=name, rtsp_url=url, enabled=bool(payload.get("enabled", True))
        )
    # Sobe a câmera em runtime (multicâmera)
    # O registro no banco vem PRIMEIRO e a captura depois: se a URL estiver
    # errada, o operador ainda tem a câmera listada para corrigir pelo PUT.
    # A falha ao abrir o RTSP não pode virar 500 - é log e segue.
    try:
        _cameras().add(cam.id, url=url, name=name, start=True)
    except Exception as exc:
        log.warning("Não foi possível iniciar a câmera %s: %s", cam.id, exc)
    # De novo com mask_url: este dicionário volta direto para o navegador.
    return _ok(camera={"id": cam.id, "name": cam.name, "rtsp_url": mask_url(cam.rtsp_url)})


@router.put("/cameras/{camera_id}", response_model=None)
def api_update_camera(camera_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any] | JSONResponse:
    """Atualiza nome/URL/flags e reinicia a captura se preciso."""
    # PATCH-like: só os campos presentes no payload entram em `fields`.
    # A whitelist de 4 chaves evita que o cliente escreva em colunas que
    # não deveria (id, created_at) via **fields.
    fields: dict[str, Any] = {}
    for key in ("name", "rtsp_url", "enabled", "is_default"):
        if key in payload and payload[key] is not None:
            fields[key] = payload[key]
    with session_scope() as session:
        cam = CameraRepository(session).update(camera_id, **fields)
        if cam is None:
            # 404 e não 400: o recurso não existe, o pedido estava correto.
            return _err(f"Câmera {camera_id} não encontrada.", 404)
        masked = mask_url(cam.rtsp_url)
    # Reinicia a captura se a URL mudou
    # Só a captura é reiniciada quando a URL/enabled muda. Mexer no nome não
    # justifica derrubar a stream: `remove` fecha a thread da câmera.
    if "rtsp_url" in fields or "enabled" in fields:
        _cameras().remove(camera_id)
        if fields.get("enabled", True):
            # URL vazia cai no default do .env dentro do CameraManager
            # (ele faz `url or settings.camera.camera_rtsp_url`).
            _cameras().add(camera_id, url=fields.get("rtsp_url", ""), name=fields.get("name", ""))
    return _ok(camera={"id": camera_id, "rtsp_url": masked})


@router.delete("/cameras/{camera_id}", response_model=None)
def api_delete_camera(camera_id: int) -> dict[str, Any] | JSONResponse:
    """Para a captura e apaga a câmera do banco."""
    # A parada vem ANTES do delete: se a consulta falhar (404), a câmera
    # continua registrada e o operador não perde o acesso aos frames.
    _cameras().remove(camera_id)
    with session_scope() as session:
        if not CameraRepository(session).delete(camera_id):
            return _err(f"Câmera {camera_id} não encontrada.", 404)
    return _ok(message=f"Câmera {camera_id} removida.")


# ================================================================= CONTAGEM ===
@router.get("/count", response_model=None)
def api_count() -> dict[str, Any]:
    """Contagem atual por pilha + total.

    Leve e sem banco: é só uma leitura do `state_store`. É o endpoint que o
    topo da tela consulta quando o WebSocket não está disponível.
    """
    return _ok(**counting_service.snapshot())


@router.post("/count/{pile_id}/correct", response_model=None)
def api_correct_count(pile_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Correção manual do operador.

    Registra ``AI_COUNT`` e ``CORRECT_COUNT`` para análise de erro e para
    gerar o offset de calibração sugerido. O valor fica travado: a IA não
    sobrescreve a correção sozinha.
    """
    value = payload.get("count")
    if value is None:
        return _err("Informe a nova contagem.")
    try:
        # int() normaliza "7" e 7.0. Strings não numéricas caem no except e
        # viram 400 com mensagem em vez de traceback 500.
        value = int(value)
    except (TypeError, ValueError):
        return _err("A contagem deve ser um número inteiro.")
    # O serviço valida o intervalo (0..10000) e exige worker vivo antes de
    # tocar no estado - a API não repete essas regras.
    result = counting_service.apply_manual_count(_inference(), pile_id, value)
    if not result.get("ok"):
        return _err(result.get("message", "Falha ao corrigir."))
    # Evento no state_store: aparece ao vivo no WebSocket e na lista de
    # eventos do banco, então a correção fica auditável na tela.
    state_store.push_event(
        "INFO", "count_corrected", f"Pilha {pile_id}: IA={result['ai_count']} -> {value}"
    )
    return _ok(**result, message=f"Pilha {pile_id} corrigida para {value} cadeiras.")


@router.post("/count/{pile_id}/release", response_model=None)
def api_release_count(pile_id: int) -> dict[str, Any]:
    """Devolve a pilha ao controle da IA após uma correção manual."""
    # Limpa o flag `manual` e destrava a estabilidade. Sem isso a pilha ficaria
    # travada no valor digitado pelo operador para sempre.
    result = counting_service.release_manual_count(_inference(), pile_id)
    if not result.get("ok"):
        return _err(result.get("message", "Falha."))
    return _ok(**result)


@router.get("/piles", response_model=None)
def api_piles(camera_id: int | None = None, limit: int = 200) -> dict[str, Any]:
    """Pilhas registradas no banco (auditoria, não é o estado ao vivo)."""
    # O estado ao vivo está em `state_store`; este endpoint lê a tabela
    # `piles`, que guarda a primeira/última sighting de cada tracking_id.
    with session_scope() as session:
        rows = []
        for p in _iter_piles(session, camera_id, limit):
            rows.append(
                {
                    "id": p.id,
                    "camera_id": p.camera_id,
                    "tracking_id": p.tracking_id,
                    "first_seen": p.first_seen.isoformat(timespec="seconds") if p.first_seen else None,
                    "last_seen": p.last_seen.isoformat(timespec="seconds") if p.last_seen else None,
                    "active": p.active,
                    "stable_count": p.stable_count,
                }
            )
    return _ok(piles=rows)


def _iter_piles(session: Any, camera_id: int | None, limit: int) -> list[Any]:
    # Helper separado para manter o handler só com a montagem do JSON.
    # O import de `Pile` é local: puxa a definição do modelo (e portanto o
    # metadata do SQLAlchemy) só quando a rota é realmente chamada.
    from app.database.models import Pile

    # `last_seen` desc para as pilhas mais recentes aparecerem primeiro -
    # é o que o operador quer ver. O LIMIT vem antes do WHERE na escrita, mas
    # o SQLAlchemy compila WHERE antes de LIMIT, então o filtro é aplicado
    # antes de cortar as linhas.
    stmt = select(Pile).order_by(Pile.last_seen.desc()).limit(limit)
    if camera_id is not None:
        stmt = stmt.where(Pile.camera_id == camera_id)
    return list(session.execute(stmt).scalars())


@router.get("/history", response_model=None)
def api_history(
    camera_id: int | None = None,
    hours: int = Query(default=24, ge=1, le=24 * 365),
    pile_id: int | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    """Histórico de contagens + série do total para o gráfico."""
    # ge/le no Query = validação com 422 pelo próprio FastAPI, antes de
    # chegar no código. Sem isso, um `hours=999999` viraria consulta pesada.
    # `.replace(tzinfo=None)` converte para o mesmo "ingênuo em UTC" que o
    # banco grava (ver `_utcnow` em app/database/models.py); comparar UTC
    # aware com ingênuo levantaria TypeError.
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    with session_scope() as session:
        repo = CountRepository(session)
        rows = repo.history(camera_id=camera_id, since=since, pile_id=pile_id, limit=limit)
        # `round(..., 4)` só para o JSON não carregar float longo demais.
        # As três confianças vêm separadas porque o dashboard mostra de onde
        # veio a dúvida: detecção (YOLO) ou estabilidade (temporal).
        records = [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(timespec="seconds"),
                "pile_id": r.pile_id,
                "camera_id": r.camera_id,
                "count": r.chair_count,
                "confidence": round(r.confidence, 4),
                "detection_confidence": round(r.detection_confidence, 4),
                "stability_confidence": round(r.stability_confidence, 4),
                "status": r.status,
                "method": r.method,
                "total": r.total_chairs,
                "previous_count": r.previous_count,
                "origin": r.origin,
            }
            for r in rows
        ]
        total_series: list[dict[str, Any]] = []
        # A série do total só faz sentido para UMA câmera: somar o histórico
        # de várias câmeras daria um número que não existe no mundo real.
        if camera_id is not None:
            series = repo.totals_series(camera_id, since)
            total_series = [
                {"timestamp": ts.isoformat(timespec="seconds"), "total": total} for ts, total in series
            ]
        # Sem camera_id, o "agora" vem do estado em memória (que já sabe a
        # câmera ativa) em vez do banco.
        total_now = repo.total_at(camera_id) if camera_id is not None else state_store.total()
    return _ok(records=records, totals_series=total_series, current_total=total_now, hours=hours)


@router.get("/corrections", response_model=None)
def api_corrections(limit: int = 200) -> dict[str, Any]:
    """Correções manuais (dados de análise de erro do modelo)."""
    # Cada correção é um par (o que a IA viu, o que o humans saw). Esse par
    # alimenta duas coisas: a estatística de erro e o offset sugerido.
    with session_scope() as session:
        repo = CorrectionRepository(session)
        rows = repo.list(limit=limit)
        items = [
            {
                "id": c.id,
                "timestamp": c.timestamp.isoformat(timespec="seconds"),
                "camera_id": c.camera_id,
                "ai_count": c.ai_count,
                "correct_count": c.correct_count,
                "ai_confidence": round(c.ai_confidence, 4),
                "author": c.author,
                "note": c.note,
            }
            for c in rows
        ]
        stats = repo.stats()
        # `offset_suggestion` abre a própria sessão: fica aqui dentro do
        # `with` só porque não importa para a transação em curso.
        suggestion = calibration_service.offset_suggestion()
    # O dashboard mostra a sugestão (ex.: "a IA conta 1 a menos em média")
    # ao lado do formulário de calibração para o operador aplicar ou ignorar.
    return _ok(corrections=items, stats=stats, offset_suggestion=suggestion)


# =================================================================== MODELOS ===
@router.get("/models", response_model=None)
def api_models() -> dict[str, Any]:
    """Modelos disponíveis, o ativo e o que está carregado em memória.

    Junta duas fontes que podem divergir: o banco (`list_models`, que
    sincroniza o disco com os registros) e o detector (`info`, que diz o que
    está de fato em RAM). Por isso os dois campos existem no payload.
    """
    detector = _detector()
    models = _models()
    return _ok(
        active=models.get_active_info(),
        loaded=detector.info(),
        models=models.list_models(),
    )


@router.post("/models/activate", response_model=None)
def api_activate_model(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Ativa um modelo em produção (hot-swap, sem derrubar o pipeline).

    Aceita `model_id` (do banco) OU `filename` (do disco). O hot-swap troca
    os pesos dentro do detector enquanto a thread de inferência continua
    rodando: a câmera não cai e o histórico não zera.
    """
    # Body com default={} para o botão do dashboard poder mandar POST vazio.
    models = _models()
    model_id = payload.get("model_id")
    filename = payload.get("filename")
    if model_id is None and not filename:
        return _err("Informe model_id ou filename.")
    result = models.activate(
        model_id=int(model_id) if model_id is not None else None,
        filename=str(filename) if filename else None,
    )
    if result.get("ok"):
        state_store.push_event("INFO", "model_activated", result.get("message", "Modelo ativado."))
        # Atualiza o state_store para o polling de /api/status refletir o
        # modelo novo antes do próximo ciclo do worker.
        state_store.set_status(model_loaded=True, model_filename=result.get("filename"))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/models/rollback", response_model=None)
def api_rollback_model() -> dict[str, Any]:
    """Volta ao modelo anterior (o que estava ativo antes do hot-swap)."""
    models = _models()
    result = models.rollback()
    # Sem push_event aqui de propósito: rollback é ação corretiva rara e o
    # retorno da chamada já informa o resultado a quem clicou no botão.
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/models/import", response_model=None)
async def api_import_model(file: UploadFile = File(...)) -> dict[str, Any]:
    """Importa um .pt treinado para ``data/models``."""
    # Handler async porque `UploadFile.read()` é awaitable. É um dos dois
    # únicos pontos do arquivo onde o body chega em streaming; nada mais
    # aqui pode ser async sem cuidado, porque o FastAPI manda `async def`
    # direto para o event loop (sem threadpool).
    models = _models()
    # `Path(...).name` descarta qualquer diretório do nome enviado pelo
    # cliente. Sem isso um "../evil.pt" escaparia de data/models.
    name = Path(file.filename or "model.pt").name
    if not name.endswith(".pt"):
        return _err("O arquivo precisa ter extensão .pt")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    target = settings.models_dir / name
    try:
        # Leitura em blocos de 1 MB: um .pt tem dezenas de MB e ler inteiro
        # na memória do request estoura o heap do servidor.
        with target.open("wb") as fh:
            while chunk := file.read(1024 * 1024):
                fh.write(chunk)
    except OSError as exc:
        return _err(f"Falha ao gravar o arquivo: {exc}")
    result = models.import_model(target)
    if result.get("ok"):
        state_store.push_event("INFO", "model_imported", result.get("message", "Modelo importado."))
    # Importar NÃO ativa: o operador escolhe o modelo explicitamente depois.
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/models/register-run", response_model=None)
def api_register_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Registra o ``best.pt`` de um run existente (``training/runs/<nome>``)."""
    run = str(payload.get("run_dir", "")).strip()
    if not run:
        return _err("Informe o run_dir (ex.: training/runs/chair_counter).")
    path = Path(run)
    if not path.is_absolute():
        # Relativo ancorado na raiz do projeto: o dashboard manda
        # "training/runs/x" e isso continua valendo de onde o servidor for
        # iniciado, em vez de depender do diretório de trabalho atual.
        path = settings.base_dir / path
    if not path.is_dir():
        return _err(f"Pasta de run não encontrada: {path}")
    models = _models()
    # `activate` é opcional: registrar um run antigo não pode trocar o modelo
    # em produção sem o operador pedir.
    result = models.register_trained(path, make_active=bool(payload.get("activate", False)))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


# =================================================================== DATASET ===
@router.get("/dataset", response_model=None)
def api_dataset() -> dict[str, Any]:
    """Resumo do dataset (aba Dataset): contagens, classes e avisos.

    Chamar `write_yaml()` de propósito: garante que o chairs.yaml usado pelo
    treino aponte para os diretórios absolutos atuais, mesmo que alguém tenha
    mexido no arquivo na mão.
    """
    ds = DatasetService()
    stats = ds.stats()
    return _ok(stats=stats.as_dict(), yaml=str(ds.write_yaml()), root=str(ds.root))


@router.get("/dataset/images", response_model=None)
def api_dataset_images(limit: int = 200) -> dict[str, Any]:
    """Lista de imagens com estado de anotação (equivale a /api/annotation)."""
    return _ok(**DatasetService().image_status(limit=limit))


@router.get("/dataset/image", response_model=None)
def api_dataset_image(path: str = Query(...)) -> Any:
    """Serve uma imagem do dataset (path relativo ao dataset root)."""
    ds = DatasetService()
    # `.resolve()` canonicaliza o caminho (incluindo links simbólicos) para
    # que a checagem abaixo realmente impeça escapar do dataset root.
    target = (ds.root / path).resolve()
    try:
        # Path traversal: se o `path` do cliente vier com "../..", o
        # `relative_to` levanta ValueError e caímos no 400. Sem essa guarda
        # daria para baixar qualquer arquivo do servidor pela API.
        target.relative_to(ds.root.resolve())
    except ValueError:
        return _err("Caminho inválido.", 400)
    if not target.is_file():
        return _err("Imagem não encontrada.", 404)
    return FileResponse(target)


@router.get("/dataset/label", response_model=None)
def api_dataset_label(path: str = Query(...)) -> Any:
    """Serve o .txt de rótulo no formato YOLO.

    A repetição da proteção de path traversal é proposital: são dois
    handlers independentes, e nenhum deve herdar segurança do outro.
    """
    ds = DatasetService()
    target = (ds.root / path).resolve()
    try:
        target.relative_to(ds.root.resolve())
    except ValueError:
        return _err("Caminho inválido.", 400)
    if not target.is_file():
        return _err("Rótulo não encontrado.", 404)
    # media_type explícito: sem isso o FastAPI inferiria do nome do arquivo e
    # poderia servir o .txt como download em vez de texto.
    return FileResponse(target, media_type="text/plain")


@router.post("/dataset/upload", response_model=None)
async def api_dataset_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Recebe imagem crua e joga na pasta `raw`, aguardando anotação."""
    ds = DatasetService()
    name = Path(file.filename or "imagem.jpg").name
    # Grava direto em `raw` (a "caixa de entrada" do dataset) e não em
    # images/train: uma imagem sem rótulo não pode entrar no treino.
    tmp = settings.source_images_dir / name
    ds.ensure_structure()
    try:
        # Mesmos blocos de 1 MB da importação de modelo: é a mesma
        # restrição de memória do FastAPI com UploadFile.
        with tmp.open("wb") as fh:
            while chunk := file.read(1024 * 1024):
                fh.write(chunk)
    except OSError as exc:
        return _err(f"Falha ao gravar: {exc}")
    result = ds.add_image(tmp, copy=False)
    # Evento para o operador ver o upload chegando ao vivo na aba Logs.
    state_store.push_event("INFO", "dataset_upload", f"Imagem adicionada: {name}")
    # A mensagem final diz o próximo passo ("agora anote-a") porque a imagem
    # sozinha não treina nada: ela ainda precisa do rótulo.
    return _ok(**result, message=f"Imagem {name} adicionada. Agora anote-a.")


@router.post("/dataset/split", response_model=None)
def api_dataset_split(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Divide o dataset em treino/validação/teste com seed reproduzível.

    A mesma seed precisa gerar sempre a mesma divisão: sem isso não dá para
    comparar dois treinamentos e saber se o ganho veio do modelo ou do sorteio.
    """
    ds = DatasetService()
    # default={} no Body: o formulário da aba Dataset manda sempre os campos,
    # mas uma chamada vazia cai nos percentuais padrão.
    result = ds.split(
        train=float(payload.get("train", 0.70)),
        val=float(payload.get("val", 0.20)),
        test=float(payload.get("test", 0.10)),
        seed=int(payload.get("seed", 42)),
    )
    if result.get("ok"):
        state_store.push_event("INFO", "dataset_split", result.get("message", ""))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/dataset/auto-label", response_model=None)
def api_dataset_auto_label(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Pré-rotula imagens sem anotação usando o modelo ativo.

    Atalho para não começar do zero, NÃO substitui o operador: o conf
    padrão (0.35) é baixo de propósito, para gerar caixas em erro que depois
    precisam de revisão humana.
    """
    ds = DatasetService()
    result = ds.auto_label_from_model(conf=float(payload.get("conf", 0.35)))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.get("/annotation", response_model=None)
def api_annotation_page(limit: int = 200) -> dict[str, Any]:
    """Dados da página /annotation."""
    ds = DatasetService()
    # Garante as pastas train/val/test antes de varrer: numa instalação nova
    # elas nem existem e a listagem viria vazia sem erro claro.
    ds.ensure_structure()
    return _ok(**ds.image_status(limit=limit))


# ================================================================ TREINAMENTO ===
@router.get("/training/status", response_model=None)
def api_training_status() -> dict[str, Any]:
    """Progresso do treino (aba Treinamento): epoch, loss e GPU.

    Chamar `check_finished()` antes de ler: é ela que percebe que o processo
    filho terminou e promove o best.pt para a lista de modelos. Sem essa
    chamada, um treino finalizado ficaria "rodando" para sempre até alguém
    reiniciar o servidor.
    """
    training_service.check_finished()
    return _ok(**training_service.status())


@router.post("/training/start", response_model=None)
def api_training_start(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Dispara o treinamento num processo separado (não bloqueia a API).

    Só um treino por vez: o serviço roda `yolo train` como subprocesso e o
    servidor precisa continuar atendendo o dashboard durante as horas de GPU.
    Dois treinos simultâneos brigariam pela mesma GPU e pela pasta de saída.
    """
    # Cada campo é repassado como None quando ausente; o serviço aplica o
    # default do .env nessa hora, deixando a config em um lugar só.
    result = training_service.start(
        model=payload.get("model"),
        epochs=payload.get("epochs"),
        image_size=payload.get("image_size"),
        batch=payload.get("batch"),
        device=payload.get("device"),
        workers=payload.get("workers"),
        patience=payload.get("patience"),
        seed=payload.get("seed"),
        name=payload.get("name"),
        data=payload.get("data"),
    )
    if result.get("ok"):
        state_store.push_event("INFO", "training_started", result.get("message", ""))
    # 409 (Conflito) e não 400 quando já existe treino rodando: o pedido era
    # válido, quem impede é o estado do sistema.
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha."), 409))


@router.post("/training/stop", response_model=None)
def api_training_stop() -> dict[str, Any]:
    """Interrompe o treino atual (o serviço tenta terminate e depois kill)."""
    result = training_service.stop()
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.get("/training/log", response_model=None)
def api_training_log(lines: int = 80) -> dict[str, Any]:
    """Cauda do log do treino (o operador acompanha epoch/loss por aqui)."""
    # Lê o arquivo e corta as últimas N linhas. Um `tail` de verdade exigiria
    # manter o arquivo aberto entre requisições; a simplicidade compensa
    # porque a aba só pede essas linhas de tempos em tempos.
    return _ok(log=training_service.tail_log(lines))


# =============================================================== CALIBRAÇÃO ===
@router.get("/calibration", response_model=None)
def api_calibration(camera_id: int | None = None) -> dict[str, Any]:
    """Calibração em vigor da câmera (aba Calibração)."""
    cid = _camera_id(camera_id)
    data = calibration_service.load(cid)
    # `suggestion` ao lado: mostra quanto a IA errou nas últimas correções
    # manuais, para o operador decidir se compensa aplicar um count_offset.
    return _ok(calibration=data.as_dict(), suggestion=calibration_service.offset_suggestion())


@router.post("/calibration", response_model=None)
def api_save_calibration(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Salva a calibração no banco/JSON e aplica na hora no worker.

    Este é o coração da aba Calibração: mudar `chair_height_px` ou
    `count_offset` só tem efeito se o worker em background receber o valor
    novo AGORA. Se ficasse só no banco, o operador teria que reiniciar a
    câmera para ver o resultado e não conseguiria ajustar "no olho".
    """
    cid = _camera_id(payload.get("camera_id"))
    data = calibration_service.save(cid, payload)
    # Aplica imediatamente no worker, sem reiniciar a câmera
    if _inference():
        # O `set_calibration` do worker tem lock próprio: roda na thread da
        # API enquanto a thread de inferência conta, e sem o lock o contador
        # poderia ler metade dos parâmetros antigos e metade dos novos, gerou
        # uma contagem com configuração misturada. Por isso nunca chamar
        # `counter.x = ...` direto daqui.
        #
        # O dicionário montado abaixo é a TRADUÇÃO: as colunas do banco (uma
        # por linha) viram os três blocos que o worker entende
        # (roi / counting / ai).
        _inference().set_calibration(
            {
                "roi": data.roi.__dict__,
                "roi_enabled": data.enabled,
                "counting": {
                    "chair_height_px": data.chair_height_px,
                    "min_confidence": data.min_confidence,
                    "chair_weight": data.chair_weight,
                    "count_offset": data.count_offset,
                    "counting_method": data.counting_method,
                    "stability_frames": data.stability_frames,
                    "change_min_frames": data.change_min_frames,
                    "stability_min_confidence": data.stability_min_confidence,
                    "pile_gap_px": data.pile_gap_px,
                },
                "ai": {
                    "confidence_threshold": data.confidence_threshold,
                    "chair_classes": data.chair_classes,
                    "pile_classes": data.pile_classes,
                },
            }
        )
    # Sem worker (câmera desligada) a calibração fica salva no banco e vale
    # para o próximo start - por isso não é erro.
    state_store.push_event("INFO", "calibration_saved", f"Calibração da câmera {cid} atualizada")
    return _ok(calibration=data.as_dict(), message="Calibração salva e aplicada.")


@router.post("/calibration/reset", response_model=None)
def api_reset_calibration(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Apaga a calibração salva e volta aos padrões do .env."""
    cid = _camera_id(payload.get("camera_id"))
    data = calibration_service.reset(cid)
    if _inference():
        # "counting": {} vazio é de propósito: o worker completa cada campo
        # ausente com o default do .env (ver InferenceService.set_calibration).
        _inference().set_calibration(
            {"roi": data.roi.__dict__, "roi_enabled": data.enabled, "counting": {}}
        )
    return _ok(calibration=data.as_dict(), message="Calibração restaurada para o padrão do .env.")


@router.post("/calibration/measure", response_model=None)
def api_measure_chair_height(payload: dict[str, Any] = Body(default={})) -> dict[str, Any] | JSONResponse:
    """Mede a altura de uma cadeira em pixels a partir de uma ROI.

    O operador desenha a caixa de UMA cadeira; o sistema mede a altura real
    em pixels. É a forma correta de obter ``chair_height_px`` (nunca chutar).
    """
    # Mede sobre o ÚLTIMO frame processado, não sobre a foto enviada: é o
    # mesmo frame que o operador está vendo no vídeo, então a medida vale.
    result = counting_service.measure_chair_height(state_store.get_frame_bgr(), payload.get("roi") or {})
    # 503 e não 400: o pedido está correto, o que falta é um frame (a câmera
    # ainda não produziu nada). 503 diz "tente de novo em instantes".
    return _ok(**result) if result.get("ok") else _err(result.get("message", "Falha."), 503)


# =================================================================== EVENTOS ===
@router.get("/events", response_model=None)
def api_events(limit: int = 200, level: str | None = None) -> dict[str, Any]:
    """Eventos gravados no banco + os que ainda estão só em memória.

    A aba Logs mostra os dois: `events` é o histórico que sobrevive a
    reinício; `live` é o buffer circular do state_store, mais imediato.
    O filtro `level` (INFO/WARN/ERROR) existe para o operador isolar só os
    erros quando o log está grande.
    """
    with session_scope() as session:
        rows = EventRepository(session).list(limit=limit, level=level)
        items = [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(timespec="seconds"),
                "level": e.level,
                "event": e.event,
                "message": e.message,
            }
            for e in rows
        ]
    return _ok(events=items, live=state_store.get_events(limit=50))


@router.delete("/events", response_model=None)
def api_clear_events() -> dict[str, Any]:
    """Limpa a tabela de eventos (botão "limpar" da aba Logs).

    Só o banco: o buffer em memória continua, senão a lista piscaria sozinha
    com os eventos que ainda chegam do worker.
    """
    from app.database.models import SystemEvent

    with session_scope() as session:
        # Delete linha a linha em vez de um DELETE FROM sem WHERE para que a
        # contagem do final (len(rows)) saia de graça. A tabela é pequena.
        rows = session.execute(select(SystemEvent)).scalars().all()
        for row in rows:
            session.delete(row)
    return _ok(message=f"{len(rows)} evento(s) removido(s).")


# ================================================================= SNAPSHOTS ===
@router.get("/snapshots", response_model=None)
def api_snapshots(limit: int = 100) -> dict[str, Any]:
    """Lista de snapshots com a URL de download de cada um."""
    with session_scope() as session:
        rows = SnapshotRepository(session).list(limit=limit)
        items = [
            {
                "id": s.id,
                "timestamp": s.timestamp.isoformat(timespec="seconds"),
                "filename": s.filename,
                # `reason` distingue snapshot manual de contagem-mudou ou
                # pilha-apareceu, que são automáticos.
                "reason": s.reason,
                "total": s.total,
                "pile_count": s.pile_count,
                "confidence": round(s.confidence, 4) if s.confidence is not None else None,
                "size_kb": round((s.snapshot_size or 0) / 1024, 1),
                # URL montada aqui (e não no frontend) para a rota do arquivo
                # ficar em um único lugar só.
                "url": f"/api/snapshots/file/{s.filename}",
            }
            for s in rows
        ]
    return _ok(snapshots=items, directory=str(settings.snapshots_dir))


@router.get("/snapshots/file/{filename}", response_model=None)
def api_snapshot_file(filename: str) -> Any:
    """Serve o JPEG do snapshot pelo nome do arquivo."""
    svc = _inference().snapshots if _inference() else None
    # O `resolve()` do SnapshotService já aplica `Path(filename).name` +
    # `relative_to` (path traversal). Delegar a ele evita duplicar a mesma
    # checagem de segurança em dois lugares do projeto.
    path = svc.resolve(filename) if svc else None
    if path is None:
        return _err("Snapshot não encontrado.", 404)
    return FileResponse(path, media_type="image/jpeg")


@router.post("/snapshots", response_model=None)
def api_create_snapshot(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Salva um snapshot manualmente (botão do dashboard)."""
    # Body com default={}: o frontend chama sem mandar nada. Total, número de
    # pilhas e confiança vêm do state_store, não do cliente, para o registro
    # refletir o que a IA realmente viu naquele instante.
    if not _inference():
        return _err("Nenhum worker ativo.", 503)
    frame = state_store.get_frame_bgr()
    if frame is None:
        return _err("Nenhum frame disponível ainda.", 503)
    res = _inference().snapshots.save(
        frame,
        reason="manual",
        total=state_store.total(),
        pile_count=state_store.pile_count,
        confidence=state_store.overall_confidence(),
    )
    if res is None:
        # 500: o frame existia, mas o cv2.imwrite falhou (disco cheio, pasta
        # sem permissão). É falha do servidor, não do pedido.
        return _err("Não foi possível salvar o snapshot.", 500)
    state_store.push_event("INFO", "snapshot_manual", f"Snapshot manual: {res['filename']}")
    return _ok(**res, message="Snapshot salvo.")


@router.delete("/snapshots/{snapshot_id}", response_model=None)
def api_delete_snapshot(snapshot_id: int) -> dict[str, Any]:
    """Apaga o registro do snapshot e o arquivo JPEG no disco."""
    with session_scope() as session:
        repo = SnapshotRepository(session)
        row = repo.get(snapshot_id)
        if row is None:
            return _err("Snapshot não encontrado.", 404)
        # O caminho do arquivo é montado DENTRO do `with`: depois do commit
        # o objeto `row` já não pode ser usado.
        path = settings.snapshots_dir / row.filename
        repo.delete(snapshot_id)
    # Banco primeiro, arquivo depois. A falha de disco é engolida de propósito:
    # um JPEG órfão é lixo recuperável, mas devolver erro depois do commit
    # já efetivado seria mentira para o frontend.
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    return _ok(message="Snapshot removido.")


# ==================================================================== STREAM ===
@router.get("/stream.mjpg", response_model=None)
def api_stream_mjpg() -> StreamingResponse:
    """Stream MJPEG do frame anotado (usado pelo <img> do dashboard).

    MJPEG em vez de WebSocket para a imagem porque o navegador decodifica e
    exibe um stream multipart sozinho, numa tag <img>, sem JS e sem canvas.
    O WebSocket fica só com os números (ver app/api/websocket.py).
    """

    def generate():  # type: ignore[no-untyped-def]
        # O handler é `def` síncrono, então este generator roda no threadpool
        # e o `time.sleep` abaixo NÃO trava o event loop do servidor. É o que
        # permite vários dashboards abertos ao mesmo tempo.
        last = -1
        idle = 0
        while idle < 100:  # ~20s sem frame: encerra
            # min_counter=last só entrega o JPEG se ele for NOVO. Sem isso
            # cada cliente receberia o mesmo frame várias vezes por segundo,
            # gastando banda à toa.
            jpeg, counter = state_store.get_frame_jpeg(last)
            if jpeg is None:
                # Sem frame novo: espera ~1/FPS e tenta de novo. O FPS vem do
                # .env para o stream respeitar o mesmo ritmo configurado.
                time.sleep(1.0 / max(1.0, settings.web.stream_fps))
                idle += 1
                continue
            idle = 0
            last = counter
            # Formato multipart/x-mixed-replace: cada "parte" é um JPEG
            # completo. É o protocolo que o <img src=...> do dashboard
            # consome sem nenhuma biblioteca no front.
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store\r\n\r\n" + jpeg + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        # `X-Accel-Buffering: no` desliga o buffer do nginx quando há proxy
        # na frente: com buffer, o vídeo chega com segundos de atraso.
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/frame.jpg", response_model=None)
def api_frame_jpeg() -> Any:
    """Frame anotado atual (para download).

    Versão "solta" do stream: usada quando o operador tira print ou quer
    o frame mais recente sem abrir outra conexão de streaming.
    """
    # min_counter=-1 devolve sempre o último frame, mesmo que nenhum tenha
    # chegado desde a última chamada - aqui não queremos esperar novidade.
    jpeg, _ = state_store.get_frame_jpeg(-1)
    if jpeg is None:
        return _err("Nenhum frame disponível.", 404)
    from fastapi.responses import Response

    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/frame.raw.jpg", response_model=None)
def api_frame_raw() -> Any:
    """Frame **sem** overlay, para conferir a imagem original.

    Usado no botão "ver original": deixa o operador conferir se a câmera está
    enquadrada do jeito que ele acha que está, sem as caixas desenhadas
    enviesando o julgamento.
    """
    frame = state_store.get_frame_bgr()
    if frame is None:
        return _err("Nenhum frame disponível.", 404)
    from fastapi.responses import Response

    # O JPEG anotado já vem pronto do worker (ele codifica uma vez por
    # frame); aqui é preciso codificar de novo. Qualidade 90: acima disso o
    # arquivo dobra de tamanho sem ganho visível, e abaixo disso o texto do
    # overlay fica serrilhado.
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return _err("Falha ao codificar o frame.", 500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# =============================================================== DIAGNÓSTICO ===
@router.post("/diagnostics/counter", response_model=None)
def api_counter_diagnostics(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Analisa uma ROI do frame atual e mostra o detalhe da contagem.

    Usado na página de Calibração para o operador ver o que o algoritmo
    enxerga (passo, picos, camadas) antes de confiar no número.
    """
    calib = calibration_service.load(_camera_id())
    result = counting_service.diagnose_roi(
        state_store.get_frame_bgr(),
        payload.get("roi") or {},
        chair_height_px=calib.chair_height_px,
        count_offset=calib.count_offset,
    )
    if not result.get("ok"):
        return _err(result.get("message", "Falha."), 503)
    est, analysis = result["estimate"], result["analysis"]
    # Salva o recorte para inspeção visual
    img_url = None
    try:
        out = settings.results_dir / "diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        # Timestamp no nome: a mesma ROI pode ser medida várias vezes seguidas
        # e o operador precisa comparar os recortes sem sobrescrever o anterior.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        roi = result["roi"]
        path = out / f"roi_{stamp}_{roi['w']}x{roi['h']}.jpg"
        frame = state_store.get_frame_bgr()
        if frame is not None:
            # Slice numpy da região medida, direto do frame BGR.
            cv2.imwrite(str(path), frame[roi["y"] : roi["y"] + roi["h"], roi["x"] : roi["x"] + roi["w"]])
            img_url = f"/api/diagnostics/image?path={path.name}"
    except Exception:
        # A imagem é um extra: se o disco falhar, a análise numérica - que é o
        # que o operador veio buscar - continua sendo devolvida normalmente.
        img_url = None
    return _ok(
        roi=result["roi"],
        estimate=est,
        analysis=analysis,
        image_url=img_url,
        message=f"Estimativa: {est['count']} cadeiras (confiança {est['confidence'] * 100:.0f}%)",
    )


@router.get("/diagnostics/image", response_model=None)
def api_diagnostics_image(path: str = Query(...)) -> Any:
    """Serve o recorte de diagnóstico salvo pela rota anterior."""
    # `Path(path).name` corta diretórios e o `relative_to` confirma que o
    # resultado continua dentro de results/diagnostics. As duas juntas
    # bloqueiam path traversal.
    target = (settings.results_dir / "diagnostics" / Path(path).name).resolve()
    try:
        target.relative_to((settings.results_dir / "diagnostics").resolve())
    except ValueError:
        return _err("Caminho inválido.", 400)
    if not target.is_file():
        return _err("Imagem não encontrada.", 404)
    return FileResponse(target)


# ===================================================================== UTILS ===
@router.get("/config", response_model=None)
def api_config() -> dict[str, Any]:
    """Configuração efetiva (sem segredos).

    Serve para a aba Configuração mostrar de onde cada valor veio, então
    reflete o que está em memória AGORA - inclusive o que a calibração
    sobrescreveu - e não apenas o arquivo .env.
    """
    return _ok(
        camera={
            "name": settings.camera.camera_name,
            # mask_url de novo: a URL efetiva pode ter vindo de cadastro ou
            # calibração, então precisa ser mascarada aqui também.
            "rtsp_url": mask_url(settings.camera.camera_rtsp_url),
            "process_fps": settings.camera.process_fps,
            "transport": settings.camera.transport,
        },
        ai={
            "model": settings.ai.yolo_model,
            "device": settings.resolve_device(),
            "confidence_threshold": settings.ai.confidence_threshold,
            "iou_threshold": settings.ai.iou_threshold,
            "imgsz": settings.ai.imgsz,
            "tracker": settings.ai.tracker,
            "chair_classes": settings.ai.chair_classes,
            "pile_classes": settings.ai.pile_classes,
        },
        counting={
            "stability_frames": settings.counting.count_stability_frames,
            "stability_mode": settings.counting.stability_mode,
            "change_min_frames": settings.counting.change_min_frames,
            "counting_method": settings.counting.counting_method,
            "min_confidence": settings.counting.min_confidence,
            "chair_height_px": settings.counting.chair_height_px,
        },
        snapshots={"enabled": settings.snapshots.save_snapshots, "interval_sec": settings.snapshots.interval_sec},
        alerts={
            "enabled": settings.alerts.enabled,
            "total_min": settings.alerts.total_min,
            "total_max": settings.alerts.total_max,
            "pile_min": settings.alerts.pile_min,
        },
        # `split("@")[-1]` descarta o "usuário:senha@" de uma URL de banco,
        # mantendo o caminho do arquivo SQLite (que é útil ver na tela).
        database=settings.web.database_url.split("@")[-1],
        host=settings.web.host,
        port=settings.web.port,
    )


@router.get("/logs", response_model=None)
def api_logs(file: str = Query(default="app.log"), lines: int = 120) -> dict[str, Any]:
    """Últimas linhas de um dos arquivos de log (aba Logs)."""
    # Allowlist + `Path(file).name`: sem isso, `?file=../../app/config.py`
    # leria um arquivo Python (ou o .env!) pela API. O `.name` sozinho já
    # impediria o escape de diretório, mas a allowlist fecha as duas portas.
    name = Path(file).name
    allowed = {"app.log", "camera.log", "ai.log", "training.log", "training_latest.log"}
    if name not in allowed:
        return _err(f"Arquivo de log não permitido. Use: {sorted(allowed)}")
    path = settings.logs_dir / name
    if not path.is_file():
        # 200 com log vazio: um log que ainda não existe é normal logo após
        # o boot. O frontend mostra "sem dados" em vez de um erro vermelho.
        return _ok(log="", message=f"{name} ainda não existe.")
    try:
        # errors="replace": um log truncado por um crash pode ter bytes
        # inválidos no fim; sem replace, a rota inteira explodiria.
        content = path.read_text(encoding="utf-8", errors="replace")
        return _ok(log="\n".join(content.splitlines()[-lines:]), file=name)
    except OSError as exc:
        return _err(str(exc), 500)


# Injetado em main.py para evitar import circular
# `app_state` fica no FIM do arquivo de propósito: os helpers acima já o
# referenciam por nome, e o Python só resolve esse nome quando a função é
# CHAMADA (em runtime), nunca no momento do import. Se estivesse no topo,
# main.py ainda não teria injetado o valor e o primeiro request quebraria.
app_state: Any = None


__all__ = ["router", "app_state"]
