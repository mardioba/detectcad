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
    return JSONResponse(status_code=code, content={"ok": False, "message": message})


# --- componentes acessados pela API -----------------------------------------
# A API precisa responder mesmo antes do bootstrap terminar (ou se a câmera
# falhou). Estes acessores entregam um objeto "vazio porém válido" nesse caso,
# em vez de estourar AttributeError e derrubar a página inteira.
_fallback: dict[str, Any] = {}


def _detector() -> YoloDetector:
    det = getattr(app_state, "detector", None)
    if det is None:
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
    return getattr(app_state, "inference", None)


def _camera_id(camera_id: int | None = None) -> int:
    if camera_id is not None:
        return camera_id
    return state_store.get_status().get("camera_id", 1)


# ============================================================ STATUS / SAÚDE ===
@router.get("/status", response_model=None)
def api_status() -> dict[str, Any]:
    """Estado geral do sistema em um único objeto."""
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
    """Liveness/ readiness simples, para monitor externo."""
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
    with session_scope() as session:
        rows = CameraRepository(session).list()
        out = [
            {
                "id": c.id,
                "name": c.name,
                # NUNCA expor a senha: mascarada aqui e no dashboard.
                "rtsp_url": mask_url(c.rtsp_url),
                "enabled": c.enabled,
                "is_default": c.is_default,
                "created_at": c.created_at.isoformat(timespec="seconds") if c.created_at else None,
            }
            for c in rows
        ]
    return _ok(cameras=out, default_camera_id=_cameras().default_id())


@router.post("/cameras", response_model=None)
def api_create_camera(payload: dict[str, Any] = Body(...)) -> dict[str, Any] | JSONResponse:
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
    try:
        _cameras().add(cam.id, url=url, name=name, start=True)
    except Exception as exc:
        log.warning("Não foi possível iniciar a câmera %s: %s", cam.id, exc)
    return _ok(camera={"id": cam.id, "name": cam.name, "rtsp_url": mask_url(cam.rtsp_url)})


@router.put("/cameras/{camera_id}", response_model=None)
def api_update_camera(camera_id: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any] | JSONResponse:
    fields: dict[str, Any] = {}
    for key in ("name", "rtsp_url", "enabled", "is_default"):
        if key in payload and payload[key] is not None:
            fields[key] = payload[key]
    with session_scope() as session:
        cam = CameraRepository(session).update(camera_id, **fields)
        if cam is None:
            return _err(f"Câmera {camera_id} não encontrada.", 404)
        masked = mask_url(cam.rtsp_url)
    # Reinicia a captura se a URL mudou
    if "rtsp_url" in fields or "enabled" in fields:
        _cameras().remove(camera_id)
        if fields.get("enabled", True):
            _cameras().add(camera_id, url=fields.get("rtsp_url", ""), name=fields.get("name", ""))
    return _ok(camera={"id": camera_id, "rtsp_url": masked})


@router.delete("/cameras/{camera_id}", response_model=None)
def api_delete_camera(camera_id: int) -> dict[str, Any] | JSONResponse:
    _cameras().remove(camera_id)
    with session_scope() as session:
        if not CameraRepository(session).delete(camera_id):
            return _err(f"Câmera {camera_id} não encontrada.", 404)
    return _ok(message=f"Câmera {camera_id} removida.")


# ================================================================= CONTAGEM ===
@router.get("/count", response_model=None)
def api_count() -> dict[str, Any]:
    """Contagem atual por pilha + total."""
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
        value = int(value)
    except (TypeError, ValueError):
        return _err("A contagem deve ser um número inteiro.")
    result = counting_service.apply_manual_count(_inference(), pile_id, value)
    if not result.get("ok"):
        return _err(result.get("message", "Falha ao corrigir."))
    state_store.push_event(
        "INFO", "count_corrected", f"Pilha {pile_id}: IA={result['ai_count']} -> {value}"
    )
    return _ok(**result, message=f"Pilha {pile_id} corrigida para {value} cadeiras.")


@router.post("/count/{pile_id}/release", response_model=None)
def api_release_count(pile_id: int) -> dict[str, Any]:
    """Devolve a pilha ao controle da IA após uma correção manual."""
    result = counting_service.release_manual_count(_inference(), pile_id)
    if not result.get("ok"):
        return _err(result.get("message", "Falha."))
    return _ok(**result)


@router.get("/piles", response_model=None)
def api_piles(camera_id: int | None = None, limit: int = 200) -> dict[str, Any]:
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
    from app.database.models import Pile

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
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    with session_scope() as session:
        repo = CountRepository(session)
        rows = repo.history(camera_id=camera_id, since=since, pile_id=pile_id, limit=limit)
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
        if camera_id is not None:
            series = repo.totals_series(camera_id, since)
            total_series = [
                {"timestamp": ts.isoformat(timespec="seconds"), "total": total} for ts, total in series
            ]
        total_now = repo.total_at(camera_id) if camera_id is not None else state_store.total()
    return _ok(records=records, totals_series=total_series, current_total=total_now, hours=hours)


@router.get("/corrections", response_model=None)
def api_corrections(limit: int = 200) -> dict[str, Any]:
    """Correções manuais (dados de análise de erro do modelo)."""
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
        suggestion = calibration_service.offset_suggestion()
    return _ok(corrections=items, stats=stats, offset_suggestion=suggestion)


# =================================================================== MODELOS ===
@router.get("/models", response_model=None)
def api_models() -> dict[str, Any]:
    detector = _detector()
    models = _models()
    return _ok(
        active=models.get_active_info(),
        loaded=detector.info(),
        models=models.list_models(),
    )


@router.post("/models/activate", response_model=None)
def api_activate_model(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Ativa um modelo em produção (hot-swap, sem derrubar o pipeline)."""
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
        state_store.set_status(model_loaded=True, model_filename=result.get("filename"))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/models/rollback", response_model=None)
def api_rollback_model() -> dict[str, Any]:
    models = _models()
    result = models.rollback()
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/models/import", response_model=None)
async def api_import_model(file: UploadFile = File(...)) -> dict[str, Any]:
    """Importa um .pt treinado para ``data/models``."""
    models = _models()
    name = Path(file.filename or "model.pt").name
    if not name.endswith(".pt"):
        return _err("O arquivo precisa ter extensão .pt")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    target = settings.models_dir / name
    try:
        with target.open("wb") as fh:
            while chunk := file.read(1024 * 1024):
                fh.write(chunk)
    except OSError as exc:
        return _err(f"Falha ao gravar o arquivo: {exc}")
    result = models.import_model(target)
    if result.get("ok"):
        state_store.push_event("INFO", "model_imported", result.get("message", "Modelo importado."))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.post("/models/register-run", response_model=None)
def api_register_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Registra o ``best.pt`` de um run existente (``training/runs/<nome>``)."""
    run = str(payload.get("run_dir", "")).strip()
    if not run:
        return _err("Informe o run_dir (ex.: training/runs/chair_counter).")
    path = Path(run)
    if not path.is_absolute():
        path = settings.base_dir / path
    if not path.is_dir():
        return _err(f"Pasta de run não encontrada: {path}")
    models = _models()
    result = models.register_trained(path, make_active=bool(payload.get("activate", False)))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


# =================================================================== DATASET ===
@router.get("/dataset", response_model=None)
def api_dataset() -> dict[str, Any]:
    ds = DatasetService()
    stats = ds.stats()
    return _ok(stats=stats.as_dict(), yaml=str(ds.write_yaml()), root=str(ds.root))


@router.get("/dataset/images", response_model=None)
def api_dataset_images(limit: int = 200) -> dict[str, Any]:
    return _ok(**DatasetService().image_status(limit=limit))


@router.get("/dataset/image", response_model=None)
def api_dataset_image(path: str = Query(...)) -> Any:
    """Serve uma imagem do dataset (path relativo ao dataset root)."""
    ds = DatasetService()
    target = (ds.root / path).resolve()
    try:
        target.relative_to(ds.root.resolve())
    except ValueError:
        return _err("Caminho inválido.", 400)
    if not target.is_file():
        return _err("Imagem não encontrada.", 404)
    return FileResponse(target)


@router.get("/dataset/label", response_model=None)
def api_dataset_label(path: str = Query(...)) -> Any:
    ds = DatasetService()
    target = (ds.root / path).resolve()
    try:
        target.relative_to(ds.root.resolve())
    except ValueError:
        return _err("Caminho inválido.", 400)
    if not target.is_file():
        return _err("Rótulo não encontrado.", 404)
    return FileResponse(target, media_type="text/plain")


@router.post("/dataset/upload", response_model=None)
async def api_dataset_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    ds = DatasetService()
    name = Path(file.filename or "imagem.jpg").name
    tmp = settings.source_images_dir / name
    ds.ensure_structure()
    try:
        with tmp.open("wb") as fh:
            while chunk := file.read(1024 * 1024):
                fh.write(chunk)
    except OSError as exc:
        return _err(f"Falha ao gravar: {exc}")
    result = ds.add_image(tmp, copy=False)
    state_store.push_event("INFO", "dataset_upload", f"Imagem adicionada: {name}")
    return _ok(**result, message=f"Imagem {name} adicionada. Agora anote-a.")


@router.post("/dataset/split", response_model=None)
def api_dataset_split(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Divide o dataset em treino/validação/teste com seed reproduzível."""
    ds = DatasetService()
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
    """Pré-rotula imagens sem anotação usando o modelo ativo."""
    ds = DatasetService()
    result = ds.auto_label_from_model(conf=float(payload.get("conf", 0.35)))
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.get("/annotation", response_model=None)
def api_annotation_page(limit: int = 200) -> dict[str, Any]:
    """Dados da página /annotation."""
    ds = DatasetService()
    ds.ensure_structure()
    return _ok(**ds.image_status(limit=limit))


# ================================================================ TREINAMENTO ===
@router.get("/training/status", response_model=None)
def api_training_status() -> dict[str, Any]:
    training_service.check_finished()
    return _ok(**training_service.status())


@router.post("/training/start", response_model=None)
def api_training_start(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
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
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha."), 409))


@router.post("/training/stop", response_model=None)
def api_training_stop() -> dict[str, Any]:
    result = training_service.stop()
    return (_ok(**result) if result.get("ok") else _err(result.get("message", "Falha.")))


@router.get("/training/log", response_model=None)
def api_training_log(lines: int = 80) -> dict[str, Any]:
    return _ok(log=training_service.tail_log(lines))


# =============================================================== CALIBRAÇÃO ===
@router.get("/calibration", response_model=None)
def api_calibration(camera_id: int | None = None) -> dict[str, Any]:
    cid = _camera_id(camera_id)
    data = calibration_service.load(cid)
    return _ok(calibration=data.as_dict(), suggestion=calibration_service.offset_suggestion())


@router.post("/calibration", response_model=None)
def api_save_calibration(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    cid = _camera_id(payload.get("camera_id"))
    data = calibration_service.save(cid, payload)
    # Aplica imediatamente no worker, sem reiniciar a câmera
    if _inference():
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
    state_store.push_event("INFO", "calibration_saved", f"Calibração da câmera {cid} atualizada")
    return _ok(calibration=data.as_dict(), message="Calibração salva e aplicada.")


@router.post("/calibration/reset", response_model=None)
def api_reset_calibration(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    cid = _camera_id(payload.get("camera_id"))
    data = calibration_service.reset(cid)
    if _inference():
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
    result = counting_service.measure_chair_height(state_store.get_frame_bgr(), payload.get("roi") or {})
    return _ok(**result) if result.get("ok") else _err(result.get("message", "Falha."), 503)


# =================================================================== EVENTOS ===
@router.get("/events", response_model=None)
def api_events(limit: int = 200, level: str | None = None) -> dict[str, Any]:
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
    from app.database.models import SystemEvent

    with session_scope() as session:
        rows = session.execute(select(SystemEvent)).scalars().all()
        for row in rows:
            session.delete(row)
    return _ok(message=f"{len(rows)} evento(s) removido(s).")


# ================================================================= SNAPSHOTS ===
@router.get("/snapshots", response_model=None)
def api_snapshots(limit: int = 100) -> dict[str, Any]:
    with session_scope() as session:
        rows = SnapshotRepository(session).list(limit=limit)
        items = [
            {
                "id": s.id,
                "timestamp": s.timestamp.isoformat(timespec="seconds"),
                "filename": s.filename,
                "reason": s.reason,
                "total": s.total,
                "pile_count": s.pile_count,
                "confidence": round(s.confidence, 4) if s.confidence is not None else None,
                "size_kb": round((s.snapshot_size or 0) / 1024, 1),
                "url": f"/api/snapshots/file/{s.filename}",
            }
            for s in rows
        ]
    return _ok(snapshots=items, directory=str(settings.snapshots_dir))


@router.get("/snapshots/file/{filename}", response_model=None)
def api_snapshot_file(filename: str) -> Any:
    svc = _inference().snapshots if _inference() else None
    path = svc.resolve(filename) if svc else None
    if path is None:
        return _err("Snapshot não encontrado.", 404)
    return FileResponse(path, media_type="image/jpeg")


@router.post("/snapshots", response_model=None)
def api_create_snapshot(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Salva um snapshot manualmente (botão do dashboard)."""
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
        return _err("Não foi possível salvar o snapshot.", 500)
    state_store.push_event("INFO", "snapshot_manual", f"Snapshot manual: {res['filename']}")
    return _ok(**res, message="Snapshot salvo.")


@router.delete("/snapshots/{snapshot_id}", response_model=None)
def api_delete_snapshot(snapshot_id: int) -> dict[str, Any]:
    with session_scope() as session:
        repo = SnapshotRepository(session)
        row = repo.get(snapshot_id)
        if row is None:
            return _err("Snapshot não encontrado.", 404)
        path = settings.snapshots_dir / row.filename
        repo.delete(snapshot_id)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    return _ok(message="Snapshot removido.")


# ==================================================================== STREAM ===
@router.get("/stream.mjpg", response_model=None)
def api_stream_mjpg() -> StreamingResponse:
    """Stream MJPEG do frame anotado (usado pelo <img> do dashboard)."""

    def generate():  # type: ignore[no-untyped-def]
        last = -1
        idle = 0
        while idle < 100:  # ~20s sem frame: encerra
            jpeg, counter = state_store.get_frame_jpeg(last)
            if jpeg is None:
                time.sleep(1.0 / max(1.0, settings.web.stream_fps))
                idle += 1
                continue
            idle = 0
            last = counter
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store\r\n\r\n" + jpeg + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/frame.jpg", response_model=None)
def api_frame_jpeg() -> Any:
    """Frame anotado atual (para download)."""
    jpeg, _ = state_store.get_frame_jpeg(-1)
    if jpeg is None:
        return _err("Nenhum frame disponível.", 404)
    from fastapi.responses import Response

    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/frame.raw.jpg", response_model=None)
def api_frame_raw() -> Any:
    """Frame **sem** overlay, para conferir a imagem original."""
    frame = state_store.get_frame_bgr()
    if frame is None:
        return _err("Nenhum frame disponível.", 404)
    from fastapi.responses import Response

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
        stamp = time.strftime("%Y%m%d_%H%M%S")
        roi = result["roi"]
        path = out / f"roi_{stamp}_{roi['w']}x{roi['h']}.jpg"
        frame = state_store.get_frame_bgr()
        if frame is not None:
            cv2.imwrite(str(path), frame[roi["y"] : roi["y"] + roi["h"], roi["x"] : roi["x"] + roi["w"]])
            img_url = f"/api/diagnostics/image?path={path.name}"
    except Exception:
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
    """Configuração efetiva (sem segredos)."""
    return _ok(
        camera={
            "name": settings.camera.camera_name,
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
        database=settings.web.database_url.split("@")[-1],
        host=settings.web.host,
        port=settings.web.port,
    )


@router.get("/logs", response_model=None)
def api_logs(file: str = Query(default="app.log"), lines: int = 120) -> dict[str, Any]:
    """Últimas linhas de um dos 4 arquivos de log."""
    name = Path(file).name
    allowed = {"app.log", "camera.log", "ai.log", "training.log", "training_latest.log"}
    if name not in allowed:
        return _err(f"Arquivo de log não permitido. Use: {sorted(allowed)}")
    path = settings.logs_dir / name
    if not path.is_file():
        return _ok(log="", message=f"{name} ainda não existe.")
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return _ok(log="\n".join(content.splitlines()[-lines:]), file=name)
    except OSError as exc:
        return _err(str(exc), 500)


# Injetado em main.py para evitar import circular
app_state: Any = None


__all__ = ["router", "app_state"]
