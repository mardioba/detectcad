"""Gerenciamento de versões de modelo.

Responsabilidades:

* listar os ``.pt`` em ``data/models/`` e sincronizar com a tabela
  ``model_versions``;
* registrar as métricas reais do treino (lidas de ``results.csv`` do run);
* ativar uma versão em produção com hot-swap (sem derrubar o pipeline) e
  possibilidade de voltar à anterior;
* manter a última versão boa para rollback automático.
"""

# ARQUIVO / MAPA
# O que faz: cuida das VERSÕES de modelo. O disco (data/models/*.pt) é a
# fonte da verdade dos arquivos; o banco (tabela model_versions) guarda
# métricas e qual está ativo. Este arquivo reconcilia os dois.
# Nada aqui inventa métrica: se não há results.csv do treino, os campos de
# precisão/mAP ficam vazios e a tela mostra "sem métricas".
# Ordem de leitura:
#   1. _VERSION_RE / parse_version_from_name - tira a versão do nome do .pt
#   2. read_metrics_from_run                  - métricas reais do treino
#   3. ModelManager.__init__                   - detector + estado do swap
#   4. scan_disk / sync_to_db                 - disco -> banco
#   5. list_models / get_active_info          - leitura para a página /modelo
#   6. activate / rollback / _resolve_target  - hot-swap em produção
#   7. import_model / register_trained        - trazer .pt novo para o disco

from __future__ import annotations

import logging
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select

from app.ai.yolo_detector import YoloDetector
from app.config import settings
from app.database.database import session_scope
from app.database.repository import EventRepository, ModelRepository

log = logging.getLogger("ai")

_VERSION_RE = re.compile(r"^(?P<stem>.+?)[-_]?v(?P<ver>\d+(?:\.\d+)*)$", re.IGNORECASE)


def parse_version_from_name(filename: str) -> tuple[str, str]:
    """``chair_counter_v3.pt`` -> ``("chair_counter", "3")``.

    O nome do arquivo é a única etiqueta de versão que existe antes do
    treino terminar. Sem sufixo ``vN`` cai em ``"1.0"`` para o registro
    continuar funcionando, em vez de falhar.
    """
    stem = Path(filename).stem
    m = _VERSION_RE.match(stem)
    if m:
        return m.group("stem"), m.group("ver")
    return stem, "1.0"


def read_metrics_from_run(run_dir: Path) -> dict[str, Any]:
    """Lê as métricas reais de um run do Ultralytics.

    Fonte: ``results.csv`` (colunas do YOLO) e o ``args.yaml``.
    Se não existirem, devolve ``None`` - nunca inventa número.
    """
    out: dict[str, Any] = {}
    csv_path = run_dir / "results.csv"
    if csv_path.is_file():
        try:
            import csv as _csv

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
            if rows:
                # O YOLO grava uma linha por época; a última é o modelo final.
                last = rows[-1]
                # Normaliza cabeçalhos: "metrics/precision(B)" -> "precision"
                # Procura por prefixo porque o sufixo ((B), (C), etc.) muda
                # entre versões do Ultralytics.
                def _get(prefix: str) -> float | None:
                    target = prefix.strip().lower()
                    for key, val in last.items():
                        if key and key.strip().lower().startswith(target):
                            try:
                                return float(val)
                            except (TypeError, ValueError):
                                # Célula vazia ("") acontece quando o treino
                                # foi interrompido: melhor None que 0.0.
                                return None
                    return None

                out["precision"] = _get("metrics/precision")
                out["recall"] = _get("metrics/recall")
                # mAP50 é prefixo de mAP50-95, por isso a ordem das buscas
                # importa: o mais específico ("mAP50(B)") vem primeiro.
                out["map50"] = _get("metrics/mAP50(B)") or _get("metrics/mAP50")
                out["map5095"] = _get("metrics/mAP50-95(B)") or _get("metrics/mAP50-95")
                out["epochs"] = len(rows)
        except Exception as exc:
            # Métrica ausente é normal (treino manual, pasta copiada).
            # Falha aqui não pode impedir a sincronia do modelo.
            log.debug("Não foi possível ler results.csv: %s", exc)

    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        try:
            import yaml

            with args_yaml.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            # args.yaml registra COMO o treino foi feito: base, imgsz, batch.
            out["base_model"] = str(data.get("model", "")) or None
            out["image_size"] = data.get("imgsz")
            out["batch"] = data.get("batch")
            out["classes"] = None
        except Exception:
            pass
    return out


class ModelManager:
    """Sincroniza disco x banco e faz o hot-swap do modelo ativo.

    O hot-swap é sempre em duas etapas: primeiro a memória (o
    ``YoloDetector`` troca o modelo, com rollback próprio) e só depois o
    banco. Assim o banco nunca diz "ativo" para um modelo que não
    conseguiu carregar.
    """

    def __init__(self, detector: YoloDetector) -> None:
        self.detector = detector
        # Serializa activate() e rollback(): duas trocas simultâneas
        # deixariam _previous apontando para o modelo errado.
        self._lock = threading.RLock()
        # Caminho do modelo anterior, para o rollback. Fica como texto
        # porque o arquivo pode sumir do disco e a checagem é na hora.
        self._previous: str | None = None
        self._swapping = False

    # ------------------------------------------------------------ sincronia
    def scan_disk(self) -> list[Path]:
        """Todos os ``.pt`` disponíveis em ``data/models``.

        Ordenado do mais novo para o mais antigo: na tela, o modelo
        recém-treinado aparece primeiro.
        """
        models_dir = settings.models_dir
        if not models_dir.is_dir():
            return []
        files = sorted(
            (p for p in models_dir.glob("*.pt") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files

    def sync_to_db(self) -> list[dict[str, Any]]:
        """Registra no banco os modelos encontrados no disco (sem deletar).

        Só faz upsert: modelo que saiu do disco continua no histórico, com
        a página mostrando que não existe mais lá.
        """
        found: list[dict[str, Any]] = []
        with session_scope() as session:
            repo = ModelRepository(session)
            active = repo.get_active()
            active_filename = active.filename if active else None
            # Se o banco não tem ativo mas o .env aponta para um .pt existente,
            # esse é o modelo ativo na prática.
            configured = settings.model_path.name
            for path in self.scan_disk():
                stem, version = parse_version_from_name(path.name)
                # As métricas vêm do run de treino com o MESMO nome-base.
                # Sem essa pasta, o modelo entra no banco sem métricas.
                metrics_path = settings.base_dir / "training" / "runs" / stem
                metrics = read_metrics_from_run(metrics_path) if metrics_path.is_dir() else {}
                repo.upsert(
                    filename=path.name,
                    version=version,
                    source_path=str(path),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    map50=metrics.get("map50"),
                    map5095=metrics.get("map5095"),
                    base_model=metrics.get("base_model"),
                    epochs=metrics.get("epochs"),
                    notes=metrics.get("notes"),
                )
                found.append(
                    {
                        "filename": path.name,
                        "version": version,
                        "path": str(path),
                        "size_mb": round(path.stat().st_size / (1024**2), 2),
                    }
                )
            # Banco sem nenhum ativo: adota o que já está rodando de fato
            # (o do banco ou o do .env) para a página não mostrar vazio.
            if repo.get_active() is None and (active_filename or configured):
                target = active_filename or configured
                row = repo.get_by_filename(target)
                if row is not None:
                    row.active = True
        return found

    def list_models(self) -> list[dict[str, Any]]:
        """Modelos do banco + métricas, marcando o ativo.

        Sempre sincroniza antes de ler: a tela deve refletir o que está
        no disco agora, não o que estava no último deploy.
        """
        self.sync_to_db()
        with session_scope() as session:
            repo = ModelRepository(session)
            rows = repo.list()
            active = repo.get_active()
            active_name = active.filename if active else None
            # Índice por nome: sem isso, cada linha buscaria o arquivo no
            # disco com uma lista (O(n²) na quantidade de modelos).
            disk = {p.name: p for p in self.scan_disk()}
            out: list[dict[str, Any]] = []
            for row in rows:
                info = {
                    "id": row.id,
                    "filename": row.filename,
                    "version": row.version,
                    "created_at": row.created_at.isoformat(timespec="seconds") if row.created_at else None,
                    "precision": row.precision,
                    "recall": row.recall,
                    "map50": row.map50,
                    "map5095": row.map5095,
                    "active": bool(row.active),
                    "notes": row.notes,
                    "base_model": row.base_model,
                    "epochs": row.epochs,
                    "dataset_size": row.dataset_size,
                    # exists_on_disk separa "registrado" de "instalado".
                    "exists_on_disk": row.filename in disk,
                    "size_mb": round(disk[row.filename].stat().st_size / (1024**2), 2)
                    if row.filename in disk
                    else None,
                }
                out.append(info)
            # Banco zerado (primeira execução): mostra pelo menos o modelo
            # do .env, marcado como "não registrado".
            if not out:
                out = [
                    {
                        "id": None,
                        "filename": configured_name,
                        "version": "não registrado",
                        "active": configured_name == active_name,
                        "exists_on_disk": (settings.models_dir / configured_name).is_file(),
                        "metrics_available": False,
                    }
                    for configured_name in [settings.model_path.name]
                ]
            return out

    def get_active_info(self) -> dict[str, Any]:
        """Modelo ativo e as métricas dele (para o cabeçalho e o dashboard).

        Sem registro no banco, a mensagem distingue os dois motivos
        possíveis: modelo nem carregado (falta treinar) ou carregado e
        apenas não registrado ainda.
        """
        with session_scope() as session:
            repo = ModelRepository(session)
            row = repo.get_active()
            if row is None:
                return {
                    "active": False,
                    "filename": settings.model_path.name,
                    "loaded": self.detector.is_loaded,
                    "message": (
                        "Modelo específico ainda não treinado."
                        if not self.detector.is_loaded
                        else "Modelo carregado, ainda não registrado no banco."
                    ),
                }
            return {
                "active": True,
                "id": row.id,
                "filename": row.filename,
                "version": row.version,
                "created_at": row.created_at.isoformat(timespec="seconds") if row.created_at else None,
                "precision": row.precision,
                "recall": row.recall,
                "map50": row.map50,
                "map5095": row.map5095,
                "notes": row.notes,
                "epochs": row.epochs,
                "dataset_size": row.dataset_size,
                "base_model": row.base_model,
            }

    # ------------------------------------------------------------- hot-swap
    def activate(self, model_id: int | None = None, filename: str | None = None) -> dict[str, Any]:
        """Ativa um modelo em produção. Mantém o anterior para rollback.

        Devolve ``{"ok": bool, "message": str, ...}`` em vez de levantar
        exceção: a chamada vem de um botão na tela e a resposta precisa
        ser mostrada ao operador.
        """
        with self._lock:
            # _swapping é a trava lógica: o lock sozinho não impediria uma
            # segunda troca vinda de outra thread que chegou depois.
            if self._swapping:
                return {"ok": False, "message": "Já há uma troca de modelo em andamento."}
            self._swapping = True
            try:
                target = self._resolve_target(model_id, filename)
                if target is None:
                    return {"ok": False, "message": "Modelo não encontrado."}
                path, row = target
                if not path.is_file():
                    return {"ok": False, "message": f"Arquivo ausente: {path}"}

                # Guarda o anterior ANTES da troca: se o modelo novo for
                # ruim em produção, é para este caminho que o rollback
                # aponta. Só grava se o caminho realmente muda.
                previous_path = None
                if self.detector.is_loaded and self.detector.configured_path != path:
                    previous_path = str(self.detector.configured_path)
                    self._previous = previous_path

                # hot_swap já devolve o modelo antigo se este falhar.
                ok, error = self.detector.hot_swap(path)
                if not ok:
                    return {"ok": False, "message": f"Falha ao ativar o modelo: {error}"}

                # Só agora (modelo já em memória) o banco é atualizado.
                with session_scope() as session:
                    repo = ModelRepository(session)
                    if row is None:
                        # Ativando por nome e sem registro: cria a linha
                        # agora, com a versão deduzida do próprio arquivo.
                        stem, version = parse_version_from_name(path.name)
                        row = repo.upsert(filename=path.name, version=version, source_path=str(path))
                    repo.set_active(row.id)
                    # Log de auditoria: troca de modelo muda a contagem de
                    # todas as pilhas e precisa ficar registrado.
                    EventRepository(session).log(
                        "INFO",
                        "model_activated",
                        f"Modelo {path.name} ativado em produção"
                        + (f" (anterior: {Path(previous_path).name})" if previous_path else ""),
                    )
                return {
                    "ok": True,
                    "filename": path.name,
                    "previous": Path(previous_path).name if previous_path else None,
                    "message": f"Modelo {path.name} ativado.",
                }
            finally:
                # O finally garante que uma exceção inesperada não deixe o
                # gerenciador travado em "troca em andamento" para sempre.
                self._swapping = False

    def rollback(self) -> dict[str, Any]:
        """Volta para o modelo anterior.

        É um activate() para trás: mesmo caminho de troca, mas o alvo vem
        do ``_previous`` guardado na ativação anterior.
        """
        with self._lock:
            if not self._previous:
                return {"ok": False, "message": "Nenhum modelo anterior disponível."}
            prev = Path(self._previous)
            if not prev.is_file():
                # Arquivo apagado entre a ativação e agora: melhor avisar
                # do que tentar carregar e falhar no meio.
                return {"ok": False, "message": f"Arquivo anterior não encontrado: {prev}"}
            ok, error = self.detector.hot_swap(prev)
            if not ok:
                return {"ok": False, "message": f"Falha no rollback: {error}"}
            with session_scope() as session:
                repo = ModelRepository(session)
                row = repo.get_by_filename(prev.name)
                if row is not None:
                    repo.set_active(row.id)
            # Guarda o modelo que acabamos de dejar como "anterior", para
            # dar rollback duas vezes funcionar.
            self._previous = str(self.detector.configured_path)
            return {"ok": True, "filename": prev.name, "message": f"Rollback para {prev.name}."}

    def _resolve_target(
        self, model_id: int | None, filename: str | None
    ) -> tuple[Path, Any] | None:
        """Traduz id ou nome em (caminho, linha do banco). ``None`` = não achou.

        A sessão é fechada ao devolver: o resto do fluxo (cópia de arquivo,
        hot-swap) não pode ficar segurando transação do banco.
        """
        with session_scope() as session:
            repo = ModelRepository(session)
            if model_id is not None:
                row = repo.get(model_id)
                if row is None:
                    return None
                # source_path é o caminho completo gravado no registro; cai
                # no models_dir se o campo estiver vazio.
                path = Path(row.source_path) if row.source_path else settings.models_dir / row.filename
            elif filename:
                row = repo.get_by_filename(filename)
                path = settings.models_dir / filename
                if row is None:
                    # Arquivo existe no disco mas não está registrado:
                    # devolve (caminho, None) e activate() cria a linha.
                    stem, version = parse_version_from_name(filename)
                    return path, None
            else:
                return None
        return path, row

    # ------------------------------------------------------------- importação
    def import_model(self, source: Path, version: str | None = None) -> dict[str, Any]:
        """Copia um ``.pt`` (de um treino ou baixado) para ``data/models``."""
        source = Path(source)
        if not source.is_file():
            return {"ok": False, "message": f"Arquivo não encontrado: {source}"}
        if source.suffix != ".pt":
            # Só .pt: o YoloDetector não sabe abrir .onnx nem .engine.
            return {"ok": False, "message": "O arquivo precisa ter extensão .pt"}
        target = settings.models_dir / source.name
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        # copy2 (e não copy) preserva mtime: o scan_disk ordena por data, e a
        # data original do arquivo é mais informativa que a da cópia.
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        stem, auto_version = parse_version_from_name(target.name)
        with session_scope() as session:
            repo = ModelRepository(session)
            repo.upsert(
                filename=target.name,
                version=version or auto_version,
                source_path=str(target),
            )
            EventRepository(session).log("INFO", "model_imported", f"Modelo importado: {target.name}")
        # Importar NÃO ativa: quem quis ativar usa activate() logo em
        # seguida, para a troca em produção ser explícita.
        return {"ok": True, "filename": target.name, "message": f"Modelo {target.name} importado."}

    def register_trained(
        self,
        run_dir: Path,
        *,
        make_active: bool = False,
        dataset_size: int | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Registra o ``best.pt`` de um run do Ultralytics.

        Chamado pelo serviço de treinamento ao terminar, ou manualmente.

        Usa o ``best.pt`` (melhor época do treino), não o ``last.pt``:
        é esse arquivo que evita overfitting.
        """
        run_dir = Path(run_dir)
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            return {"ok": False, "message": f"best.pt não encontrado em {run_dir}"}
        # O .pt sai da pasta do run para data/models: é o único diretório
        # que o YoloDetector e o hot_swap olham.
        target = settings.models_dir / best.name
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)

        metrics = read_metrics_from_run(run_dir)
        stem, version = parse_version_from_name(target.name)
        with session_scope() as session:
            repo = ModelRepository(session)
            # upsert: retreinar o mesmo nome atualiza a linha em vez de
            # duplicar, preservando o histórico de eventos.
            row = repo.upsert(
                filename=target.name,
                version=version,
                source_path=str(target),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                map50=metrics.get("map50"),
                map5095=metrics.get("map5095"),
                base_model=metrics.get("base_model"),
                epochs=metrics.get("epochs"),
                dataset_size=dataset_size,
                notes=notes,
            )
            if make_active:
                repo.set_active(row.id)
            # O log de evento só acrescenta as métricas se existirem: um
            # treino interrompido gera evento sem P/R, sem número falso.
            EventRepository(session).log(
                "INFO",
                "model_registered",
                f"Modelo {target.name} registrado"
                + (
                    f" | P={metrics.get('precision')} R={metrics.get('recall')} "
                    f"mAP50={metrics.get('map50')} mAP50-95={metrics.get('map5095')}"
                    if metrics.get("precision") is not None
                    else ""
                ),
            )
        return {
            "ok": True,
            "filename": target.name,
            "metrics": metrics,
            "message": f"Modelo {target.name} registrado no banco.",
        }


# Nome do modelo do .env, resolvido na importação. Fica disponível para
# chamadores que só precisam do nome sem abrir o settings de novo.
configured_name = settings.model_path.name


__all__ = ["ModelManager", "parse_version_from_name", "read_metrics_from_run"]
