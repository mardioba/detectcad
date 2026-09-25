"""Gerenciamento de versões de modelo.

Responsabilidades:

* listar os ``.pt`` em ``data/models/`` e sincronizar com a tabela
  ``model_versions``;
* registrar as métricas reais do treino (lidas de ``results.csv`` do run);
* ativar uma versão em produção com hot-swap (sem derrubar o pipeline) e
  possibilidade de voltar à anterior;
* manter a última versão boa para rollback automático.
"""

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
    """``chair_counter_v3.pt`` -> ``("chair_counter", "3")``."""
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
                last = rows[-1]
                # Normaliza cabeçalhos: "metrics/precision(B)" -> "precision"
                def _get(prefix: str) -> float | None:
                    target = prefix.strip().lower()
                    for key, val in last.items():
                        if key and key.strip().lower().startswith(target):
                            try:
                                return float(val)
                            except (TypeError, ValueError):
                                return None
                    return None

                out["precision"] = _get("metrics/precision")
                out["recall"] = _get("metrics/recall")
                out["map50"] = _get("metrics/mAP50(B)") or _get("metrics/mAP50")
                out["map5095"] = _get("metrics/mAP50-95(B)") or _get("metrics/mAP50-95")
                out["epochs"] = len(rows)
        except Exception as exc:
            log.debug("Não foi possível ler results.csv: %s", exc)

    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        try:
            import yaml

            with args_yaml.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            out["base_model"] = str(data.get("model", "")) or None
            out["image_size"] = data.get("imgsz")
            out["batch"] = data.get("batch")
            out["classes"] = None
        except Exception:
            pass
    return out


class ModelManager:
    """Sincroniza disco x banco e faz o hot-swap do modelo ativo."""

    def __init__(self, detector: YoloDetector) -> None:
        self.detector = detector
        self._lock = threading.RLock()
        self._previous: str | None = None
        self._swapping = False

    # ------------------------------------------------------------ sincronia
    def scan_disk(self) -> list[Path]:
        """Todos os ``.pt`` disponíveis em ``data/models``."""
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
        """Registra no banco os modelos encontrados no disco (sem deletar)."""
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
            if repo.get_active() is None and (active_filename or configured):
                target = active_filename or configured
                row = repo.get_by_filename(target)
                if row is not None:
                    row.active = True
        return found

    def list_models(self) -> list[dict[str, Any]]:
        """Modelos do banco + métricas, marcando o ativo."""
        self.sync_to_db()
        with session_scope() as session:
            repo = ModelRepository(session)
            rows = repo.list()
            active = repo.get_active()
            active_name = active.filename if active else None
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
                    "exists_on_disk": row.filename in disk,
                    "size_mb": round(disk[row.filename].stat().st_size / (1024**2), 2)
                    if row.filename in disk
                    else None,
                }
                out.append(info)
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
        """Ativa um modelo em produção. Mantém o anterior para rollback."""
        with self._lock:
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

                previous_path = None
                if self.detector.is_loaded and self.detector.configured_path != path:
                    previous_path = str(self.detector.configured_path)
                    self._previous = previous_path

                ok, error = self.detector.hot_swap(path)
                if not ok:
                    return {"ok": False, "message": f"Falha ao ativar o modelo: {error}"}

                with session_scope() as session:
                    repo = ModelRepository(session)
                    if row is None:
                        stem, version = parse_version_from_name(path.name)
                        row = repo.upsert(filename=path.name, version=version, source_path=str(path))
                    repo.set_active(row.id)
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
                self._swapping = False

    def rollback(self) -> dict[str, Any]:
        """Volta para o modelo anterior."""
        with self._lock:
            if not self._previous:
                return {"ok": False, "message": "Nenhum modelo anterior disponível."}
            prev = Path(self._previous)
            if not prev.is_file():
                return {"ok": False, "message": f"Arquivo anterior não encontrado: {prev}"}
            ok, error = self.detector.hot_swap(prev)
            if not ok:
                return {"ok": False, "message": f"Falha no rollback: {error}"}
            with session_scope() as session:
                repo = ModelRepository(session)
                row = repo.get_by_filename(prev.name)
                if row is not None:
                    repo.set_active(row.id)
            self._previous = str(self.detector.configured_path)
            return {"ok": True, "filename": prev.name, "message": f"Rollback para {prev.name}."}

    def _resolve_target(
        self, model_id: int | None, filename: str | None
    ) -> tuple[Path, Any] | None:
        with session_scope() as session:
            repo = ModelRepository(session)
            if model_id is not None:
                row = repo.get(model_id)
                if row is None:
                    return None
                path = Path(row.source_path) if row.source_path else settings.models_dir / row.filename
            elif filename:
                row = repo.get_by_filename(filename)
                path = settings.models_dir / filename
                if row is None:
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
            return {"ok": False, "message": "O arquivo precisa ter extensão .pt"}
        target = settings.models_dir / source.name
        settings.models_dir.mkdir(parents=True, exist_ok=True)
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
        """
        run_dir = Path(run_dir)
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            return {"ok": False, "message": f"best.pt não encontrado em {run_dir}"}
        target = settings.models_dir / best.name
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)

        metrics = read_metrics_from_run(run_dir)
        stem, version = parse_version_from_name(target.name)
        with session_scope() as session:
            repo = ModelRepository(session)
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


configured_name = settings.model_path.name


__all__ = ["ModelManager", "parse_version_from_name", "read_metrics_from_run"]
