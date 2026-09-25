"""Gravação e listagem de snapshots.

Um snapshot é uma imagem JPEG com o overlay das contagens, gravada quando:

* a contagem de uma pilha muda;
* uma pilha aparece ou desaparece;
* a confiança cai abaixo do limite;
* a câmera reconecta;
* o operador pede (botão no dashboard ou ``POST /api/snapshots``);
* um intervalo configurável decorre (``SNAPSHOT_INTERVAL_SEC``).

As imagens são a matéria-prima do dataset: o operador pode copiar as imagens
úteis para ``training/datasets/raw/`` e anotá-las.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.database.database import session_scope
from app.database.repository import SnapshotRepository

log = logging.getLogger("app")

REASONS = {
    "manual": "solicitado pelo operador",
    "count_change": "contagem mudou",
    "pile_appear": "nova pilha",
    "pile_disappear": "pilha desapareceu",
    "low_confidence": "confiança baixa",
    "camera_reconnect": "câmera reconectou",
    "interval": "intervalo periódico",
    "correction": "correção manual de contagem",
}


class SnapshotService:
    """Grava e gerencia snapshots no disco e no banco."""

    def __init__(self, camera_id: int = 1, camera_name: str = "camera1") -> None:
        self.camera_id = camera_id
        self.camera_name = _slug(camera_name)
        self._last_interval = 0.0
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        d = settings.snapshots_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _filename(self, when: datetime, reason: str) -> str:
        stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{stamp}_{self.camera_name}_{reason}.jpg"

    def save(
        self,
        frame: np.ndarray | None,
        *,
        reason: str = "manual",
        total: int | None = None,
        pile_count: int | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any] | None:
        """Salva um frame. Devolve ``None`` se não havia frame."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        when = datetime.now()
        name = self._filename(when, reason)
        path = self.directory / name
        try:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(settings.snapshots.jpeg_quality)]
            ok = cv2.imwrite(str(path), frame, params)
        except Exception as exc:
            log.error("Falha ao gravar snapshot: %s", exc)
            return None
        if not ok:
            log.error("cv2.imwrite falhou para %s", path)
            return None

        size = path.stat().st_size
        try:
            with session_scope() as session:
                SnapshotRepository(session).add(
                    filename=name,
                    reason=reason,
                    camera_id=self.camera_id,
                    total=total,
                    pile_count=pile_count,
                    confidence=confidence,
                    snapshot_size=size,
                )
        except Exception:
            log.exception("Não foi possível registrar o snapshot no banco")

        log.info("Snapshot salvo: %s (%s, %.1f KB)", name, REASONS.get(reason, reason), size / 1024)
        return {
            "filename": name,
            "path": str(path),
            "reason": reason,
            "reason_label": REASONS.get(reason, reason),
            "size_kb": round(size / 1024, 1),
            "timestamp": when.isoformat(timespec="seconds"),
            "url": f"/api/snapshots/file/{name}",
        }

    def maybe_interval(self, frame: np.ndarray | None, **kwargs: Any) -> dict[str, Any] | None:
        """Salva se ``SNAPSHOT_INTERVAL_SEC > 0`` e o intervalo passou."""
        interval = settings.snapshots.interval_sec
        if not settings.snapshots.save_snapshots or interval <= 0 or frame is None:
            return None
        now = time.time()
        with self._lock:
            if now - self._last_interval < interval:
                return None
            self._last_interval = now
        return self.save(frame, reason="interval", **kwargs)

    def resolve(self, filename: str) -> Path | None:
        """Resolve um nome de arquivo dentro do diretório de snapshots.

        Protege contra path traversal (``../``).
        """
        name = Path(filename).name
        path = (self.directory / name).resolve()
        try:
            path.relative_to(self.directory.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    def purge_old(self, keep_days: int | None = None) -> int:
        """Remove snapshots antigos. ``keep_days=0`` (padrão) desativa."""
        days = settings.snapshots.keep_days if keep_days is None else keep_days
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for path in self.directory.glob("*.jpg"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            log.info("%d snapshots antigos removidos", removed)
        return removed


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower() or "camera1"


__all__ = ["SnapshotService", "REASONS"]
