"""Camada de acesso a dados.

Cada repositório é uma classe pequena e focada, com métodos de uso claro.
Nenhum SQLAlchemy direto vaza para os serviços.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import (
    Base,
    Calibration,
    Camera,
    CountCorrection,
    CountRecord,
    ModelVersion,
    Pile,
    Snapshot,
    SystemEvent,
    _utcnow,
)
from app.schemas import CountStatus, EventLevel

log = logging.getLogger("app")


# --------------------------------------------------------------------------- #
# Câmeras
# --------------------------------------------------------------------------- #
class CameraRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, camera_id: int) -> Camera | None:
        return self.s.get(Camera, camera_id)

    def get_default(self) -> Camera | None:
        stmt = select(Camera).order_by(Camera.is_default.desc(), Camera.id.asc()).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def list(self, only_enabled: bool = False) -> list[Camera]:
        stmt = select(Camera).order_by(Camera.id.asc())
        if only_enabled:
            stmt = stmt.where(Camera.enabled.is_(True))
        return list(self.s.execute(stmt).scalars())

    def create(self, name: str, rtsp_url: str, enabled: bool = True, is_default: bool = False) -> Camera:
        cam = Camera(name=name, rtsp_url=rtsp_url, enabled=enabled, is_default=is_default)
        self.s.add(cam)
        self.s.flush()
        return cam

    def update(self, camera_id: int, **fields: Any) -> Camera | None:
        cam = self.get(camera_id)
        if cam is None:
            return None
        for key, value in fields.items():
            if value is not None and hasattr(cam, key):
                setattr(cam, key, value)
        self.s.flush()
        return cam

    def delete(self, camera_id: int) -> bool:
        cam = self.get(camera_id)
        if cam is None:
            return False
        self.s.delete(cam)
        self.s.flush()
        return True

    def ensure_from_env(self, name: str, rtsp_url: str) -> Camera:
        """Garante que exista uma câmera padrão vinda do .env."""
        cam = self.get_default()
        if cam is None:
            cam = self.create(name=name, rtsp_url=rtsp_url, is_default=True)
            log.info("Câmera padrão criada a partir do .env: %s", name)
        return cam


# --------------------------------------------------------------------------- #
# Pilhas
# --------------------------------------------------------------------------- #
class PileRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get_by_tracking(self, camera_id: int, tracking_id: int) -> Pile | None:
        stmt = select(Pile).where(Pile.camera_id == camera_id, Pile.tracking_id == tracking_id)
        return self.s.execute(stmt).scalars().first()

    def touch(self, pile: Pile, *, active: bool = True, stable_count: int | None = None) -> Pile:
        pile.last_seen = _utcnow()
        pile.active = active
        if stable_count is not None:
            pile.stable_count = stable_count
        return pile

    def get_or_create(self, camera_id: int, tracking_id: int) -> Pile:
        pile = self.get_by_tracking(camera_id, tracking_id)
        if pile is None:
            pile = Pile(camera_id=camera_id, tracking_id=tracking_id, first_seen=_utcnow(), last_seen=_utcnow())
            self.s.add(pile)
            self.s.flush()
        return pile

    def list_active(self, camera_id: int) -> list[Pile]:
        stmt = select(Pile).where(Pile.camera_id == camera_id, Pile.active.is_(True))
        return list(self.s.execute(stmt).scalars())

    def list(self, camera_id: int | None = None, limit: int = 200) -> list[Pile]:
        stmt = select(Pile).order_by(Pile.last_seen.desc()).limit(limit)
        if camera_id is not None:
            stmt = stmt.where(Pile.camera_id == camera_id)
        return list(self.s.execute(stmt).scalars())

    def deactivate_missing(self, camera_id: int, seen_tracking_ids: Sequence[int]) -> list[int]:
        """Sincroniza o estado ``active`` com o que foi visto neste frame.

        Pilhas não vistas são marcadas inativas; pilhas vistas voltam a
        ficar ativas (uma pilha pode sumir por oclusão e reaparecer com o
        mesmo ``tracking_id``). Retorna os tracking_ids que mudaram.
        """
        seen = set(seen_tracking_ids)
        changed: list[int] = []
        stmt = select(Pile).where(Pile.camera_id == camera_id)
        for pile in self.s.execute(stmt).scalars():
            if pile.tracking_id in seen:
                if not pile.active:
                    pile.active = True
                    changed.append(pile.tracking_id)
            elif pile.active:
                pile.active = False
                changed.append(pile.tracking_id)
        return changed


# --------------------------------------------------------------------------- #
# Contagens
# --------------------------------------------------------------------------- #
class CountRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def add(
        self,
        *,
        camera_id: int,
        pile_id: int | None,
        chair_count: int,
        confidence: float,
        detection_confidence: float = 0.0,
        stability_confidence: float = 0.0,
        status: CountStatus | str = CountStatus.STABLE,
        method: str = "auto",
        total_chairs: int = 0,
        pile_count: int = 1,
        previous_count: int | None = None,
        origin: str = "ai",
    ) -> CountRecord:
        status_value = status.value if isinstance(status, CountStatus) else str(status)
        rec = CountRecord(
            camera_id=camera_id,
            pile_id=pile_id,
            chair_count=int(chair_count),
            confidence=float(confidence),
            detection_confidence=float(detection_confidence),
            stability_confidence=float(stability_confidence),
            status=status_value,
            method=method,
            total_chairs=int(total_chairs),
            pile_count=int(pile_count),
            previous_count=previous_count,
            origin=origin,
        )
        self.s.add(rec)
        self.s.flush()
        return rec

    def last_for_pile(self, pile_id: int) -> CountRecord | None:
        stmt = (
            select(CountRecord)
            .where(CountRecord.pile_id == pile_id, CountRecord.origin == "ai")
            .order_by(CountRecord.timestamp.desc())
            .limit(1)
        )
        return self.s.execute(stmt).scalar_one_or_none()

    def total_at(self, camera_id: int) -> int:
        """Total de cadeiras na leitura mais recente de cada pilha."""
        sub = (
            select(CountRecord.pile_id, func.max(CountRecord.id).label("last_id"))
            .where(CountRecord.camera_id == camera_id, CountRecord.origin == "ai", CountRecord.pile_id.is_not(None))
            .group_by(CountRecord.pile_id)
            .subquery()
        )
        stmt = select(func.coalesce(func.sum(CountRecord.chair_count), 0)).where(
            CountRecord.id.in_(select(sub.c.last_id))
        )
        return int(self.s.execute(stmt).scalar_one() or 0)

    def history(
        self,
        camera_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        pile_id: int | None = None,
        limit: int = 500,
    ) -> list[CountRecord]:
        stmt = select(CountRecord).order_by(CountRecord.timestamp.desc()).limit(limit)
        if camera_id is not None:
            stmt = stmt.where(CountRecord.camera_id == camera_id)
        if since is not None:
            stmt = stmt.where(CountRecord.timestamp >= since)
        if until is not None:
            stmt = stmt.where(CountRecord.timestamp <= until)
        if pile_id is not None:
            stmt = stmt.where(CountRecord.pile_id == pile_id)
        return list(self.s.execute(stmt).scalars())

    def totals_series(self, camera_id: int, since: datetime) -> list[tuple[datetime, int]]:
        """Série (timestamp, total) para o gráfico, reconstruindo o total
        histórico a partir dos deltas de cada pilha.

        ``count_records`` guarda o total no momento da mudança; para o gráfico
        basta interpolar o último total conhecido até o próximo evento.
        """
        recs = self.history(camera_id=camera_id, since=since, limit=100000)
        recs.reverse()
        series: list[tuple[datetime, int]] = []
        for rec in recs:
            if rec.total_chairs and (not series or series[-1][1] != rec.total_chairs):
                series.append((rec.timestamp, int(rec.total_chairs)))
        return series


# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #
class EventRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def log(
        self,
        level: EventLevel | str,
        event: str,
        message: str = "",
        camera_id: int | None = None,
    ) -> SystemEvent:
        lvl = level.value if isinstance(level, EventLevel) else str(level)
        row = SystemEvent(level=lvl, event=event[:120], message=message, camera_id=camera_id)
        self.s.add(row)
        self.s.flush()
        return row

    def list(self, limit: int = 300, level: str | None = None) -> list[SystemEvent]:
        stmt = select(SystemEvent).order_by(SystemEvent.timestamp.desc()).limit(limit)
        if level:
            stmt = stmt.where(SystemEvent.level == level)
        return list(self.s.execute(stmt).scalars())

    def purge(self, older_than_days: int = 30) -> int:
        cutoff = _utcnow() - timedelta(days=older_than_days)
        rows = self.s.execute(select(SystemEvent).where(SystemEvent.timestamp < cutoff)).scalars().all()
        for row in rows:
            self.s.delete(row)
        return len(rows)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
class ModelRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def list(self) -> list[ModelVersion]:
        stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
        return list(self.s.execute(stmt).scalars())

    def get(self, model_id: int) -> ModelVersion | None:
        return self.s.get(ModelVersion, model_id)

    def get_active(self) -> ModelVersion | None:
        stmt = select(ModelVersion).where(ModelVersion.active.is_(True)).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def get_by_filename(self, filename: str) -> ModelVersion | None:
        stmt = select(ModelVersion).where(ModelVersion.filename == filename).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def upsert(
        self,
        *,
        filename: str,
        version: str,
        precision: float | None = None,
        recall: float | None = None,
        map50: float | None = None,
        map5095: float | None = None,
        source_path: str | None = None,
        notes: str | None = None,
        base_model: str | None = None,
        epochs: int | None = None,
        dataset_size: int | None = None,
        classes: str | None = None,
        active: bool = False,
    ) -> ModelVersion:
        row = self.get_by_filename(filename)
        if row is None:
            row = ModelVersion(filename=filename, version=version, created_at=_utcnow())
            self.s.add(row)
        row.version = version
        row.precision = precision
        row.recall = recall
        row.map50 = map50
        row.map5095 = map5095
        row.source_path = source_path
        row.notes = notes
        row.base_model = base_model
        row.epochs = epochs
        row.dataset_size = dataset_size
        row.classes = classes
        if active:
            self.deactivate_all()
            row.active = True
        self.s.flush()
        return row

    def deactivate_all(self) -> None:
        for row in self.s.execute(select(ModelVersion)).scalars():
            row.active = False

    def set_active(self, model_id: int) -> ModelVersion | None:
        row = self.get(model_id)
        if row is None:
            return None
        self.deactivate_all()
        row.active = True
        self.s.flush()
        return row


# --------------------------------------------------------------------------- #
# Calibração
# --------------------------------------------------------------------------- #
class CalibrationRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, camera_id: int) -> Calibration | None:
        stmt = select(Calibration).where(Calibration.camera_id == camera_id).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def upsert(self, camera_id: int, **fields: Any) -> Calibration:
        row = self.get(camera_id)
        if row is None:
            row = Calibration(camera_id=camera_id)
            self.s.add(row)
        for key, value in fields.items():
            if hasattr(row, key):
                setattr(row, key, value)
        row.updated_at = _utcnow()
        self.s.flush()
        return row

    def as_dict(self, camera_id: int) -> dict[str, Any]:
        row = self.get(camera_id)
        if row is None:
            return {}
        data: dict[str, Any] = {
            "enabled": row.enabled,
            "roi": {"x": row.roi_x, "y": row.roi_y, "w": row.roi_w, "h": row.roi_h},
            "chair_height_px": row.chair_height_px,
            "confidence_threshold": row.confidence_threshold,
            "stability_frames": row.stability_frames,
            "min_confidence": row.min_confidence,
            "counting_method": row.counting_method,
            "chair_weight": row.chair_weight,
            "pile_gap_px": row.pile_gap_px,
            "updated_at": row.updated_at.isoformat(timespec="seconds") if row.updated_at else None,
        }
        if row.params_json:
            try:
                data["params"] = json.loads(row.params_json)
            except json.JSONDecodeError:
                data["params"] = {}
        return data


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
class SnapshotRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def add(
        self,
        *,
        filename: str,
        reason: str = "manual",
        camera_id: int | None = None,
        total: int | None = None,
        pile_count: int | None = None,
        confidence: float | None = None,
        snapshot_size: int | None = None,
    ) -> Snapshot:
        row = Snapshot(
            camera_id=camera_id,
            filename=filename,
            reason=reason,
            total=total,
            pile_count=pile_count,
            confidence=confidence,
            snapshot_size=snapshot_size,
        )
        self.s.add(row)
        self.s.flush()
        return row

    def list(self, camera_id: int | None = None, limit: int = 100) -> list[Snapshot]:
        stmt = select(Snapshot).order_by(Snapshot.timestamp.desc()).limit(limit)
        if camera_id is not None:
            stmt = stmt.where(Snapshot.camera_id == camera_id)
        return list(self.s.execute(stmt).scalars())

    def get(self, snapshot_id: int) -> Snapshot | None:
        return self.s.get(Snapshot, snapshot_id)

    def delete(self, snapshot_id: int) -> bool:
        row = self.get(snapshot_id)
        if row is None:
            return False
        self.s.delete(row)
        self.s.flush()
        return True


# --------------------------------------------------------------------------- #
# Correções manuais
# --------------------------------------------------------------------------- #
class CorrectionRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def add(
        self,
        *,
        camera_id: int,
        pile_id: int | None,
        ai_count: int,
        correct_count: int,
        ai_confidence: float = 0.0,
        author: str = "operador",
        note: str | None = None,
        image_path: str | None = None,
    ) -> CountCorrection:
        row = CountCorrection(
            camera_id=camera_id,
            pile_id=pile_id,
            ai_count=ai_count,
            correct_count=correct_count,
            ai_confidence=ai_confidence,
            author=author,
            note=note,
            image_path=image_path,
        )
        self.s.add(row)
        self.s.flush()
        return row

    def list(self, limit: int = 200) -> list[CountCorrection]:
        stmt = select(CountCorrection).order_by(CountCorrection.timestamp.desc()).limit(limit)
        return list(self.s.execute(stmt).scalars())

    def stats(self) -> dict[str, Any]:
        rows = self.list(limit=100000)
        if not rows:
            return {"total": 0, "with_error": 0, "mean_abs_error": 0.0}
        errors = [abs(r.correct_count - r.ai_count) for r in rows]
        return {
            "total": len(rows),
            "with_error": sum(1 for e in errors if e > 0),
            "mean_abs_error": round(sum(errors) / len(errors), 3),
            "exact_match_rate": round(sum(1 for e in errors if e == 0) / len(errors), 4),
        }


__all__ = [
    "CameraRepository",
    "PileRepository",
    "CountRepository",
    "EventRepository",
    "ModelRepository",
    "CalibrationRepository",
    "SnapshotRepository",
    "CorrectionRepository",
]
