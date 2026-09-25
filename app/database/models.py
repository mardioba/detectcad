"""Tabelas do banco (SQLAlchemy 2.0, anotado por estilo declarativo).

Esquema:

* ``cameras``        - câmeras cadastradas (preparado para multicâmera)
* ``piles``          - pilhas detectadas, com identidade estável
* ``count_records``  - histórico de contagens (gravado só em mudanças reais)
* ``model_versions`` - versões de modelo YOLO e métricas reais de treino
* ``system_events``  - log de eventos do sistema
* ``calibration``    - ROI e parâmetros por câmera
* ``snapshots``      - imagens salvas para auditoria/dataset
* ``count_corrections`` - correções manuais do operador (IA vs. humano)
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Agora em UTC, sem fuso (compatível com o DateTime do SQLite)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Camera(Base):
    """Câmera IP/RTSP cadastrada."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    piles: Mapped[list["Pile"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", passive_deletes=True
    )


class Pile(Base):
    """Pilha identificada. ``tracking_id`` mantém a identidade entre frames."""

    __tablename__ = "piles"
    __table_args__ = (
        Index("ix_piles_camera_active", "camera_id", "active"),
        Index("ix_piles_camera_track", "camera_id", "tracking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    tracking_id: Mapped[int] = mapped_column(Integer, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    stable_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="piles")


class CountRecord(Base):
    """Registro de contagem.

    Só é gravado quando a contagem **muda de fato** (ver
    :class:`app.services.inference_service.InferenceService`), evitando
    milhares de linhas idênticas.
    """

    __tablename__ = "count_records"
    __table_args__ = (
        Index("ix_count_records_camera_ts", "camera_id", "timestamp"),
        Index("ix_count_records_pile_ts", "pile_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    pile_id: Mapped[int | None] = mapped_column(
        ForeignKey("piles.id", ondelete="CASCADE"), nullable=True, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    chair_count: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stability_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="STABLE")
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    total_chairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pile_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    previous_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="ai")  # ai | manual


class ModelVersion(Base):
    """Versão de modelo YOLO com as métricas reais do treino."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map5095: Mapped[float | None] = mapped_column(Float, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classes: Mapped[str | None] = mapped_column(String(255), nullable=True)


class SystemEvent(Base):
    """Evento do sistema (info/warn/error). Alimenta a página de Logs."""

    __tablename__ = "system_events"
    __table_args__ = (Index("ix_events_ts", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    event: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Calibration(Base):
    """ROI e parâmetros de contagem por câmera (um registro por câmera)."""

    __tablename__ = "calibration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), unique=True, index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    roi_x: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_y: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_w: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chair_height_px: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    stability_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.55)
    counting_method: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    chair_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    pile_gap_px: Mapped[int] = mapped_column(Integer, nullable=False, default=40)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class Snapshot(Base):
    """Imagem salva (auditoria e material para o dataset)."""

    __tablename__ = "snapshots"
    __table_args__ = (Index("ix_snapshots_camera_ts", "camera_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pile_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_size: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CountCorrection(Base):
    """Correção manual do operador.

    Guarda ``ai_count`` e ``correct_count`` para virar dado de treino /
    análise de erro do modelo.
    """

    __tablename__ = "count_corrections"
    __table_args__ = (Index("ix_corrections_camera_ts", "camera_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    ai_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    author: Mapped[str] = mapped_column(String(64), nullable=False, default="operador")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)


__all__ = [
    "Base",
    "Camera",
    "Pile",
    "CountRecord",
    "ModelVersion",
    "SystemEvent",
    "Calibration",
    "Snapshot",
    "CountCorrection",
]
