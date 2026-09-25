"""Enums e estruturas de dados compartilhadas (sem dependência de ORM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def localnow() -> datetime:
    return datetime.now()


class CameraState(str, Enum):
    """Estados possíveis da câmera."""

    OFFLINE = "OFFLINE"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class CountStatus(str, Enum):
    """Estado da contagem de uma pilha."""

    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class SourceMode(str, Enum):
    """Origem dos frames - permite testar sem câmera."""

    RTSP = "rtsp"
    VIDEO = "video"
    IMAGE = "image"
    NONE = "none"


class EventLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class Detection:
    """Uma detecção YOLO já normalizada (coordenadas em pixels da imagem)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str
    track_id: int | None = None
    mask: Any = None  # ndarray binário (uint8 0/255) ou None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def iou(self, other: "Detection") -> float:
        return iou_xyxy(self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2)

    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "conf": round(self.conf, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


def iou_xyxy(
    ax1: float, ay1: float, ax2: float, ay2: float, bx1: float, by1: float, bx2: float, by2: float
) -> float:
    """IoU entre dois retângulos ``xyxy``."""
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def x_overlap_ratio(a: "Detection", b: "Detection") -> float:
    """Fração da largura de ``a`` que se sobrepõe horizontalmente com ``b``.

    Como as pilhas estão lado a lado e nunca atrás umas das outras, a
    sobreposição horizontal é o sinal mais confiável para separar pilhas.
    """
    inter = min(a.x2, b.x2) - max(a.x1, b.x1)
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, a.width)


@dataclass(slots=True)
class PileCandidate:
    """Pilha detectada no frame, antes da estabilização de ID."""

    bbox: tuple[float, float, float, float]
    chair_detections: list[Detection] = field(default_factory=list)
    conf: float = 0.0
    source: Literal["pile_class", "chair_group"] = "chair_group"
    partial: bool = False

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass(slots=True)
class CountEstimate:
    """Resultado do StackCounter para uma pilha, em um único frame."""

    count: int
    confidence: float
    detection_confidence: float
    method: str
    candidates: dict[str, float] = field(default_factory=dict)
    candidate_weights: dict[str, float] = field(default_factory=dict)
    chair_height_px: float = 0.0
    pitch_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "method": self.method,
            "candidates": {k: round(v, 2) for k, v in self.candidates.items()},
            "weights": {k: round(v, 3) for k, v in self.candidate_weights.items()},
            "chair_height_px": round(self.chair_height_px, 1),
            "pitch_px": round(self.pitch_px, 1),
            "notes": self.notes,
        }


@dataclass(slots=True)
class PileState:
    """Estado estável de uma pilha através do tempo."""

    pile_id: int
    tracking_id: int
    bbox: tuple[float, float, float, float]
    raw_count: int
    stable_count: int
    confidence: float
    detection_confidence: float
    stability_confidence: float
    status: CountStatus
    method: str = "auto"
    candidates: dict[str, float] = field(default_factory=dict)
    pitch_px: float = 0.0
    chair_height_px: float = 0.0
    partial: bool = False
    missed_frames: int = 0
    manual: bool = False
    first_seen: datetime = field(default_factory=localnow)
    last_seen: datetime = field(default_factory=localnow)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pile_id": self.pile_id,
            "tracking_id": self.tracking_id,
            "bbox": [round(v, 1) for v in self.bbox],
            "raw_count": self.raw_count,
            "count": self.stable_count,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "stability_confidence": round(self.stability_confidence, 4),
            "status": self.status.value,
            "method": self.method,
            "candidates": {k: round(v, 2) for k, v in self.candidates.items()},
            "pitch_px": round(self.pitch_px, 1),
            "chair_height_px": round(self.chair_height_px, 1),
            "partial": self.partial,
            "manual": self.manual,
            "missed_frames": self.missed_frames,
            "first_seen": self.first_seen.isoformat(timespec="seconds"),
            "last_seen": self.last_seen.isoformat(timespec="seconds"),
            "notes": self.notes,
        }


__all__ = [
    "CameraState",
    "CountStatus",
    "SourceMode",
    "EventLevel",
    "Detection",
    "PileCandidate",
    "CountEstimate",
    "PileState",
    "iou_xyxy",
    "x_overlap_ratio",
    "localnow",
    "utcnow",
]
