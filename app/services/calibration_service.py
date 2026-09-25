"""Calibração por câmera.

Grava e lê a tabela ``calibration`` e mantém uma cópia em JSON em
``data/config/camera_<id>.json`` (útil para backup e para inspection manual).

Tudo que o operador ajusta no dashboard - ROI, altura da cadeira, limiares,
método de contagem - passa por aqui e é aplicado no worker em tempo real.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.counting.stack_counter import suggest_offset
from app.database.database import session_scope
from app.database.repository import CalibrationRepository, CorrectionRepository

log = logging.getLogger("app")


@dataclass
class RoiSpec:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def is_set(self) -> bool:
        return self.w > 0 and self.h > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class CalibrationData:
    """Configuração completa de uma câmera."""

    camera_id: int = 1
    enabled: bool = False
    roi: RoiSpec = field(default_factory=RoiSpec)
    chair_height_px: float = 0.0
    confidence_threshold: float = 0.50
    stability_frames: int = 10
    min_confidence: float = 0.55
    counting_method: str = "auto"
    chair_weight: float = 1.0
    count_offset: int = 0
    pile_gap_px: int = 40
    stability_min_confidence: float = 0.60
    change_min_frames: int = 3
    chair_classes: str = "chair"
    pile_classes: str = "pile"
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class CalibrationService:
    """Leitura/escrita da calibração + persistência em JSON."""

    def __init__(self) -> None:
        self._cache: dict[int, CalibrationData] = {}

    # ------------------------------------------------------------------ paths
    def json_path(self, camera_id: int) -> Path:
        d = settings.config_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"camera_{camera_id}.json"

    # ------------------------------------------------------------------- load
    def load(self, camera_id: int) -> CalibrationData:
        """Carrega do banco; se não houver, semeia com os valores do .env."""
        with session_scope() as session:
            repo = CalibrationRepository(session)
            raw = repo.as_dict(camera_id)
        data = self._defaults(camera_id)
        if raw:
            roi = raw.get("roi") or {}
            data.enabled = bool(raw.get("enabled", data.enabled))
            data.roi = RoiSpec(
                x=int(roi.get("x", 0) or 0),
                y=int(roi.get("y", 0) or 0),
                w=int(roi.get("w", 0) or 0),
                h=int(roi.get("h", 0) or 0),
            )
            data.chair_height_px = float(raw.get("chair_height_px", 0.0) or 0.0)
            data.confidence_threshold = float(raw.get("confidence_threshold", data.confidence_threshold))
            data.stability_frames = int(raw.get("stability_frames", data.stability_frames))
            data.min_confidence = float(raw.get("min_confidence", data.min_confidence))
            data.counting_method = str(raw.get("counting_method", data.counting_method))
            data.chair_weight = float(raw.get("chair_weight", data.chair_weight))
            data.pile_gap_px = int(raw.get("pile_gap_px", data.pile_gap_px))
            params = raw.get("params") or {}
            data.count_offset = int(params.get("count_offset", 0) or 0)
            data.stability_min_confidence = float(params.get("stability_min_confidence", data.stability_min_confidence))
            data.change_min_frames = int(params.get("change_min_frames", data.change_min_frames))
            data.chair_classes = str(params.get("chair_classes", data.chair_classes))
            data.pile_classes = str(params.get("pile_classes", data.pile_classes))
            data.updated_at = raw.get("updated_at")
        self._cache[camera_id] = data
        return data

    def _defaults(self, camera_id: int) -> CalibrationData:
        return CalibrationData(
            camera_id=camera_id,
            confidence_threshold=settings.ai.confidence_threshold,
            stability_frames=settings.counting.count_stability_frames,
            min_confidence=settings.counting.min_confidence,
            counting_method=settings.counting.counting_method,
            chair_weight=settings.ai.chair_weight,
            chair_height_px=settings.counting.chair_height_px,
            pile_gap_px=settings.counting.pile_gap_px,
            stability_min_confidence=settings.counting.stability_min_confidence,
            change_min_frames=settings.counting.change_min_frames,
            chair_classes=settings.ai.chair_classes,
            pile_classes=settings.ai.pile_classes,
        )

    # ------------------------------------------------------------------- save
    def save(self, camera_id: int, payload: dict[str, Any]) -> CalibrationData:
        """Grava a calibração no banco e no JSON.

        Garante que a câmera exista: a tabela ``calibration`` tem chave
        estrangeira para ``cameras``, então salvar a calibração de uma câmera
        ainda não cadastrada quebraria com erro de integridade.
        """
        from app.database.repository import CameraRepository

        current = self.load(camera_id)
        roi_in = payload.get("roi")
        if isinstance(roi_in, dict):
            current.roi = RoiSpec(
                x=int(roi_in.get("x", current.roi.x) or 0),
                y=int(roi_in.get("y", current.roi.y) or 0),
                w=int(roi_in.get("w", current.roi.w) or 0),
                h=int(roi_in.get("h", current.roi.h) or 0),
            )
        elif isinstance(roi_in, str) and roi_in.strip():
            try:
                x, y, w, h = (int(v) for v in roi_in.replace(" ", "").split(",")[:4])
                current.roi = RoiSpec(x, y, w, h)
            except ValueError:
                pass

        _set_if(payload, "enabled", current, bool)
        _set_if(payload, "chair_height_px", current, float)
        _set_if(payload, "confidence_threshold", current, float)
        _set_if(payload, "stability_frames", current, int)
        _set_if(payload, "min_confidence", current, float)
        _set_if(payload, "counting_method", current, str)
        _set_if(payload, "chair_weight", current, float)
        _set_if(payload, "count_offset", current, int)
        _set_if(payload, "pile_gap_px", current, int)
        _set_if(payload, "stability_min_confidence", current, float)
        _set_if(payload, "change_min_frames", current, int)
        _set_if(payload, "chair_classes", current, str)
        _set_if(payload, "pile_classes", current, str)

        current.enabled = bool(current.enabled)
        current.updated_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

        with session_scope() as session:
            cam_repo = CameraRepository(session)
            if cam_repo.get(camera_id) is None:
                # Câmera ainda não cadastrada (ex.: calibração salva antes do
                # primeiro acesso). Cria o registro para não violar a FK.
                cam_repo.create(
                    name=str(payload.get("camera_name") or settings.camera.camera_name),
                    rtsp_url=settings.camera.camera_rtsp_url,
                    is_default=camera_id == 1,
                )
            repo = CalibrationRepository(session)
            repo.upsert(
                camera_id=camera_id,
                enabled=current.enabled,
                roi_x=current.roi.x,
                roi_y=current.roi.y,
                roi_w=current.roi.w,
                roi_h=current.roi.h,
                chair_height_px=current.chair_height_px,
                confidence_threshold=current.confidence_threshold,
                stability_frames=current.stability_frames,
                min_confidence=current.min_confidence,
                counting_method=current.counting_method,
                chair_weight=current.chair_weight,
                pile_gap_px=current.pile_gap_px,
                params_json=json.dumps(
                    {
                        "count_offset": current.count_offset,
                        "stability_min_confidence": current.stability_min_confidence,
                        "change_min_frames": current.change_min_frames,
                        "chair_classes": current.chair_classes,
                        "pile_classes": current.pile_classes,
                    }
                ),
            )
        # Persistência em arquivo (backup legível)
        try:
            self.json_path(camera_id).write_text(
                json.dumps(current.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Não foi possível gravar o JSON de calibração: %s", exc)
        self._cache[camera_id] = current
        log.info("Calibração da câmera %s salva", camera_id)
        return current

    def reset(self, camera_id: int) -> CalibrationData:
        return self.save(camera_id, self._defaults(camera_id).as_dict())

    # ------------------------------------------------------- sugestão de offset
    def offset_suggestion(self) -> dict[str, Any]:
        """Sugere ``count_offset`` a partir das correções manuais reais."""
        with session_scope() as session:
            corrections = CorrectionRepository(session).list(limit=1000)
        pairs = [(c.ai_count, c.correct_count) for c in corrections]
        result = suggest_offset(pairs)
        result["corrections_count"] = len(pairs)
        return result


def _set_if(payload: dict[str, Any], key: str, obj: Any, caster: Any) -> None:
    if key in payload and payload[key] is not None:
        try:
            setattr(obj, key, caster(payload[key]))
        except (TypeError, ValueError):
            log.warning("Valor inválido para %s: %r", key, payload[key])


calibration_service = CalibrationService()

__all__ = ["CalibrationService", "CalibrationData", "RoiSpec", "calibration_service"]
