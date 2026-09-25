"""Cálculo e apresentação da confiança.

A confiança exibida no dashboard é a **confiança da contagem de
quantas cadeiras existem**, não a confiança do detector YOLO. São coisas
diferentes:

* ``detection_confidence`` - quão bem o YOLO enxergou a pilha.
* ``stability_confidence`` - quanto a contagem concorda consigo mesma ao
  longo do tempo (filtro de estabilidade).
* ``confidence``            - combinação das duas acima com a concordância
  entre os estimadores (período, picos, detecções, tamanho).

Um exemplo real de saída:

    PILHA 1
    23 cadeiras
    Confiança 97%

e, quando a leitura é duvidosa:

    PILHA 1
    ?  contagem indeterminada
    Confiança 48%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas import CountStatus

# Peso de cada componente na confiança final.
# A detecção e a estabilidade dominam; a concordância entre estimadores é o
# fator que mais distingue "23 com certeza" de "23 por coincidência".
W_DETECTION = 0.35
W_STABILITY = 0.30
W_AGREEMENT = 0.35


@dataclass
class ConfidenceResult:
    confidence: float
    detection_confidence: float
    stability_confidence: float
    level: str          # high | medium | low | unknown
    label: str          # texto pronto para o overlay
    uncertain: bool     # True -> mostrar "?" em vez do número

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "stability_confidence": round(self.stability_confidence, 4),
            "level": self.level,
            "label": self.label,
            "uncertain": self.uncertain,
        }


def combine(
    *,
    detection_confidence: float,
    stability_confidence: float,
    agreement: float = 0.5,
    pile_confidence: float = 0.5,
    min_confidence: float = 0.55,
) -> ConfidenceResult:
    """Funde os componentes de confiança em um valor único 0..1.

    ``pile_confidence`` é a confiança que o :class:`StackCounter` calculou
    (já inclui a concordância entre os estimadores). Quando ela existe, ela
    entra como o termo principal e a detecção/stabilidade a refinam.
    """
    det = float(np.clip(detection_confidence, 0.0, 1.0))
    stab = float(np.clip(stability_confidence, 0.0, 1.0))
    agr = float(np.clip(agreement, 0.0, 1.0))
    pile = float(np.clip(pile_confidence, 0.0, 1.0))

    combined = W_DETECTION * det + W_STABILITY * stab + W_AGREEMENT * agr
    # A confiança do contador não pode ser "esquecida": no mínimo 70% dela.
    combined = max(combined, 0.70 * pile)

    level, uncertain = _level(combined, min_confidence)
    return ConfidenceResult(
        confidence=float(np.clip(combined, 0.0, 1.0)),
        detection_confidence=det,
        stability_confidence=stab,
        level=level,
        label=f"Confiança {combined * 100:.0f}%",
        uncertain=uncertain,
    )


def _level(value: float, min_confidence: float) -> tuple[str, bool]:
    """Classifica a confiança e decide se o número deve ser exibido."""
    if value < 0.30:
        return "unknown", True
    if value < min_confidence:
        return "low", True
    if value < 0.80:
        return "medium", False
    return "high", False


# Cor do overlay por estado da contagem (BGR para OpenCV, hex para o front).
STATUS_COLORS_BGR: dict[CountStatus, tuple[int, int, int]] = {
    CountStatus.STABLE: (60, 200, 60),       # verde
    CountStatus.UNSTABLE: (40, 200, 240),    # âmbar
    CountStatus.LOW_CONFIDENCE: (60, 120, 255),  # laranja
    CountStatus.PARTIAL: (200, 120, 220),    # roxo
    CountStatus.UNKNOWN: (110, 110, 110),    # cinza
}

STATUS_COLORS_HEX: dict[CountStatus, str] = {
    CountStatus.STABLE: "#3ec93e",
    CountStatus.UNSTABLE: "#f0c828",
    CountStatus.LOW_CONFIDENCE: "#ff783e",
    CountStatus.PARTIAL: "#dc78dc",
    CountStatus.UNKNOWN: "#6e6e6e",
}

STATUS_LABELS_PT: dict[CountStatus, str] = {
    CountStatus.STABLE: "ESTÁVEL",
    CountStatus.UNSTABLE: "INSTÁVEL",
    CountStatus.LOW_CONFIDENCE: "BAIXA CONFIANÇA",
    CountStatus.PARTIAL: "PARCIAL",
    CountStatus.UNKNOWN: "INDETERMINADO",
}


def status_color_bgr(status: CountStatus) -> tuple[int, int, int]:
    return STATUS_COLORS_BGR.get(status, (200, 200, 200))


def status_color_hex(status: CountStatus) -> str:
    return STATUS_COLORS_HEX.get(status, "#c8c8c8")


def status_label(status: CountStatus) -> str:
    return STATUS_LABELS_PT.get(status, status.value)


__all__ = [
    "ConfidenceResult",
    "combine",
    "status_color_bgr",
    "status_color_hex",
    "status_label",
    "STATUS_COLORS_BGR",
    "STATUS_COLORS_HEX",
    "STATUS_LABELS_PT",
]
