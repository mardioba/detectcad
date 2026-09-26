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

# ===========================================================================
# ARQUIVO / MAPA  -  confidence.py
# ===========================================================================
# Última etapa antes da tela: junta as várias confianças em UM número e
# decide se o operador vê "23" ou "?".
#
# - ConfidenceResult: saída (número, nível, texto pronto, flag "?").
# - combine: a fusão. 3 componentes linearly pesados + um piso vindo da
#   confiança do próprio contador.
# - _level: corta o número contínuo em 4 faixas e diz se deve mostrar "?".
# - STATUS_COLORS_BGR / _HEX / STATUS_LABELS_PT: cor e texto por status.
#   BGR é a ordem do OpenCV (o overlay é desenhado no frame), HEX é do CSS.
# - status_color_bgr / status_color_hex / status_label: acesso com default
#   cinza para status desconhecido (não quebra o render se o enum crescer).
#
# Ordem de leitura: combine -> _level -> as tabelas de cor/label.
#
# Por que um arquivo só para isso: a confiança exibida é o contrato com o
# operador. Se ela mudar de fórmula, muda o que as pessoas veem. Fica
# separado para dar pra revisar/testar sem mexer na contagem.
# ===========================================================================

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
    """Confiança final exibida, já com nível e texto prontos.

    ``uncertain`` é o campo que manda: quando True, a UI troca o número por
    "?". Não é só cosmético - é o que impede alguém de anotar 23 no
    papel quando a leitura era um chute.
    """

    confidence: float
    # As duas entradas originais, preservadas para o painel poder explicar a
    # composição ("YOLO 0.90 | estabilidade 0.72 | acordo 0.65").
    detection_confidence: float
    stability_confidence: float
    level: str          # high | medium | low | unknown
    label: str          # texto pronto para o overlay
    uncertain: bool     # True -> mostrar "?" em vez do número

    def as_dict(self) -> dict[str, Any]:
        """Dump para JSON (API/dashboard)."""
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
    # clip em 0..1: os componentes vêm de fontes diferentes (YOLO, filtro de
    # estabilidade) e um valor fora da escala aqui corromperia toda a média.
    det = float(np.clip(detection_confidence, 0.0, 1.0))
    stab = float(np.clip(stability_confidence, 0.0, 1.0))
    agr = float(np.clip(agreement, 0.0, 1.0))
    pile = float(np.clip(pile_confidence, 0.0, 1.0))

    # Soma dos pesos = 1.0, então combined cai no máximo em 1.0 mesmo com
    # tudo perfeito. Nenhum termo isolado decide: é média, não máximo.
    combined = W_DETECTION * det + W_STABILITY * stab + W_AGREEMENT * agr
    # A confiança do contador não pode ser "esquecida": no mínimo 70% dela.
    # Workaround deliberado: quando não há YOLO (galpão sem modelo), det=0
    # e a média cairia para ~0.3 mesmo com a análise de padrão impecável.
    # O piso devolve ao contador o peso que a detecção não tem.
    combined = max(combined, 0.70 * pile)

    # level/uncertain saem de um único corte, para que "high" e "mostrar o
    # número" nunca se contradigam.
    level, uncertain = _level(combined, min_confidence)
    return ConfidenceResult(
        confidence=float(np.clip(combined, 0.0, 1.0)),
        detection_confidence=det,
        stability_confidence=stab,
        level=level,
        # Porcentagem inteira: o operador precisa ler de relance, e 96.7% vs
        # 97% é diferença que não existe no mundo real.
        label=f"Confiança {combined * 100:.0f}%",
        uncertain=uncertain,
    )


def _level(value: float, min_confidence: float) -> tuple[str, bool]:
    """Classifica a confiança e decide se o número deve ser exibido.

    Corta em 4 faixas. Os dois primeiros (abaixo de 0.30 e abaixo do
    ``min_confidence``) marcam ``uncertain``: abaixo disso mostrar um número
    inteiro é pior do que mostrar "?", porque o número vai ser anotado.

    0.80 é o corte de "high": acima disso o número pode ser mostrado sem
    ressalva. Entre ``min_confidence`` (0.55) e 0.80 o número aparece mas
    com a etiqueta "medium" - é o caso comum de uma pilha boa com detecção
    parcial.
    """
    if value < 0.30:
        # 0.30 é o piso absoluto: abaixo disso não há nem YOLO, nem
        # estabilidade, nem padrão - é ruído.
        return "unknown", True
    if value < min_confidence:
        return "low", True
    if value < 0.80:
        return "medium", False
    return "high", False


# Cor do overlay por estado da contagem (BGR para OpenCV, hex para o front).
# Regra de cor: verde = pode confiar e anotar; âmbar/laranja = número
# provavelmente errado; cinza = não há número. A distinção entre âmbar
# (UNSTABLE) e laranja (LOW_CONFIDENCE) é proposital: são problemas
# diferentes e o operador reage diferente (esperar vs. recalibrar).
STATUS_COLORS_BGR: dict[CountStatus, tuple[int, int, int]] = {
    CountStatus.STABLE: (60, 200, 60),       # verde
    CountStatus.UNSTABLE: (40, 200, 240),    # âmbar
    CountStatus.LOW_CONFIDENCE: (60, 120, 255),  # laranja
    CountStatus.PARTIAL: (200, 120, 220),    # roxo
    CountStatus.UNKNOWN: (110, 110, 110),    # cinza
}

# Mesmas cores em hex, para o CSS. Os valores são a mesma cor vista de outro
# ângulo: converter BGR->RGB trocando os dois primeiros canais. Se mudar uma
# tabela, mude a outra junto.
STATUS_COLORS_HEX: dict[CountStatus, str] = {
    CountStatus.STABLE: "#3ec93e",
    CountStatus.UNSTABLE: "#f0c828",
    CountStatus.LOW_CONFIDENCE: "#ff783e",
    CountStatus.PARTIAL: "#dc78dc",
    CountStatus.UNKNOWN: "#6e6e6e",
}

# Texto do overlay por status. Em maiúsculas e curto: é lido à distância,
# no vídeo, enquanto a pessoa está empilhando cadeira.
STATUS_LABELS_PT: dict[CountStatus, str] = {
    CountStatus.STABLE: "ESTÁVEL",
    CountStatus.UNSTABLE: "INSTÁVEL",
    CountStatus.LOW_CONFIDENCE: "BAIXA CONFIANÇA",
    CountStatus.PARTIAL: "PARCIAL",
    CountStatus.UNKNOWN: "INDETERMINADO",
}


def status_color_bgr(status: CountStatus) -> tuple[int, int, int]:
    """Cor BGR do overlay. Cinza claro se o status não estiver no mapa."""
    return STATUS_COLORS_BGR.get(status, (200, 200, 200))


def status_color_hex(status: CountStatus) -> str:
    """Cor hex para o CSS. Cinza claro se o status não estiver no mapa."""
    return STATUS_COLORS_HEX.get(status, "#c8c8c8")


def status_label(status: CountStatus) -> str:
    """Texto do overlay; cai no ``value`` do enum se não houver tradução."""
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
