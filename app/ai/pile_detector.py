"""Detecção de pilhas a partir das detecções do YOLO.

Suporta as duas estratégias pedidas:

* **classe ``pile``** - o modelo devolve a pilha inteira; as cadeiras
  individuais são usadas só como apoio.
* **classe ``chair`` apenas** - as cadeiras são agrupadas por posição
  horizontal para formar a pilha (funciona porque as pilhas ficam lado a
  lado).

A escolha é automática: se o modelo tem classe ``pile``, usa; senão agrupa.
Também é forçável pelo .env (``CHAIR_CLASSES`` / ``PILE_CLASSES``).
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.schemas import Detection, PileCandidate, x_overlap_ratio

log = logging.getLogger("ai")


class PileDetector:
    """Transforma detecções YOLO em :class:`PileCandidate`."""

    def __init__(
        self,
        chair_classes: set[int],
        pile_classes: set[int],
        *,
        overlap_min: float = 0.30,
        gap_px: float = 40.0,
        partial_ratio: float = 0.12,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.chair_classes = chair_classes
        self.pile_classes = pile_classes
        self.overlap_min = overlap_min
        self.gap_px = gap_px
        self.partial_ratio = partial_ratio
        self.image_size = image_size

    def configure(self, image_size: tuple[int, int] | None = None) -> None:
        """Ajusta o detector com o tamanho real do frame (ROI applied)."""
        if image_size is not None:
            self.image_size = image_size

    # ------------------------------------------------------------------ core
    def detect(
        self,
        detections: list[Detection],
        *,
        chair_weight: float | None = None,
    ) -> list[PileCandidate]:
        """Retorna as pilhas do frame.

        ``chair_weight``: quantas cadeiras cada detecção de ``chair``
        representa (1.0 = uma detecção é uma cadeira). Se o modelo foi
        treinado com ``pile``, o peso não se aplica às pilhas.
        """
        if not detections:
            return []

        chairs = [d for d in detections if d.class_id in self.chair_classes]
        piles_cls = [d for d in detections if d.class_id in self.pile_classes]
        weight = settings.ai.chair_weight if chair_weight is None else chair_weight

        if piles_cls:
            # Estratégia A: o modelo já delimita a pilha.
            return self._from_pile_class(piles_cls, chairs, weight)
        if chairs:
            # Estratégia B: agrupa as cadeiras por posição horizontal.
            return self._from_chair_group(chairs, weight)
        return []

    # ------------------------------------------------------------- estratégias
    def _from_pile_class(
        self, piles: list[Detection], chairs: list[Detection], weight: float
    ) -> list[PileCandidate]:
        cands: list[PileCandidate] = []
        for pile_det in piles:
            inside = [c for c in chairs if _inside(c, pile_det)]
            conf = self._pile_conf(pile_det, inside, weight)
            cands.append(
                PileCandidate(
                    bbox=(pile_det.x1, pile_det.y1, pile_det.x2, pile_det.y2),
                    chair_detections=inside,
                    conf=conf,
                    source="pile_class",
                    partial=self._is_partial(pile_det),
                )
            )
        return cands

    def _from_chair_group(self, chairs: list[Detection], weight: float) -> list[PileCandidate]:
        cands: list[PileCandidate] = []
        for group in self._cluster(chairs):
            if not group:
                continue
            x1 = min(d.x1 for d in group)
            y1 = min(d.y1 for d in group)
            x2 = max(d.x2 for d in group)
            y2 = max(d.y2 for d in group)
            # Padding leve para não cortar a base/encaixe da pilha.
            pad = max(2.0, 0.04 * (y2 - y1))
            bbox = (x1, max(0.0, y1 - pad), x2, y2 + pad)
            conf = self._group_conf(group, weight)
            cands.append(
                PileCandidate(
                    bbox=bbox,
                    chair_detections=list(group),
                    conf=conf,
                    source="chair_group",
                    partial=self._is_partial(PileCandidate(bbox=bbox)),
                )
            )
        return cands

    def _cluster(self, chairs: list[Detection]) -> list[list[Detection]]:
        """Clustering 1D no eixo X, respeitando a premissa "lado a lado"."""
        ordered = sorted(chairs, key=lambda d: d.x1)
        if not ordered:
            return []
        groups: list[list[Detection]] = [[ordered[0]]]
        for det in ordered[1:]:
            anchor = groups[-1][-1]
            overlap = x_overlap_ratio(det, anchor)
            gap = det.x1 - max(g.x2 for g in groups[-1])
            # Mesma pilha: sobreposição horizontal relevante OU encostadas
            # dentro da folga configurada.
            if overlap >= self.overlap_min or (-self.gap_px <= gap <= self.gap_px and overlap > 0.05):
                groups[-1].append(det)
            else:
                groups.append([det])
        return groups

    # ------------------------------------------------------------- confiança
    def _pile_conf(self, pile: Detection, inside: list[Detection], weight: float) -> float:
        """Confiança da detecção da pilha, com reforço das cadeiras internas."""
        if not inside:
            return pile.conf * 0.6  # pilha sem cadeira dentro: confiança reduzida
        mean_conf = sum(d.conf for d in inside) / len(inside)
        # Muitas cadeiras detectadas dentro da pilha reforçam a confiança.
        support = min(1.0, 0.6 + 0.1 * math_log1p(len(inside)))
        return min(1.0, 0.6 * pile.conf + 0.4 * mean_conf * support)

    def _group_conf(self, group: list[Detection], weight: float) -> float:
        """Confiança de uma pilha formada por agrupamento de cadeiras.

        Média das confidências com penalidade para grupos muito pequenos
        (1-2 cadeiras podem ser falso positivo) e para grupos enormes
        (podem ser várias pilhas coladas).
        """
        if not group:
            return 0.0
        mean_conf = sum(d.conf for d in group) / len(group)
        n = len(group)
        if n == 1:
            size_penalty = 0.7
        elif n == 2:
            size_penalty = 0.85
        elif n <= 60:
            size_penalty = 1.0
        else:
            size_penalty = max(0.5, 1.0 - (n - 60) / 120.0)
        return max(0.0, min(1.0, mean_conf * size_penalty))

    def _is_partial(self, pile: PileCandidate) -> bool:
        """Pilha cortada pelas bordas da imagem (contagem subestimada)."""
        if self.image_size is None:
            return False
        w, h = self.image_size
        left = pile.x1 <= w * self.partial_ratio
        right = pile.x2 >= w * (1.0 - self.partial_ratio)
        top = pile.y1 <= h * self.partial_ratio
        bottom = pile.y2 >= h * (1.0 - 0.02)
        return bool((left or right or top) and bottom)


def _inside(inner: Detection, outer: Detection) -> bool:
    """A cadeira está contida (pelo menos 60% da área) dentro da pilha?"""
    ix1 = max(inner.x1, outer.x1)
    iy1 = max(inner.y1, outer.y1)
    ix2 = min(inner.x2, outer.x2)
    iy2 = min(inner.y2, outer.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner.area <= 0:
        return False
    return inter / inner.area >= 0.6


def math_log1p(x: float) -> float:
    import math

    return math.log1p(max(0.0, float(x)))


def build_pile_detector(
    class_names: dict[int, str], image_size: tuple[int, int] | None = None
) -> PileDetector:
    """Cria o detector lendo os nomes de classe do modelo e o .env."""
    chairs, piles = settings.class_indices(class_names)
    if not chairs and not piles:
        # Modelo com classes desconhecidas: assume que todas são cadeira.
        chairs = set(class_names.keys()) if class_names else set()
        if chairs:
            log.warning(
                "Nenhuma classe de cadeira/pilha reconhecida. Usando todas as classes (%s) como cadeira.",
                class_names,
            )
    return PileDetector(
        chair_classes=chairs,
        pile_classes=piles,
        overlap_min=settings.counting.pile_overlap_min,
        gap_px=settings.counting.pile_gap_px,
        partial_ratio=settings.counting.partial_ratio,
        image_size=image_size,
    )


__all__ = ["PileDetector", "build_pile_detector"]
