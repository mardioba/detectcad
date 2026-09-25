"""Tracking de pilhas.

Duas opções, selecionáveis por ``TRACKER=`` no .env:

* ``internal`` (padrão) - tracker próprio baseado em IoU horizontal + vertical,
  adequado ao cenário do projeto (câmera fixa, pilhas lado a lado, sem
  sobreposição em profundidade). É determinístico e não depende de arquivos
  externos do Ultralytics.
* ``bytetrack`` / ``botsort`` - delegam ao Ultralytics, útil se as pilhas
  se movem muito (cobrindo/descobrindo pilhas durante a operação).

A identidade da pilha é o que permite mostrar "Pilha 1 / Pilha 2" de forma
estável no dashboard e no banco de dados.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from app.schemas import Detection, PileCandidate, iou_xyxy

log = logging.getLogger("ai")

_TRACK_SETTINGS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "botsort.yaml": "botsort.yaml",
    "bytetrack.yaml": "bytetrack.yaml",
}


@dataclass
class Track:
    """Rastro persistente de uma pilha."""

    tracking_id: int
    bbox: tuple[float, float, float, float]
    hits: int = 1
    misses: int = 0
    age: int = 1
    velocity: tuple[float, float, field] = field(default=(0.0, 0.0, 0.0))
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    pile_id: int = 0

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
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    def predict(self) -> tuple[float, float, float, float]:
        """BBox prevista um passo à frente (velocidade simples)."""
        vx, vy = self.velocity[0], self.velocity[1]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return (self.bbox[0] + vx, self.bbox[1] + vy, self.bbox[0] + vx + w, self.bbox[1] + vy + h)

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        prev_cx, prev_cy = self.cx, self.cy
        self.bbox = bbox
        new_cx, new_cy = self.cx, self.cy
        # Média móvel da velocidade para suavizar.
        self.velocity = (
            0.5 * self.velocity[0] + 0.5 * (new_cx - prev_cx),
            0.5 * self.velocity[1] + 0.5 * (new_cy - prev_cy),
            0.0,
        )
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.history.append((new_cx, new_cy, time_stamp()))

    def mark_missed(self) -> None:
        self.misses += 1
        self.age += 1


def time_stamp() -> float:
    import time

    return time.time()


class PileTracker:
    """Tracker IoU com predição de velocidade e IDs estáveis.

    Lógica de association (greedy por maior similaridade):

    1. Prediz a posição de cada track existente.
    2. Calcula a similaridade de cada par (candidato, track).
    3. Associa de cima para baixo enquanto a similaridade passar do limiar.
    4. Tracks não associados contam ``misses``; são removidos após o limite.
    5. Candidatos sem track recebem um novo ID incremental (nunca reaproveita
       um ID já encerrado, evitando "pilha 1" reaparecer com outro sentido).
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 1,
        center_weight: float = 0.6,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.center_weight = center_weight
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._confirmed: dict[int, int] = {}

    # ------------------------------------------------------------------ utils
    def _similarity(self, track: Track, cand: PileCandidate) -> float:
        px1, py1, px2, py2 = track.predict()
        iou = iou_xyxy(px1, py1, px2, py2, cand.x1, cand.y1, cand.x2, cand.y2)
        # Distância de centro normalizada pelo tamanho médio da pilha.
        w = max(1.0, (cand.width + (track.bbox[2] - track.bbox[0])) / 2.0)
        h = max(1.0, (cand.height + (track.bbox[3] - track.bbox[1])) / 2.0)
        dcx = (track.cx - cand.cx) / w
        dcy = (track.cy - cand.cy) / h
        center_sim = math.exp(-math.hypot(dcx, dcy))
        return (1.0 - self.center_weight) * iou + self.center_weight * center_sim

    # ------------------------------------------------------------------- API
    def update(self, candidates: list[PileCandidate]) -> list[tuple[int, PileCandidate, Track]]:
        """Associa candidatos às tracks. Retorna ``(tracking_id, cand, track)``."""
        pairs: list[tuple[float, int, int]] = []
        for ti, track in self._tracks.items():
            for ci, cand in enumerate(candidates):
                sim = self._similarity(track, cand)
                if sim >= self.iou_threshold:
                    pairs.append((sim, ti, ci))
        # Maior similaridade primeiro (greedy matching).
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_tracks: set[int] = set()
        used_cands: set[int] = set()
        self._assigned: dict[int, int] = {}
        matches: list[tuple[int, int]] = []
        for _sim, ti, ci in pairs:
            if ti in used_tracks or ci in used_cands:
                continue
            used_tracks.add(ti)
            used_cands.add(ci)
            matches.append((ti, ci))

        for ti, ci in matches:
            self._tracks[ti].update(candidates[ci].bbox)
            self._assigned[ci] = ti

        for ti, track in self._tracks.items():
            if ti not in used_tracks:
                track.mark_missed()

        for cand_idx, cand in enumerate(candidates):
            if cand_idx in used_cands:
                continue
            new_id = self._next_id
            self._next_id += 1
            track = Track(tracking_id=new_id, bbox=cand.bbox)
            self._tracks[new_id] = track
            # Candidatos sem track anterior também recebem track agora: sem
            # isso a primeira detecção de uma pilha nunca apareceria na saída.
            self._assigned[cand_idx] = new_id

        expired = [tid for tid, t in self._tracks.items() if t.misses > self.max_age]
        for tid in expired:
            log.debug("Track %d expirou após %d frames perdidos", tid, self._tracks[tid].misses)
            self._tracks.pop(tid, None)

        results: list[tuple[int, PileCandidate, Track]] = []
        for ci, cand in enumerate(candidates):
            ti = self._assigned.get(ci)
            if ti is None or ti not in self._tracks:
                continue
            results.append((ti, cand, self._tracks[ti]))
        return results

    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.misses == 0]

    def all_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def track_count(self) -> int:
        return len(self._tracks)


def resolve_tracker_config(name: str) -> tuple[str, str | None]:
    """Traduz ``TRACKER=`` para ``(modo, arquivo_yaml)``.

    ``("internal", None)`` ou ``("ultralytics", "bytetrack.yaml")``.
    """
    key = (name or "internal").strip().lower()
    if key in ("internal", "none", "custom", ""):
        return "internal", None
    if key in _TRACK_SETTINGS:
        return "ultralytics", _TRACK_SETTINGS[key]
    # Aceita caminho customizado para um YAML de tracker.
    if key.endswith(".yaml"):
        return "ultralytics", name
    return "internal", None


def group_by_x(
    detections: Iterable[Detection],
    *,
    overlap_min: float = 0.30,
    gap_px: float = 40.0,
) -> list[list[Detection]]:
    """Agrupa detecções de cadeira em pilhas usando a posição horizontal.

    Premissa do projeto: pilhas lado a lado, nunca uma atrás da outra nem
    sobrepostas. Logo, duas cadeiras pertencem à mesma pilha se suas caixas
    se sobrepõem horizontalmente acima de ``overlap_min`` da largura; caso
    contrário, se a distância horizontal for maior que ``gap_px``, começam
    pilhas diferentes.
    """
    dets = sorted(detections, key=lambda d: d.x1)
    if not dets:
        return []

    groups: list[list[Detection]] = [[dets[0]]]
    current_right = dets[0].x2
    for det in dets[1:]:
        inter = min(current_right, det.x2) - max(groups[-1][-1].x1, det.x1)
        same_pile = inter > 0 and (inter / max(1e-6, det.width)) >= overlap_min
        gap = det.x1 - current_right
        if same_pile and gap <= gap_px:
            groups[-1].append(det)
            current_right = max(current_right, det.x2)
        else:
            groups.append([det])
            current_right = det.x2
    return groups


__all__ = [
    "Track",
    "PileTracker",
    "resolve_tracker_config",
    "group_by_x",
]
