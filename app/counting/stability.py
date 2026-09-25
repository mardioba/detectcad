"""Filtro de estabilidade temporal.

Regra central do projeto: **um frame não muda o que aparece no dashboard**.

Com ``COUNT_STABILITY_FRAMES=10`` e ``STABILITY_MODE=mode`` (votação):

    20, 20, 21, 20, 20, 20, 20, 20, 20, 20  ->  20   (voto minoritário eliminado)
    20, 20, 21, 21, 21, 21                   ->  21   (novo valor dominante)

Modos:

``mode``   - valor mais frequente na janela (recomendado; imune a outliers)
``median`` - mediana (boa quando há muitos outliers)
``mean``   - média arredondada (suaviza, mas pode "inventar" valores)

Além da agregação, o filtro mede a **confiança de estabilidade**: concordância
entre os valores da janela. É ela que define se a contagem está STABLE ou
UNSTABLE.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque

from app.config import settings
from app.schemas import CountStatus

log = logging.getLogger("ai")


@dataclass
class StabilityResult:
    """Saída do filtro de estabilidade para uma pilha."""

    stable_count: int
    raw_count: int
    stability_confidence: float
    votes: dict[str, int] = field(default_factory=dict)
    window_size: int = 0
    agreed: bool = True
    method: str = "mode"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stable_count": self.stable_count,
            "raw_count": self.raw_count,
            "stability_confidence": round(self.stability_confidence, 4),
            "votes": self.votes,
            "window_size": self.window_size,
            "agreed": self.agreed,
            "method": self.method,
        }


class StabilityFilter:
    """Janela deslizante por pilha."""

    def __init__(
        self,
        window: int | None = None,
        mode: str | None = None,
        change_min_frames: int | None = None,
        min_agreement: float | None = None,
    ) -> None:
        self.window = int(window or settings.counting.count_stability_frames)
        self.mode = (mode or settings.counting.stability_mode).lower()
        self.change_min_frames = int(change_min_frames or settings.counting.change_min_frames)
        self.min_agreement = (
            float(min_agreement) if min_agreement is not None else settings.counting.stability_min_confidence
        )
        self._windows: dict[int, Deque[int]] = {}
        self._displayed: dict[int, int] = {}   # contagem realmente exibida
        self._candidate: dict[int, int] = {}   # novo valor em teste
        self._candidate_hits: dict[int, int] = {}
        self._locked: set[int] = set()         # pfixadas manualmente pelo operador

    # ------------------------------------------------------------------ API
    def update(self, pile_id: int, raw_count: int) -> StabilityResult:
        """Adiciona uma leitura e devolve o valor estabilizado.

        Se a pilha foi corrigida manualmente (:meth:`force`), o valor do
        operador é preservado: a IA não sobrescreve a correção sozinha.

        Regra da troca de valor: um novo valor só entra no display depois de
        aparecer ``change_min_frames`` vezes consecutivas na janela. Isso
        impede a troca por causa de um pico isolado.
        """
        win = self._windows.get(pile_id)
        if win is None:
            win = deque(maxlen=self.window)
            self._windows[pile_id] = win
        win.append(int(raw_count))

        aggregated = self._aggregate(win)
        votes = dict(Counter(win).most_common())
        stability_conf = self._agreement(win, aggregated)

        # Correção manual: o operador manda, a IA não mexe.
        if pile_id in self._locked:
            return StabilityResult(
                stable_count=int(self._displayed.get(pile_id, aggregated)),
                raw_count=int(raw_count),
                stability_confidence=stability_conf,
                votes=votes,
                window_size=len(win),
                agreed=True,
                method=self.mode,
            )

        current = self._displayed.get(pile_id)
        if current is None:
            # Primeira leitura da pilha: exibe já, mas exige a janela mínima.
            self._displayed[pile_id] = aggregated
            self._candidate[pile_id] = aggregated
            self._candidate_hits[pile_id] = len(win)
            return StabilityResult(
                stable_count=aggregated,
                raw_count=int(raw_count),
                stability_confidence=stability_conf,
                votes=votes,
                window_size=len(win),
                agreed=True,
                method=self.mode,
            )

        if aggregated == current:
            self._candidate.pop(pile_id, None)
            self._candidate_hits[pile_id] = 0
        else:
            # O valor agregado mudou: conta ocorrências consecutivas.
            if self._candidate.get(pile_id) == aggregated:
                self._candidate_hits[pile_id] = self._candidate_hits.get(pile_id, 0) + 1
            else:
                self._candidate[pile_id] = aggregated
                self._candidate_hits[pile_id] = 1
            hits = self._candidate_hits[pile_id]
            ready = hits >= self.change_min_frames or hits >= max(1, int(0.6 * len(win)))
            if ready:
                log.info(
                    "Pilha %s: contagem %d -> %d após %d leituras consistentes",
                    pile_id, current, aggregated, hits,
                )
                self._displayed[pile_id] = aggregated
                self._candidate.pop(pile_id, None)
                self._candidate_hits[pile_id] = 0

        displayed = self._displayed.get(pile_id, aggregated)
        return StabilityResult(
            stable_count=int(displayed),
            raw_count=int(raw_count),
            stability_confidence=stability_conf,
            votes=votes,
            window_size=len(win),
            agreed=abs(displayed - aggregated) <= 1,
            method=self.mode,
        )

    def get(self, pile_id: int) -> int | None:
        return self._displayed.get(pile_id)

    def force(self, pile_id: int, value: int) -> None:
        """Fixa um valor (correção manual do operador).

        O valor fica **travado**: leituras seguintes da IA não sobrescrevem a
        correção. Use :meth:`unlock` para devolver a pilha ao controle da IA
        (por exemplo, quando a pilha for desmanchada e remontada).
        """
        self._displayed[pile_id] = int(value)
        win = self._windows.get(pile_id)
        if win is not None:
            win.clear()
            for _ in range(min(self.window, self.change_min_frames)):
                win.append(int(value))
        self._candidate[pile_id] = int(value)
        self._candidate_hits[pile_id] = 0
        self._locked.add(pile_id)

    def unlock(self, pile_id: int) -> None:
        """Libera a pilha: a IA volta a decidir a contagem."""
        self._locked.discard(pile_id)
        win = self._windows.get(pile_id)
        if win is not None:
            win.clear()
        self._candidate.pop(pile_id, None)
        self._candidate_hits[pile_id] = 0

    def is_locked(self, pile_id: int) -> bool:
        return pile_id in self._locked

    def remove(self, pile_id: int) -> None:
        self._windows.pop(pile_id, None)
        self._displayed.pop(pile_id, None)
        self._candidate.pop(pile_id, None)
        self._candidate_hits.pop(pile_id, None)
        self._locked.discard(pile_id)

    def clear(self) -> None:
        self._windows.clear()
        self._displayed.clear()
        self._candidate.clear()
        self._candidate_hits.clear()
        self._locked.clear()

    def history(self, pile_id: int) -> list[int]:
        return list(self._windows.get(pile_id, []))

    # ------------------------------------------------------------- interno
    def _aggregate(self, values: list[int]) -> int:
        if not values:
            return 0
        if self.mode == "mean":
            return int(round(statistics.fmean(values)))
        if self.mode == "median":
            return int(statistics.median(values))
        # mode (padrão): desempata pelo valor mais recente
        counts = Counter(values)
        top = max(counts.values())
        tied = {v for v, c in counts.items() if c == top}
        for v in reversed(values):  # reversed -> o mais recente ganha
            if v in tied:
                return int(v)
        return int(values[-1])

    def _agreement(self, values: list[int], aggregated: int) -> float:
        """Concordância da janela: fração de leituras no valor aggregates.

        Também cai um pouco se a janela ainda está enchendo (poucos frames).
        """
        if not values:
            return 0.0
        agree = sum(1 for v in values if v == aggregated)
        ratio = agree / len(values)
        fill = min(1.0, len(values) / max(1, self.window))
        return float(max(0.0, min(1.0, ratio * (0.5 + 0.5 * fill))))


def decide_status(
    *,
    count: int,
    confidence: float,
    stability_confidence: float,
    partial: bool,
    min_confidence: float | None = None,
    stability_min: float | None = None,
    min_chairs: int | None = None,
) -> CountStatus:
    """Define o estado exibido no dashboard.

    Prioridade: UNKNOWN > LOW_CONFIDENCE > UNSTABLE > PARTIAL > STABLE.
    """
    min_conf = settings.counting.min_confidence if min_confidence is None else min_confidence
    stab_min = settings.counting.stability_min_confidence if stability_min is None else stability_min
    min_chairs = settings.counting.min_chairs_for_valid if min_chairs is None else min_chairs

    if count <= 0:
        return CountStatus.UNKNOWN
    if confidence < min_conf:
        return CountStatus.LOW_CONFIDENCE
    if stability_confidence < stab_min:
        return CountStatus.UNSTABLE
    if partial or count < min_chairs:
        return CountStatus.PARTIAL
    return CountStatus.STABLE


__all__ = ["StabilityFilter", "StabilityResult", "decide_status"]
