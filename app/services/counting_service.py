"""Serviço de contagem: interface pública das operações de contagem.

O trabalho pesado acontece em :class:`~app.services.inference_service.InferenceService`
(fps a tempo real) e em :mod:`app.counting`. Este módulo é a **fachada**
usada pelo dashboard e pela API para tudo que é contagem:

* estado atual (por pilha e total);
* correção manual do operador;
* calibração da altura da cadeira e sugestão de offset;
* diagnóstico de uma região específica.

Mantê-lo separado deixa a API simples de usar e dá um único lugar para
regras de negócio de contagem, sem espalhar lógica pela camada web.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.counting.confidence import status_color_hex, status_label
from app.counting.stability import StabilityFilter
from app.counting.stack_counter import StackCounter, suggest_offset
from app.schemas import CountStatus, PileState
from app.services.state_store import StateStore, state_store

log = logging.getLogger("ai")


class CountingService:
    """Operações de contagem sobre o estado atual."""

    def __init__(self, store: StateStore | None = None) -> None:
        self.store = store or state_store

    # ------------------------------------------------------------- leitura
    def snapshot(self) -> dict[str, Any]:
        """Estado completo de contagem, pronto para JSON."""
        piles = self.store.get_piles()
        totals = self.store.totals_breakdown()
        return {
            "total": self.store.total(only_stable=True),
            "totals": totals,
            "pile_count": len(piles),
            "confidence": round(self.store.overall_confidence(), 4),
            "piles": [self.describe(p) for p in piles],
            "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }

    def describe(self, pile: PileState) -> dict[str, Any]:
        """Uma pilha com rótulos prontos para o frontend."""
        data = pile.as_dict()
        data["status_label"] = status_label(pile.status)
        data["color"] = status_color_hex(pile.status)
        data["needs_attention"] = pile.status in (
            CountStatus.LOW_CONFIDENCE,
            CountStatus.UNKNOWN,
            CountStatus.PARTIAL,
        )
        return data

    def pile(self, pile_id: int) -> PileState | None:
        return self.store.get_pile(pile_id)

    # -------------------------------------------------------------- escrita
    def apply_manual_count(self, inference: Any, pile_id: int, value: int) -> dict[str, Any]:
        """Aplica uma correção manual (valida e delega ao worker)."""
        if value < 0 or value > 10000:
            return {"ok": False, "message": "Contagem fora do intervalo (0-10000)."}
        if inference is None:
            return {"ok": False, "message": "Nenhum worker de inferência ativo."}
        return inference.manual_count(pile_id, value)

    def release_manual_count(self, inference: Any, pile_id: int) -> dict[str, Any]:
        """Devolve a pilha ao controle da IA após uma correção manual."""
        if inference is None:
            return {"ok": False, "message": "Nenhum worker de inferência ativo."}
        inference.stability.unlock(pile_id)
        runtime = inference._piles.get(pile_id)
        if runtime is not None:
            runtime.manual = False
        pile = self.store.get_pile(pile_id)
        if pile is not None:
            pile.manual = False
        self.store.set_piles(self.store.get_piles())
        log.info("Pilha %s voltou ao controle automático", pile_id)
        return {"ok": True, "pile_id": pile_id, "message": f"Pilha {pile_id} voltou ao modo automático."}

    # ---------------------------------------------------------- diagnóstico
    def diagnose_roi(
        self, frame: np.ndarray | None, roi: dict[str, Any], chair_height_px: float, count_offset: int = 0
    ) -> dict[str, Any]:
        """Analisa uma região e mostra o detalhe do que o algoritmo enxerga."""
        if frame is None:
            return {"ok": False, "message": "Nenhum frame disponível ainda."}
        try:
            x, y, w, h = (int(roi[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": "Informe a ROI: x, y, w, h."}
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w < 8 or h < 8:
            return {"ok": False, "message": "ROI muito pequena."}
        crop = frame[y : y + h, x : x + w]
        counter = StackCounter(chair_height_px=chair_height_px, count_offset=count_offset)
        est = counter.count(crop, chair_height_px=chair_height_px)
        return {
            "ok": True,
            "roi": {"x": x, "y": y, "w": w, "h": h},
            "estimate": est.as_dict(),
            "analysis": counter.diagnose(crop, chair_height_px),
        }

    def measure_chair_height(self, frame: np.ndarray | None, roi: dict[str, Any]) -> dict[str, Any]:
        """Mede a altura de uma cadeira em pixels a partir de uma ROI."""
        if frame is None:
            return {"ok": False, "message": "Nenhum frame disponível ainda."}
        try:
            x, y, w, h = (int(roi[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": "Informe a ROI da cadeira: x, y, w, h."}
        fh, fw = frame.shape[:2]
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            return {"ok": False, "message": "A ROI está fora da imagem."}
        crop = frame[y : y + h, x : x + w]
        analysis = StackCounter().diagnose(crop)
        return {
            "ok": True,
            "roi": {"x": x, "y": y, "w": w, "h": h},
            "measured_height_px": h,
            "measured_width_px": w,
            "analysis": analysis,
            "message": (
                f"Altura medida: {h} px. Use como CHAIR_HEIGHT_PX se esta caixa for "
                f"exatamente uma cadeira."
            ),
        }

    # ------------------------------------------------------------- Feedback
    def offset_suggestion(self, corrections: list[tuple[int, int]]) -> dict[str, Any]:
        """Sugere o offset de contagem a partir das correções manuais."""
        return suggest_offset(corrections)

    def error_statistics(self, corrections: list[tuple[int, int]]) -> dict[str, Any]:
        """Erro real do modelo medido nas correções do operador."""
        if not corrections:
            return {"samples": 0, "mean_abs_error": None, "exact_match_rate": None,
                    "message": "Nenhuma correção manual registrada."}
        errs = [abs(h - a) for a, h in corrections]
        return {
            "samples": len(errs),
            "mean_abs_error": round(sum(errs) / len(errs), 3),
            "exact_match_rate": round(sum(1 for e in errs if e == 0) / len(errs), 4),
            "max_abs_error": max(errs),
        }


# Fachada compartilhada (mesma instância usada pela API)
counting_service = CountingService()

__all__ = ["CountingService", "counting_service"]
