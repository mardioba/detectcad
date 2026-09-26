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

# ARQUIVO / MAPA
# Fachada de contagem. Não tem thread própria nem estado próprio: lê do
# StateStore e delega escrita para o InferenceService.
#
# Ordem de leitura:
#   1. snapshot / describe / pile  -> leitura para a API
#   2. apply_manual_count / release_manual_count -> escrita pela via do worker
#   3. diagnose_roi / measure_chair_height        -> ferramentas de calibração
#   4. offset_suggestion / error_statistics       -> qualidade do modelo
#
# Ponto central: este arquivo VALIDA e TRADUZ. Quem conta de verdade é o
# InferenceService (worker) e o app/counting (algoritmo). Se uma regra de
# negócio de contagem precisa mudar, o lugar dela é aqui - não na API.

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
        # store é injetável para os testes: um StateStore falso isola a fachada
        # sem precisar de worker nem de câmera.
        self.store = store or state_store

    # ------------------------------------------------------------- leitura
    def snapshot(self) -> dict[str, Any]:
        """Estado completo de contagem, pronto para JSON."""
        piles = self.store.get_piles()
        totals = self.store.totals_breakdown()
        return {
            # "total" é o confiável (só STABLE/PARTIAL/manual). O Breakdown
            # vai junto para a tela poder mostrar o provisório ao lado.
            "total": self.store.total(only_stable=True),
            "totals": totals,
            "pile_count": len(piles),
            "confidence": round(self.store.overall_confidence(), 4),
            "piles": [self.describe(p) for p in piles],
            # __import__ evita um import datetime no topo só para isto.
            "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }

    def describe(self, pile: PileState) -> dict[str, Any]:
        """Uma pilha com rótulos prontos para o frontend.

        O as_dict() vem cru (números); aqui entra a camada de apresentação:
        texto, cor e a flag que faz a pilha piscar na tela.
        """
        data = pile.as_dict()
        data["status_label"] = status_label(pile.status)
        # Hex, não BGR: quem pinta é o CSS do navegador, não o OpenCV.
        data["color"] = status_color_hex(pile.status)
        # PARTIAL entra em needs_attention mesmo sendo contada no total:
        # o operador precisa olhar uma pilha cortada pela borda da imagem.
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
        # Teto de 10000 é folgado de propósito: o objetivo é barrada de
        # digitação boba, não validar fisicamente. 0 é aceito (pilha vazia
        # é um evento real na operação).
        if value < 0 or value > 10000:
            return {"ok": False, "message": "Contagem fora do intervalo (0-10000)."}
        if inference is None:
            # Worker não subiu (modelo ausente, câmera fora). A pilha continua
            # existindo no store, mas não há quem a traverse.
            return {"ok": False, "message": "Nenhum worker de inferência ativo."}
        # O worker cuida do lock, do registro em count_corrections e do
        # snapshot. Aqui é só a validação de entrada.
        return inference.manual_count(pile_id, value)

    def release_manual_count(self, inference: Any, pile_id: int) -> dict[str, Any]:
        """Devolve a pilha ao controle da IA após uma correção manual.

        "Destravar" tem duas metades: o StabilityFilter (que ficou com o
        valor fixado) e as flags `manual` do runtime e do PileState. Esquecer
        uma delas deixa a pilha travada para sempre ou faz a IA sobrescrever
        o valor do operador.
        """
        if inference is None:
            return {"ok": False, "message": "Nenhum worker de inferência ativo."}
        # unlock() limpa o lock e a candidata da janela de estabilidade.
        inference.stability.unlock(pile_id)
        runtime = inference._piles.get(pile_id)
        if runtime is not None:
            # _piles é "privado" do worker, mas este serviço é o ponto de
            # integração autorizado: evitamos duplicar essa estrutura aqui.
            runtime.manual = False
        pile = self.store.get_pile(pile_id)
        if pile is not None:
            pile.manual = False
        # Reindexa para os listeners do WS perceberem a mudança.
        self.store.set_piles(self.store.get_piles())
        log.info("Pilha %s voltou ao controle automático", pile_id)
        return {"ok": True, "pile_id": pile_id, "message": f"Pilha {pile_id} voltou ao modo automático."}

    # ---------------------------------------------------------- diagnóstico
    def diagnose_roi(
        self, frame: np.ndarray | None, roi: dict[str, Any], chair_height_px: float, count_offset: int = 0
    ) -> dict[str, Any]:
        """Analisa uma região e mostra o detalhe do que o algoritmo enxerga.

        É a ferramenta de depuração do lado certo: em vez de chutar a
        CHAIR_HEIGHT_PX, o operador desenha a caixa de UMA cadeira e lê as
        métricas (período, picos, número de camadas) que o algoritmo viu.
        Roda sobre o frame real, com um StackCounter descartável - não toca no
        contador do worker nem no estado global.
        """
        if frame is None:
            return {"ok": False, "message": "Nenhum frame disponível ainda."}
        try:
            x, y, w, h = (int(roi[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": "Informe a ROI: x, y, w, h."}
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        # Limita ao frame: uma ROI desenhada meio fora da imagem é comum
        # quando o operador arrasta com o zoom do navegador.
        w, h = min(w, fw - x), min(h, fh - y)
        if w < 8 or h < 8:
            # 8 px é o piso: abaixo disso não há camada nenhuma a detectar e
            # a análise só retornaria ruído.
            return {"ok": False, "message": "ROI muito pequena."}
        crop = frame[y : y + h, x : x + w]
        counter = StackCounter(chair_height_px=chair_height_px, count_offset=count_offset)
        est = counter.count(crop, chair_height_px=chair_height_px)
        return {
            "ok": True,
            # ROI já corrigida/clampada, para o frontend desenhar a mesma
            # caixa que o backend analisou.
            "roi": {"x": x, "y": y, "w": w, "h": h},
            "estimate": est.as_dict(),
            "analysis": counter.diagnose(crop, chair_height_px),
        }

    def measure_chair_height(self, frame: np.ndarray | None, roi: dict[str, Any]) -> dict[str, Any]:
        """Mede a altura de uma cadeira em pixels a partir de uma ROI.

        A premissa é do operador, não do algoritmo: se a caixa desenhada for
        exatamente uma cadeira, a ALTURA dela em pixels é a CHAIR_HEIGHT_PX que
        os métodos por tamanho e por período precisam.
        """
        if frame is None:
            return {"ok": False, "message": "Nenhum frame disponível ainda."}
        try:
            x, y, w, h = (int(roi[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": "Informe a ROI da cadeira: x, y, w, h."}
        fh, fw = frame.shape[:2]
        # Aqui, diferente de diagnose_roi, a ROI precisa estar DENTRO do frame:
        # medir a altura de um recorte cortado daria uma altura falsa.
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            return {"ok": False, "message": "A ROI está fora da imagem."}
        crop = frame[y : y + h, x : x + w]
        # StackCounter() sem altura: o diagnóstico serve justamente para
        # descobrir a altura, não para aplicá-la.
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
        """Sugere o offset de contagem a partir das correções manuais.

        Pares são ``(contagem_da_ia, contagem_do_operador)``. A função
        delegate só sugere com evidência suficiente (ver suggest_offset);
        sem amostra suficiente ela devolve offset=0 em vez de chutar.
        """
        return suggest_offset(corrections)

    def error_statistics(self, corrections: list[tuple[int, int]]) -> dict[str, Any]:
        """Erro real do modelo medido nas correções do operador.

        É a única métrica de qualidade que não depende de annotated test set:
        mede o erro contra a verdade humana, no ambiente real. Serve para
        dizer se vale a pena treinar de novo.
        """
        if not corrections:
            # Sem amostra não devolve 0 (que pareceria "erro zero"):
            # devolve None, que o frontend mostra como "sem dados".
            return {"samples": 0, "mean_abs_error": None, "exact_match_rate": None,
                    "message": "Nenhuma correção manual registrada."}
        errs = [abs(h - a) for a, h in corrections]
        return {
            "samples": len(errs),
            # Erro médio absoluto, em CADEIRAS. Vale a ordem de grandeza:
            # 1,5 significa "erra uma cadeira e meia por pilha, no promedio".
            "mean_abs_error": round(sum(errs) / len(errs), 3),
            # Fração de acertos exatos: 0.5 = metade das pilhas contadas
            # exatamente igual ao que o operador disse.
            "exact_match_rate": round(sum(1 for e in errs if e == 0) / len(errs), 4),
            # max_abs_error é o pior caso seen: é o número que mostra se o
            # modelo às vezes duplica uma pilha inteira.
            "max_abs_error": max(errs),
        }


# Fachada compartilhada (mesma instância usada pela API)
# Instância única, como no StateStore: a API e os testes precisam falar com a
# mesma fachada, senão uma correção passaria num objeto e o estado leria outro.
counting_service = CountingService()

__all__ = ["CountingService", "counting_service"]
