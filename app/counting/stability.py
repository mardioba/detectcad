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

# ===========================================================================
# ARQUIVO / MAPA  -  stability.py
# ===========================================================================
# Onde o StackCounter deixa de ser um número e vira o número que aparece na
# tela. Regra central: um frame não muda o que o operador vê.
#
# - StabilityResult (dataclass): o que sai do filtro. Carrega o valor
#   bruto, o estabilizado, os votos e a confiança de estabilidade.
#
# - StabilityFilter: uma JANELA deslizante por pile_id. Estado por pilha:
#     _windows        últimos N valores brutos (deque com maxlen)
#     _displayed      o que está no dashboard agora
#     _candidate      valor novo em teste (ainda não aceito)
#     _candidate_hits quantas leituras seguidas confirmaram o _candidate
#     _locked         pilhas com correção manual - a IA não mexe
#
#   update()  - ponto de entrada por frame. Agrega a janela, decide se o
#               valor exibido pode trocar.
#   force()   - correção manual do operador: trava o valor.
#   unlock()  - devolve a pilha para a IA.
#   remove()/clear() - esquecer pilhas (saiu da cena / reinício).
#
# - _aggregate: como juntar N números em 1 (mode | median | mean).
# - _agreement: quanto a janela concorda consigo mesma, já corrigida pelo
#   quanto a janela está cheia.
# - decide_status: transforma (contagem, confiança, estabilidade) no status
#   mostrado, com prioridade fixa.
#
# Ordem de leitura: _aggregate -> update (regra da troca) -> _agreement ->
# decide_status.
#
# Por que janela e não média móvel simples: a média "escorrega" durante uma
# troca real (20 -> 21 -> 22 aparece como 20,7 -> 21,3 -> 21,9). Com mode, o
# valor só muda quando a MAIORIA da janela já mudou.
# ===========================================================================

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
    """Saída do filtro de estabilidade para uma pilha.

    ``stable_count`` é o que o dashboard mostra; ``raw_count`` é o que o
    StackCounter acabou de ler. Comparar os dois é o jeito mais rápido de
    ver que a pilha está em transição.
    """

    stable_count: int
    raw_count: int
    stability_confidence: float
    # votes = {contagem: ocorrências na janela}. Sai no dict para o painel
    # mostrar "18 (7x), 20 (3x)".
    votes: dict[str, int] = field(default_factory=dict)
    window_size: int = 0
    # agreed = o valor exibido bate com o agregado (tolerância de 1 cadeira).
    agreed: bool = True
    method: str = "mode"

    def as_dict(self) -> dict[str, Any]:
        """Dump para JSON (API/dashboard)."""
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
    """Janela deslizante por pilha.

    Uma instância vive durante a sessão de vídeo. Nada é global: o estado é
    por ``pile_id``, porque duas pilhas no mesmo frame evoluem de forma
    independente.
    """

    def __init__(
        self,
        window: int | None = None,
        mode: str | None = None,
        change_min_frames: int | None = None,
        min_agreement: float | None = None,
    ) -> None:
        # `or` (e não `is None`) em window/change_min_frames: 0 seria
        # configuração inválida, e cair no default é melhor que quebrar.
        self.window = int(window or settings.counting.count_stability_frames)
        self.mode = (mode or settings.counting.stability_mode).lower()
        self.change_min_frames = int(change_min_frames or settings.counting.change_min_frames)
        self.min_agreement = (
            float(min_agreement) if min_agreement is not None else settings.counting.stability_min_confidence
        )
        # maxlen=window: o deque SE APAGA sozinho quando estoura. Por isso
        # _windows não cresce sem limite com o tempo de vídeo.
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

        # Aggregado = a resposta da janela (mode/median/mean).
        aggregated = self._aggregate(win)
        # Votes: contagem de ocorrências de cada valor. É o que permite
        # mostrar no painel por que o número não mudou.
        votes = dict(Counter(win).most_common())
        stability_conf = self._agreement(win, aggregated)

        # Correção manual: o operador manda, a IA não mexe.
        if pile_id in self._locked:
            # A janela CONTINUA encheu com leituras da IA: só o valor
            # exibido fica travado. Isso é deliberado - quando a pilha for
            # destravada (unlock), o histórico recente já mostra o que a IA
            # realmente está enxergando, sem "ressuscitar" leitura velha.
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
            # Mostrar o primeiro número é intencional: no primeiro frame não
            # existe histórico, e esperar 10 frames deixaria a pilha sem
            # número por um instante desnecessário.
            self._displayed[pile_id] = aggregated
            self._candidate[pile_id] = aggregated
            # hits = len(win) (1) já conta como 1 confirmação. Se a janela
            # ainda não encheu, a regra de troca abaixo é a que protege.
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
            # Voltou ao valor exibido: a troca em teste foi cancelada e os
            # hits zeram. Sem isso, uma troca interrompida voltaria a contar
            # do ponto onde parou.
            self._candidate.pop(pile_id, None)
            self._candidate_hits[pile_id] = 0
        else:
            # O valor agregado mudou: conta ocorrências consecutivas.
            if self._candidate.get(pile_id) == aggregated:
                self._candidate_hits[pile_id] = self._candidate_hits.get(pile_id, 0) + 1
            else:
                # Valor diferente do que estava em teste: recomeça a contagem.
                self._candidate[pile_id] = aggregated
                self._candidate_hits[pile_id] = 1
            hits = self._candidate_hits[pile_id]
            # Duas formas de aceitar: (a) change_min_frames leituras seguidas
            # (padrão 3), ou (b) 60% da janela já votando no novo valor. (b)
            # evita que uma janela grande (N=30) demore 30 frames para
            # acompanhar uma troca real e rápida.
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
            # agreed = 1 cadeira de tolerância porque arredondar extensão
            # pode oscilar entre 19 e 20 numa pilha de 20 sem que nada tenha
            # mudado de fato.
            agreed=abs(displayed - aggregated) <= 1,
            method=self.mode,
        )

    def get(self, pile_id: int) -> int | None:
        """Valor exibido (ou None se a pilha nunca foi lida)."""
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
            # Enche a janela com o valor do operador. Efeito colateral
            # desejado: a concordância sobe para ~1.0 e o status vira STABLE,
            # porque de fato aquele valor está "acertado" por definição.
            win.clear()
            for _ in range(min(self.window, self.change_min_frames)):
                win.append(int(value))
        self._candidate[pile_id] = int(value)
        self._candidate_hits[pile_id] = 0
        self._locked.add(pile_id)

    def unlock(self, pile_id: int) -> None:
        """Libera a pilha: a IA volta a decidir a contagem.

        Zera a janela de propósito: uma janela com leituras antigas misturadas
        à nova realidade atrasaria a convergência.
        """
        self._locked.discard(pile_id)
        win = self._windows.get(pile_id)
        if win is not None:
            win.clear()
        self._candidate.pop(pile_id, None)
        self._candidate_hits[pile_id] = 0

    def is_locked(self, pile_id: int) -> bool:
        """A pilha está com correção manual travada?"""
        return pile_id in self._locked

    def remove(self, pile_id: int) -> None:
        """Esquece a pilha (saiu da cena). Libera memória e estado."""
        self._windows.pop(pile_id, None)
        self._displayed.pop(pile_id, None)
        self._candidate.pop(pile_id, None)
        self._candidate_hits.pop(pile_id, None)
        self._locked.discard(pile_id)

    def clear(self) -> None:
        """Zera tudo. Usado quando a sessão de vídeo reinicia."""
        self._windows.clear()
        self._displayed.clear()
        self._candidate.clear()
        self._candidate_hits.clear()
        self._locked.clear()

    def history(self, pile_id: int) -> list[int]:
        """Cópia da janela atual (mais antiga primeiro). Para depurar."""
        return list(self._windows.get(pile_id, []))

    # ------------------------------------------------------------- interno
    def _aggregate(self, values: list[int]) -> int:
        """Junta a janela em UM número, conforme ``self.mode``.

        mode (padrão) é imune a outlier; median também; mean "suaviza" mas
        pode criar um número que nenhuma leitura viu - por isso não é padrão.
        """
        if not values:
            return 0
        if self.mode == "mean":
            return int(round(statistics.fmean(values)))
        if self.mode == "median":
            return int(statistics.median(values))
        # mode (padrão): desempata pelo valor mais recente
        counts = Counter(values)
        top = max(counts.values())
        # tied = todos os valores empatados na maior frequência. Numa janela
        # par com metade 20 e metade 21, os dois entram e o desempate decide.
        tied = {v for v, c in counts.items() if c == top}
        for v in reversed(values):  # reversed -> o mais recente ganha
            if v in tied:
                return int(v)
        # inalcançável com valores não vazios, mas mantém o retorno inteiro.
        return int(values[-1])

    def _agreement(self, values: list[int], aggregated: int) -> float:
        """Concordância da janela: fração de leituras no valor aggregates.

        Também cai um pouco se a janela ainda está enchendo (poucos frames).
        """
        if not values:
            return 0.0
        # Conta quantas leituras bateram exatamente com o agregado.
        agree = sum(1 for v in values if v == aggregated)
        ratio = agree / len(values)
        # fill = 0..1 conforme o preenchimento da janela. Multiplicar por
        # (0.5 + 0.5*fill) impõe que 1 leitura de uma janela de 10 NÃO vale
        # como 100% de concordância: no primeiro frame o máximo é 0.55, e
        # isso é intencional - o filtro precisa de tempo para estabilizar.
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

    A ordem importa: os estados são testes em sequência, e o primeiro que
    casa vence. "Não sei" (UNKNOWN) ganha de tudo; depois "não tenho certeza
    do número" (LOW_CONFIDENCE); depois "o número existe mas está mudando"
    (UNSTABLE). Só um número que é confiável, estável e completo vira STABLE.
    """
    min_conf = settings.counting.min_confidence if min_confidence is None else min_confidence
    stab_min = settings.counting.stability_min_confidence if stability_min is None else stability_min
    min_chairs = settings.counting.min_chairs_for_valid if min_chairs is None else min_chairs

    if count <= 0:
        # count=0 nunca é "pilha vazia" neste sistema: é o código de
        # "nenhum estimador respondeu" (ver StackCounter.count).
        return CountStatus.UNKNOWN
    if confidence < min_conf:
        # Confiança do contador, já com detecção + estabilidade + acordo entre
        # estimadores combinados (ver confidence.combine).
        return CountStatus.LOW_CONFIDENCE
    if stability_confidence < stab_min:
        # A janelinha não fecha: pode ser pilha sendo mexida OU leitura
        # instável. Nos dois casos o número não deve ser levado a sério.
        return CountStatus.UNSTABLE
    if partial or count < min_chairs:
        # parcial = a ROI saiu da imagem (pilha cortada), ou a pilha tem menos
        # cadeiras que o mínimo válido do negócio.
        return CountStatus.PARTIAL
    return CountStatus.STABLE


__all__ = ["StabilityFilter", "StabilityResult", "decide_status"]
