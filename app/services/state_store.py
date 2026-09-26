"""Estado compartilhado entre os workers de IA e a API/WebSocket.

Um único ponto de verdade para:

* último frame processado (com overlay) e sua versão JPEG;
* estado de cada pilha (contagem, confiança, status);
* totais;
* status da câmera e do modelo;
* histórico recente para o gráfico.

Os workers escrevem; a API e o WebSocket leem. Tudo com ``RLock`` porque os
dois lados rodam em threads diferentes (FastAPI em threadpool + workers
dedicados).
"""

# ARQUIVO / MAPA
# Estado global em memória. Não é o banco: é o retrato de "agora" que o
# dashboard lê. Se o processo reiniciar, tudo aqui se perde de propósito
# (o histórico real está no SQLite).
#
# Ordem de leitura:
#   1. frames     -> set_frame / get_frame_jpeg / get_frame_bgr
#   2. pilhas     -> set_piles / get_pile / total / totals_breakdown
#   3. histórico  -> push_history / get_history (deque limitado)
#   4. status     -> set_status / get_status (dict mesclado)
#   5. eventos    -> push_event / get_events (deque limitado)
#   6. listeners  -> _notify (fan-out para o WebSocket)
#   7. snapshot   -> o payload JSON completo do WS
#
# Duas linhas de regra que valem para o arquivo todo:
#   * nunca guardar o dict de pilhas sem passar por set_piles;
#   * leitura que vai para a API/JSON usa deepcopy (get_status) ou devolve
#     cópias de listas, para o consumidor nunca ver o estado mudando no meio
#     da serialização.

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from app.schemas import CameraState, CountStatus, PileState, localnow

log = logging.getLogger("app")

MAX_HISTORY = 720  # pontos no gráfico em memória


class StateStore:
    """Estado global do sistema, seguro para leitura/escrita concorrente."""

    def __init__(self) -> None:
        # RLock (e não Lock) porque get_status/get_piles são chamados de dentro
        # de snapshot(), que por sua vez é chamado já com o lock em algumas
        # rotas: um Lock não reentrante deadlockaria.
        self._lock = threading.RLock()
        # JPEG: o que o dashboard baixa. BGR: o array numpy, usado por quem
        # precisa da imagem sem recomprimir (ex.: snapshot de correção manual).
        self._frame_jpeg: bytes | None = None
        self._frame_bgr: Any = None
        self._frame_ts: float = 0.0
        # Contador monotônico de frames. O endpoint de stream compara com o
        # valor que o cliente já tem e só retransmite se mudou: é assim que se
        # evita reenviar a mesma imagem várias vezes por segundo.
        self._frame_counter: int = 0
        # Índice por pile_id. Substitui o dict inteiro a cada frame, em vez de
        # mutar: quem já estava lendo uma lista antiga continua com ela.
        self._piles: dict[int, PileState] = {}
        # True enquanto o modelo em uso foi treinado para este ambiente. O
        # worker reescreve a cada frame. False = as contagens dos estimadores
        # são diagnóstico, não medição, e o total oficial fica em zero.
        # Default True para não mudar o comportamento de quem usa o store
        # fora do worker (testes, utilitários).
        self._counts_measurable: bool = True
        # deques com maxlen: a memória não cresce sem limite e o descarte do
        # item mais antigo é automático (não precisa de cleanup manual).
        self._history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self._events: deque[dict[str, Any]] = deque(maxlen=200)
        # Dict de status é MESCLADO, não substituído: cada chamada escreve só
        # os campos que ela sabe. Por isso "started_at" continua valendo.
        self._status: dict[str, Any] = {
            "mode": "starting",
            "started_at": localnow().isoformat(timespec="seconds"),
        }
        self._listeners: list[Callable[[str, Any], None]] = []
        # Serial de mudança: o WebSocket só reenvia o snapshot inteiro quando
        # este número sobe. Evita_diff de alguns KB por frame.
        self._change_serial: int = 0

    # ------------------------------------------------------------------ frames
    def set_frame(self, jpeg: bytes, frame_bgr: Any, timestamp: float) -> None:
        """Publica a última imagem processada. Chamado só pelo worker."""
        with self._lock:
            self._frame_jpeg = jpeg
            self._frame_bgr = frame_bgr
            self._frame_ts = timestamp
            self._frame_counter += 1

    def get_frame_jpeg(self, min_counter: int = 0) -> tuple[bytes | None, int]:
        """JPEG novo, ou ``None`` se nada mudou desde ``min_counter``.

        O cliente manda o contador que recebeu antes. Devolver ``None`` em vez
        de uma cópia é o que faz o stream não gastar banda à toa quando a
        câmera está parada.
        """
        with self._lock:
            if self._frame_counter > min_counter:
                return self._frame_jpeg, self._frame_counter
            return None, self._frame_counter

    def get_frame_bgr(self) -> Any:
        with self._lock:
            return self._frame_bgr

    @property
    def frame_counter(self) -> int:
        with self._lock:
            return self._frame_counter

    @property
    def frame_timestamp(self) -> float:
        with self._lock:
            return self._frame_ts

    # ------------------------------------------------------------------ pilhas
    def set_piles(self, piles: list[PileState]) -> None:
        """Substitui o retrato de pilhas. Chamado uma vez por frame."""
        with self._lock:
            # Reconstrói o dict a partir da lista: a ordem de inserção vira
            # "ordem em que a câmera mostra as pilhas".
            self._piles = {p.pile_id: p for p in piles}

    def get_piles(self) -> list[PileState]:
        with self._lock:
            # Cópia rasa da lista: o dict interno pode ser trocado no frame
            # seguinte sem afetar quem já pegou esta lista.
            return list(self._piles.values())

    def get_pile(self, pile_id: int) -> PileState | None:
        with self._lock:
            # Devolve a referência, não cópia: manual_count() muta o PileState
            # de propósito (corrige o valor no lugar) e depois chama set_piles.
            return self._piles.get(pile_id)

    @property
    def pile_count(self) -> int:
        with self._lock:
            return len(self._piles)

    def set_counts_measurable(self, measurable: bool) -> None:
        """O worker avisa se as contagens do frame são medições.

        False = não há modelo treinado para este ambiente, então os números
        dos estimadores são diagnóstico, não contagem (ver
        InferenceService._counts_are_measurable). Com False, `total()` só
        soma o que o OPERADORcorrigiu à mão, porque o número humano é a
        única verdade disponível nesse regime.

        O default é True de propósito: um StateStore usado fora do worker
        (testes, utilitários) continua com o comportamento antigo em vez de
        passar a devolver zero sem ninguém ter pedido.
        """
        with self._lock:
            self._counts_measurable = bool(measurable)

    def _trusted(self, pile: PileState) -> bool:
        """Esta pilha entra no total "em que a operação confia"?"""
        if pile.manual:
            return True
        if not self._counts_measurable:
            return False
        return pile.status in (CountStatus.STABLE, CountStatus.PARTIAL)

    def total(self, only_stable: bool = True) -> int:
        """Total de cadeiras.

        Por padrão soma apenas as pilhas em estado STABLE ou PARTIAL - é o
        número em que a operação confia. Contagens LOW_CONFIDENCE/UNKNOWN
        são exibidas separadamente no dashboard.
        """
        with self._lock:
            total = 0
            for pile in self._piles.values():
                # Pilha manual (operador corrigiu) entra SEMPRE, em qualquer
                # modo: o número humano é a verdade, não passa pelo filtro de
                # estabilidade nem pelo status.
                if pile.manual:
                    total += pile.stable_count
                    continue
                if not only_stable:
                    total += pile.stable_count
                elif self._trusted(pile):
                    total += pile.stable_count
            return total

    def totals_breakdown(self) -> dict[str, Any]:
        """O total em três versões, para a tela mostrar "N (+M instável)".

        total_stable  - em que a operação pode confiar (STABLE/PARTIAL/manual)
        total_tentative - o resto, exibido separadamente como provisório
        total_all     - soma dos dois; útil para saber o pior caso
        """
        with self._lock:
            piles = list(self._piles.values())
            stable = sum(p.stable_count for p in piles if self._trusted(p))
            tentative = sum(p.stable_count for p in piles if not self._trusted(p))
            return {
                "total_stable": stable,
                "total_tentative": tentative,
                "total_all": stable + tentative,
                "pile_count": len(piles),
            }

    def overall_confidence(self) -> float:
        """Confiança geral = média ponderada pelo número de cadeiras.

        Ponderar por ``max(1, stable_count)`` faz a confiança de uma pilha de
        30 cadeiras pesar 30x a de uma pilha de 1 cadeira, que praticamente não
        afeta o total. O piso de 1 evita divisão por zero em pilha vazia.
        A lista é tirada dentro do lock, mas a conta fica fora: segurar o lock
        durante a soma travaria o worker à toa.
        """
        with self._lock:
            piles = list(self._piles.values())
        if not piles:
            return 0.0
        num = sum(p.confidence * max(1, p.stable_count) for p in piles)
        den = sum(max(1, p.stable_count) for p in piles)
        return num / den if den else 0.0

    # ---------------------------------------------------------------- histórico
    def push_history(self, point: dict[str, Any]) -> None:
        with self._lock:
            self._history.append(point)

    def get_history(self, limit: int = 240) -> list[dict[str, Any]]:
        """Últimos ``limit`` pontos, do mais antigo ao mais recente."""
        with self._lock:
            data = list(self._history)
        # tail e não head: o gráfico lê da esquerda para a direita.
        return data[-limit:]

    # ------------------------------------------------------------------ status
    def set_status(self, **fields: Any) -> None:
        """Atualiza campos soltos do status. Não substitui o dict inteiro."""
        with self._lock:
            self._status.update(fields)
            # updated_at serve para o frontend detectar "o worker travou"
            # mesmo sem o frame_counter mudar.
            self._status["updated_at"] = time.time()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            # deepcopy: este dict vai para jsonify(). Sem a cópia, o worker
            # poderia mutar um dict no meio da serialização e o cliente
            # receberia um JSON quebrado.
            return copy.deepcopy(self._status)

    def bump_change(self) -> int:
        """Sinaliza que houve mudança relevante e devolve o novo serial."""
        with self._lock:
            self._change_serial += 1
            return self._change_serial

    @property
    def change_serial(self) -> int:
        with self._lock:
            return self._change_serial

    # ------------------------------------------------------------------ eventos
    def push_event(self, level: str, event: str, message: str) -> None:
        """Registra um evento de log operacional (INFO/WARNING/ERROR)."""
        item = {"timestamp": localnow().isoformat(timespec="seconds"), "level": level, "event": event, "message": message}
        with self._lock:
            self._events.append(item)
        # _notify FORA do lock: um listener lento (o WS pode bloquear no
        # send) não pode segurar o worker enquanto ele conta.
        self._notify("event", item)

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            data = list(self._events)
        # [::-1] no fim: o painel de eventos mostra o mais recente no topo,
        # ao contrário do gráfico, que cresce para a direita.
        return data[-limit:][::-1]

    # ---------------------------------------------------------------- listeners
    def add_listener(self, cb: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def _notify(self, kind: str, payload: Any) -> None:
        # list(...)snapshot: um listener pode se remover durante a chamada e
        # a iteração sobre a lista viva estouraria o índice.
        for cb in list(self._listeners):
            try:
                cb(kind, payload)
            except Exception:  # pragma: no cover
                # Um listener quebrado (cliente WS desconectado) nunca pode
                # derrubar quem chamou push_event.
                log.exception("Listener do StateStore falhou")

    # ------------------------------------------------------------------ resumo
    def snapshot(self) -> dict[str, Any]:
        """Payload completo para o WebSocket.

        Montado a partir de várias chamadas com lock próprio, em vez de ler
        tudo sob um lock só: o snapshot pode mostrar um total de uma frame e as
        pilhas da seguinte, e isso é aceitável para o dashboard. O que não pode
        é derrubar a thread por causa disso.
        """
        piles = self.get_piles()
        status = self.get_status()
        return {
            "type": "state",
            "timestamp": localnow().isoformat(timespec="seconds"),
            "camera_state": status.get("camera_state", CameraState.OFFLINE.value),
            # as_dict() gera um dict novo por pilha: o WS serializa sem
            # enxergar o PileState original.
            "piles": [p.as_dict() for p in piles],
            "pile_count": len(piles),
            "total": self.total(only_stable=True),
            "totals": self.totals_breakdown(),
            "confidence": round(self.overall_confidence(), 4),
            "fps": round(status.get("process_fps", 0.0), 2),
            "inference_ms": status.get("inference_ms", 0.0),
            "model_loaded": status.get("model_loaded", False),
            "model_trained": status.get("model_trained", False),
            "mode": status.get("mode", "starting"),
            "alerts": status.get("alerts", []),
            # O cliente compara com o seu e só pede de novo se mudou.
            "change_serial": self.change_serial,
        }


# Instância única compartilhada por toda a aplicação.
# Criada na importação: qualquer `from app.services.state_store import
# state_store` pega sempre o MESMO objeto, que é o que o worker e a API
# precisam. Criar uma segunda instância aqui deixaria os dois lados sem se ver.
state_store = StateStore()

__all__ = ["StateStore", "state_store", "MAX_HISTORY"]
