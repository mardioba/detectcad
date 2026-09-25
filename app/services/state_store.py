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
        self._lock = threading.RLock()
        self._frame_jpeg: bytes | None = None
        self._frame_bgr: Any = None
        self._frame_ts: float = 0.0
        self._frame_counter: int = 0
        self._piles: dict[int, PileState] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self._events: deque[dict[str, Any]] = deque(maxlen=200)
        self._status: dict[str, Any] = {
            "mode": "starting",
            "started_at": localnow().isoformat(timespec="seconds"),
        }
        self._listeners: list[Callable[[str, Any], None]] = []
        self._change_serial: int = 0

    # ------------------------------------------------------------------ frames
    def set_frame(self, jpeg: bytes, frame_bgr: Any, timestamp: float) -> None:
        with self._lock:
            self._frame_jpeg = jpeg
            self._frame_bgr = frame_bgr
            self._frame_ts = timestamp
            self._frame_counter += 1

    def get_frame_jpeg(self, min_counter: int = 0) -> tuple[bytes | None, int]:
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
        with self._lock:
            self._piles = {p.pile_id: p for p in piles}

    def get_piles(self) -> list[PileState]:
        with self._lock:
            return list(self._piles.values())

    def get_pile(self, pile_id: int) -> PileState | None:
        with self._lock:
            return self._piles.get(pile_id)

    @property
    def pile_count(self) -> int:
        with self._lock:
            return len(self._piles)

    def total(self, only_stable: bool = True) -> int:
        """Total de cadeiras.

        Por padrão soma apenas as pilhas em estado STABLE ou PARTIAL - é o
        número em que a operação confia. Contagens LOW_CONFIDENCE/UNKNOWN
        são exibidas separadamente no dashboard.
        """
        with self._lock:
            total = 0
            for pile in self._piles.values():
                if pile.manual:
                    total += pile.stable_count
                    continue
                if not only_stable:
                    total += pile.stable_count
                elif pile.status in (CountStatus.STABLE, CountStatus.PARTIAL):
                    total += pile.stable_count
            return total

    def totals_breakdown(self) -> dict[str, Any]:
        with self._lock:
            piles = list(self._piles.values())
            stable = sum(
                p.stable_count
                for p in piles
                if p.status in (CountStatus.STABLE, CountStatus.PARTIAL) or p.manual
            )
            tentative = sum(
                p.stable_count
                for p in piles
                if p.status not in (CountStatus.STABLE, CountStatus.PARTIAL)
            )
            return {
                "total_stable": stable,
                "total_tentative": tentative,
                "total_all": stable + tentative,
                "pile_count": len(piles),
            }

    def overall_confidence(self) -> float:
        """Confiança geral = média ponderada pelo número de cadeiras."""
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
        with self._lock:
            data = list(self._history)
        return data[-limit:]

    # ------------------------------------------------------------------ status
    def set_status(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)
            self._status["updated_at"] = time.time()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def bump_change(self) -> int:
        with self._lock:
            self._change_serial += 1
            return self._change_serial

    @property
    def change_serial(self) -> int:
        with self._lock:
            return self._change_serial

    # ------------------------------------------------------------------ eventos
    def push_event(self, level: str, event: str, message: str) -> None:
        item = {"timestamp": localnow().isoformat(timespec="seconds"), "level": level, "event": event, "message": message}
        with self._lock:
            self._events.append(item)
        self._notify("event", item)

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            data = list(self._events)
        return data[-limit:][::-1]

    # ---------------------------------------------------------------- listeners
    def add_listener(self, cb: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def _notify(self, kind: str, payload: Any) -> None:
        for cb in list(self._listeners):
            try:
                cb(kind, payload)
            except Exception:  # pragma: no cover
                log.exception("Listener do StateStore falhou")

    # ------------------------------------------------------------------ resumo
    def snapshot(self) -> dict[str, Any]:
        """Payload completo para o WebSocket."""
        piles = self.get_piles()
        status = self.get_status()
        return {
            "type": "state",
            "timestamp": localnow().isoformat(timespec="seconds"),
            "camera_state": status.get("camera_state", CameraState.OFFLINE.value),
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
            "change_serial": self.change_serial,
        }


# Instância única compartilhada por toda a aplicação.
state_store = StateStore()

__all__ = ["StateStore", "state_store", "MAX_HISTORY"]
