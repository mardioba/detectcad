"""WebSocket ``/ws``.

Envia em tempo real:

* ``state``  - snapshot completo (pilhas, totais, confiança, status da câmera)
* ``frame``  - opcionalmente, o frame JPEG anotado em base64
* ``event``  - eventos pontuais (mudança de contagem, erro, alerta)
* ``ping``   - keepalive

O cliente recebe os dados em JSON. O stream de imagem é separado (MJPEG em
``/api/stream.mjpg``) para não duplicar banda do WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.config import settings
from app.services.state_store import state_store

log = logging.getLogger("app")

router = APIRouter()


class ConnectionManager:
    """Gerencia as conexões WebSocket ativas."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        log.info("WebSocket conectado (%d ativos)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        log.info("WebSocket desconectado (%d ativos)", len(self._connections))

    @property
    def count(self) -> int:
        return len(self._connections)

    async def send_json(self, payload: dict[str, Any]) -> None:
        """Envia para todos. Clientes mortos são removidos."""
        if not self._connections:
            return
        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                if ws.client_state is not WebSocketState.CONNECTED:
                    dead.append(ws)
                    continue
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    async def broadcast_event(self, level: str, event: str, message: str) -> None:
        await self.send_json(
            {
                "type": "event",
                "timestamp": payload_timestamp(),
                "level": level,
                "event": event,
                "message": message,
            }
        )


def payload_timestamp() -> str:
    from app.schemas import localnow

    return localnow().isoformat(timespec="seconds")


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, frames: bool = Query(default=False)) -> None:
    """Canal em tempo real do dashboard.

    ``/ws?frames=true`` inclui o JPEG do frame em base64 a cada atualização
    (mais simples, mas mais pesado que o stream MJPEG).
    """
    await manager.connect(ws)
    send_frames = frames
    include_jpeg = False
    last_counter = -1

    try:
        await ws.send_json(
            {
                "type": "hello",
                "message": "Conectado ao Contador de Cadeiras",
                "server_time": payload_timestamp(),
                "include_jpeg": include_jpeg,
            }
        )
        # Snapshot inicial
        await ws.send_json(state_store.snapshot())

        interval = 1.0 / max(0.2, settings.web.websocket_broadcast_hz)
        while True:
            await asyncio.sleep(interval)
            payload = state_store.snapshot()
            if send_frames:
                jpeg, counter = state_store.get_frame_jpeg(last_counter)
                if jpeg:
                    last_counter = counter
                    payload["frame_b64"] = base64.b64encode(jpeg).decode("ascii")
                    payload["frame_counter"] = counter
            await ws.send_json(payload)

    except WebSocketDisconnect:
        pass
    except (asyncio.CancelledError, RuntimeError):
        pass
    except Exception:
        log.exception("Erro no WebSocket")
    finally:
        await manager.disconnect(ws)


@router.get("/ws/status")
async def ws_status() -> dict[str, Any]:
    """Quantas conexões estão abertas (útil em diagnóstico)."""
    return {"connections": manager.count, "broadcast_hz": settings.web.websocket_broadcast_hz}


async def close_all() -> None:
    with contextlib.suppress(Exception):
        for ws in list(manager._connections):
            await ws.close()
        manager._connections.clear()


__all__ = ["router", "manager", "websocket_endpoint", "close_all"]
