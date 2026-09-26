"""WebSocket ``/ws``.

Envia em tempo real:

* ``state``  - snapshot completo (pilhas, totais, confiança, status da câmera)
* ``frame``  - opcionalmente, o frame JPEG anotado em base64
* ``event``  - eventos pontuais (mudança de contagem, erro, alerta)
* ``ping``   - keepalive

O cliente recebe os dados em JSON. O stream de imagem é separado (MJPEG em
``/api/stream.mjpg``) para não duplicar banda do WebSocket.
"""

# ============================================================================
# ARQUIVO / MAPA  -  app/api/websocket.py
# ============================================================================
# Canal em tempo real do dashboard. Complementa o REST: enquanto as rotas de
# app/api/routes.py respondem a perguntas ("qual o total?"), aqui o servidor
# empurra o estado sem o cliente precisar perguntar.
#
# ORDEM DE LEITURA:
#   1. ConnectionManager  conjunto de sockets abertos + lock asyncio
#   2. payload_timestamp  carimbo de tempo usado em todos os payloads
#   3. manager            instância única (importada por main.py)
#   4. /ws                o handler do canal
#   5. /ws/status         quantas conexões estão abertas
#   6. close_all          fechamento no shutdown
#
# POR QUE A IMAGEM NÃO VEM AQUI:
#   O JPEG é pesado e não muda de forma útil entre dois updates de estado.
#   Mandá-lo junto dobra a banda do WebSocket sem agregar informação. Por
#   isso a imagem vai por MJPEG (app/api/routes.py) e o WebSocket carrega só
#   os números. A flag `?frames=true` existe para depurar/testar, não para o
#   uso normal.
#
# CUIDADO COM O EVENT LOOP:
#   Tudo aqui é `async def` e roda no event loop. NUNCA usar `time.sleep`,
#   `requests` ou I/O de disco bloqueante aqui - isso travaria todas as
#   outras conexões, inclusive as do dashboard.
# ============================================================================

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


# Nota sobre este router: ele NÃO tem `prefix="/api"` (ao contrário do de
# routes.py). O WebSocket fica em `/ws` na raiz porque o caminho é declarado
# em @router.websocket("/ws") - e é o que o dashboard.js espera.


class ConnectionManager:
    """Gerencia as conexões WebSocket ativas.

    `asyncio.Lock` (e não `threading.Lock`) porque todos os usuários deste
    objeto rodam no mesmo event loop. O lock protege a coleção em dois
    momentos: alguém conecta enquanto outro está enviando.
    """

    def __init__(self) -> None:
        # `set` e não `list`: não há ordem de entrega a respeitar e a
        # busca por socket morto é O(1).
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        # `accept()` antes do lock: o handshake precisa completar primeiro.
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        log.info("WebSocket conectado (%d ativos)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        # `discard` e não `remove`: o socket pode já ter sido removido pelo
        # `send_json` quando percebeu que estava morto. Chamar disconnect
        # duas vezes não pode levantar KeyError.
        async with self._lock:
            self._connections.discard(ws)
        log.info("WebSocket desconectado (%d ativos)", len(self._connections))

    @property
    def count(self) -> int:
        # Leitura sem lock: `len()` de um set é atômico e um número meio
        # desatualizado não faz mal nenhum num diagnóstico.
        return len(self._connections)

    async def send_json(self, payload: dict[str, Any]) -> None:
        """Envia para todos. Clientes mortos são removidos.

        Um cliente que fechou a aba (sem mandar close) continua no set até
        aparecer um erro. Enviar para ele levanta exceção - e essa exceção
        NÃO pode derrubar o broadcast dos outros, então ela é capturada e o
        socket marcado como morto.
        """
        if not self._connections:
            return
        # `default=str` evita que um objeto não serializável (Path, enum,
        # numpy) quebre o broadcast inteiro.
        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            # `list(...)` cria a cópia ANTES do loop: se outro handler
            # desconectar durante o `await` abaixo, a coleção não muda no
            # meio da iteração (que levantaria RuntimeError).
            try:
                # Checagem de estado ANTES do envio: evita a exceção e o
                # custo de uma escrita que não vai chegar a lugar nenhum.
                if ws.client_state is not WebSocketState.CONNECTED:
                    dead.append(ws)
                    continue
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        if dead:
            # Remove os mortos só DEPOIS do loop, sob o lock. Limpar durante
            # a iteração invalidaria a cópia que está sendo percorrida.
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    async def broadcast_event(self, level: str, event: str, message: str) -> None:
        """Empurra um evento pontual (contagem mudou, erro, alerta).

        Chamado de uma THREAD do worker via `asyncio.run_coroutine_threadsafe`
        (ver o lifespan em main.py). É por isso que este método é corrotina:
        o worker não tem como rodar `await` diretamente, então entrega a
        corrotina ao loop.
        """
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


# Instância única: main.py importa este mesmo objeto para registrar o
# listener do state_store e fechar as conexões no shutdown.
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
        # Envia um estado completo logo de cara, para a tela não ficar
        # vazia enquanto o primeiro `while` ainda está esperando o intervalo.
        # O frontend pode renderizar esse snapshot imediatamente.

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
        # CancelledError: o servidor está shutting down. RuntimeError: enviar
        # num socket já fechado. Nenhum dos dois é erro do usuário, então são
        # engolidos sem log para não poluir o app.log.
        pass
    except Exception:
        log.exception("Erro no WebSocket")
    finally:
        # `finally` garante a desconexão em TODOS os caminhos, inclusive
        # cancelamento. Sem isso, um cliente que caiu sem handshake limpo
        # ficaria listado em `manager._connections` para sempre.
        await manager.disconnect(ws)


@router.get("/ws/status")
async def ws_status() -> dict[str, Any]:
    """Quantas conexões estão abertas (útil em diagnóstico).

    Se este número não voltar a zero depois de todo mundo fechar a aba, há
    socket vazado no `ConnectionManager`.
    """
    return {"connections": manager.count, "broadcast_hz": settings.web.websocket_broadcast_hz}


async def close_all() -> None:
    """Fecha todas as conexões. Chamado no shutdown (lifespan em main.py).

    Sem isso, o Uvicorn ficaria esperando as conexões WebSocket morrerem
    sozinhas no shutdown, adicioneando atraso ou travamento no restart.
    """
    # `suppress(Exception)`: durante o shutdown os sockets já podem estar
    # quebrados, e não vale a pena falhar o encerramento por causa disso.
    with contextlib.suppress(Exception):
        for ws in list(manager._connections):
            await ws.close()
        # Limpa o set para que /ws/status não reporte conexões fantasma
        # durante o desligamento.
        manager._connections.clear()


__all__ = ["router", "manager", "websocket_endpoint", "close_all"]
