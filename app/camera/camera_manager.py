"""Gerenciador de câmeras.

Suporta N câmeras desde o início (multicâmera), mesmo com o caso de uso
principal sendo uma única câmera. Cada câmera tem seu próprio
:class:`FrameSource` e seu próprio worker de inferência.
"""

# ARQUIVO / MAPA - camera_manager.py
# O que faz: registro de câmeras (id -> runtime) e o "status agregado" que o
#   dashboard mostra.
# Ordem de leitura:
#   1. CameraRuntime        -> fonte + último frame de uma câmera
#   2. CameraManager.__init__ -> dicionário e trava
#   3. add / get / remove   -> ciclo de vida
#   4. start_default / stop_all -> subiram e derrubam tudo
#   5. status               -> o que a API devolve
# Quem lê frame NÃO é este arquivo: é o worker de IA, via
#   runtime.take_latest(). O manager só registra câmeras e publica estado.
# Ponto de atenção: a trava do manager é RLock porque start_default() chama
#   add(), e os dois já seguram a trava; reentrante evita se matar sozinho.

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from app.camera.rtsp_camera import FrameInfo, FrameSource
from app.config import settings
from app.schemas import CameraState, SourceMode

log = logging.getLogger("camera")


class CameraRuntime:
    """Uma câmera em execução: fonte + estado + trava de reuso de frame."""

    def __init__(self, source: FrameSource, camera_id: int | None = None, name: str = "") -> None:
        self.source = source
        self.camera_id = camera_id
        self.name = name or source.name
        # Cache do ÚLTIMO frame processado. Serve para o dashboard mostrar
        # imagem mesmo quando a câmera já caiu (a fila da fonte está vazia).
        self._last_frame: tuple[Any, FrameInfo] | None = None
        # Lock simples (não precisa ser RLock aqui: nada chama dois métodos
        # que travam o mesmo objeto aninhado).
        self._lock = threading.Lock()

    def take_latest(self) -> tuple[Any, FrameInfo] | None:
        """Consome o próximo frame real da fonte (não bloqueia a UI)."""
        # Timeout curtíssimo (50 ms) de propósito: o loop de inferência roda a
        # process_fps (3 fps por padrão) e não pode ficar preso esperando frame.
        # Atenção ao nome: aqui não existe polling nem "último frame" garantido.
        # É só um read() na fila, que já descarta quadro velho. Quem quiser
        # pular quadro de propósito usa FrameSource.latest() (esvazia a fila).
        item = self.source.read(timeout=0.05)
        if item is not None:
            with self._lock:
                self._last_frame = item
        return item

    def last_frame(self) -> tuple[Any, FrameInfo] | None:
        """Último frame visto, ou None se a câmera nunca entregou nada."""
        with self._lock:
            return self._last_frame

    def status(self) -> dict[str, Any]:
        """Status da câmera para a API, já com o tamanho real do último frame."""
        info = self.last_frame()
        data = self.source.stats.as_dict()
        if info is not None:
            # O tamanho do frame real ganha do da câmera: alguns RTSP
            # anunciam dimensões erradas até o primeiro keyframe.
            data["width"] = info[0].shape[1]
            data["height"] = info[0].shape[0]
        return data


class CameraManager:
    """Registro de câmeras ativas.

    ``start_default()`` sobe a câmera definida no ``.env``. As demais podem ser
    adicionadas depois pela API (dashboard) ou por código.
    """

    def __init__(self) -> None:
        # id da câmera -> runtime. Dict para add/remove em O(1) e para o
        # dashboard poder listar sem varrer thread nenhuma.
        self._cameras: dict[int, CameraRuntime] = {}
        self._default_id: int = 1
        # RLock: start_default() chama add(), que também trava. RLock deixa a
        # reentrada passar; com Lock comum isso viraria deadlock.
        self._lock = threading.RLock()
        self._event_callback: Callable[[str, str, str], None] | None = None

    # ------------------------------------------------------------------ setup
    def set_event_callback(self, cb: Callable[[str, str, str], None]) -> None:
        """Liga o log de eventos das câmeras que JÁ existem e das futuras."""
        self._event_callback = cb
        # Repassa às câmeras criadas antes da chamada; add() cuida das futuras.
        for runtime in self._cameras.values():
            runtime.source.set_event_callback(cb)

    def add(
        self,
        camera_id: int,
        *,
        url: str = "",
        name: str = "",
        mode: SourceMode = SourceMode.RTSP,
        start: bool = True,
        loop_video: bool = True,
    ) -> CameraRuntime:
        """Registra (e opcionalmente sobe) uma câmera. Idempotente por id."""
        with self._lock:
            # Id já existente devolve a mesma câmera em vez de criar outra:
            # dois apertos no botão do dashboard não podem abrir dois
            # VideoCapture no mesmo RTSP.
            if camera_id in self._cameras:
                return self._cameras[camera_id]
            # Todo o operacional vem do .env, e não de argumento, para que o
            # comportamento de produção e de teste seja o mesmo.
            source = FrameSource(
                url=url or settings.camera.camera_rtsp_url,
                name=name or settings.camera.camera_name,
                mode=mode,
                open_timeout_sec=settings.camera.open_timeout_sec,
                read_timeout_sec=settings.camera.read_timeout_sec,
                retry_delay_sec=settings.camera.retry_delay_sec,
                max_retries=settings.camera.max_retries,
                transport=settings.camera.transport,
                buffer_size=settings.camera.frame_buffer_size,
            )
            # Loop ligado por padrão: em teste, um vídeo curto nunca "termina"
            # e o pipeline continua recebendo frames.
            source.set_video_loop(loop_video)
            if self._event_callback is not None:
                source.set_event_callback(self._event_callback)
            runtime = CameraRuntime(source=source, camera_id=camera_id, name=name or settings.camera.camera_name)
            self._cameras[camera_id] = runtime
            if start:
                # start() fora daqui deixaria a trava livre, mas a thread de
                # captura já é separada; segurar a trava evita corrida entre
                # duas threads adicionando câmeras ao mesmo tempo.
                runtime.source.start()
            return runtime

    def get(self, camera_id: int) -> CameraRuntime | None:
        """Pega uma câmera pelo id, ou None se não existir."""
        with self._lock:
            return self._cameras.get(camera_id)

    def default_id(self) -> int:
        """Id da câmera principal."""
        with self._lock:
            return self._default_id

    def set_default(self, camera_id: int) -> None:
        """Define a câmera principal. Ignora id desconhecido."""
        with self._lock:
            if camera_id in self._cameras:
                self._default_id = camera_id

    def list(self) -> list[CameraRuntime]:
        """Cópia da lista de câmeras (a API itera fora da trava)."""
        with self._lock:
            return list(self._cameras.values())

    def remove(self, camera_id: int) -> bool:
        """Tira a câmera do registro e para a captura."""
        with self._lock:
            runtime = self._cameras.pop(camera_id, None)
        if runtime is None:
            return False
        # stop() fora da trava: ele espera a thread de captura morrer (até
        # 5 s) e travar o manager nesse tempo deixaria a API inteira sem
        # resposta.
        runtime.source.stop()
        log.info("Câmera %s removida", camera_id)
        return True

    def start_default(self) -> CameraRuntime:
        """Sobe a câmera principal vinda do .env (ou a primeira cadastrada)."""
        with self._lock:
            if self._cameras:
                rid = self._default_id
                runtime = self._cameras.get(rid)
                if runtime is not None:
                    # Só chama start() se a fonte não está aberta: `opened`
                    # é o teste barato para "já está rodando" e evita criar
                    # thread duplicada quando o worker chama isto a cada ciclo.
                    if not runtime.source.stats.opened:
                        runtime.source.start()
                    return runtime
            # Registro vazio (primeiro start do servidor): cria a do .env.
            return self.add(
                self._default_id,
                url=settings.camera.camera_rtsp_url,
                name=settings.camera.camera_name,
                mode=SourceMode.RTSP,
            )

    def stop_all(self) -> None:
        """Esvazia o registro e para todas as capturas."""
        with self._lock:
            runtimes = list(self._cameras.values())
            self._cameras.clear()
        # Igual ao remove(): parar fora da trava, senão o dashboard espera.
        for runtime in runtimes:
            runtime.source.stop()

    # ----------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        """Resumo das câmeras para a API: estado agregado + uma entrada por câmera."""
        runtimes = self.list()
        if not runtimes:
            # Formato vazio já no formato completo: o dashboard nunca precisa
            # tratar "sem câmeras" como um caso especial.
            return {
                "any_online": False,
                "state": CameraState.OFFLINE.value,
                "cameras": [],
            }
        # `state` é sempre o da principal; com a principal fora do ar, o
        # dashboard olha "cameras" e não fica cego às que estão funcionando.
        # Leitura fora da trava: status() é chamado a cada refresh da tela e
        # pegar a trava aqui podia esperar um start()/remove() em andamento.
        # No pior caso devolve a principal de um instante já obsoleto.
        default = self._cameras.get(self._default_id) or runtimes[0]
        states = [r.source.stats.state for r in runtimes]
        # "any_online" responde com um único booleano para as duas telas do
        # dashboard; só uma câmera ONLINE liga o "no ar".
        if any(s is CameraState.ONLINE for s in states):
            any_online = True
        elif any(s in (CameraState.CONNECTING, CameraState.RECONNECTING) for s in states):
            any_online = False
        else:
            any_online = False
        return {
            "any_online": any_online,
            "state": default.source.stats.state.value,
            "default_camera_id": default.camera_id,
            # O spread do status individual vem DEPOIS do id/nome para não
            # deixar o as_dict() da fonte sobrescrever esses campos.
            "cameras": [
                {"camera_id": r.camera_id, "name": r.name, **r.status()} for r in runtimes
            ],
        }


__all__ = ["CameraManager", "CameraRuntime"]
