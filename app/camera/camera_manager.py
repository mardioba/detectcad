"""Gerenciador de câmeras.

Suporta N câmeras desde o início (multicâmera), mesmo com o caso de uso
principal sendo uma única câmera. Cada câmera tem seu próprio
:class:`FrameSource` e seu próprio worker de inferência.
"""

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
        self._last_frame: tuple[Any, FrameInfo] | None = None
        self._lock = threading.Lock()

    def take_latest(self) -> tuple[Any, FrameInfo] | None:
        """Consome o próximo frame real da fonte (não bloqueia a UI)."""
        item = self.source.read(timeout=0.05)
        if item is not None:
            with self._lock:
                self._last_frame = item
        return item

    def last_frame(self) -> tuple[Any, FrameInfo] | None:
        with self._lock:
            return self._last_frame

    def status(self) -> dict[str, Any]:
        info = self.last_frame()
        data = self.source.stats.as_dict()
        if info is not None:
            data["width"] = info[0].shape[1]
            data["height"] = info[0].shape[0]
        return data


class CameraManager:
    """Registro de câmeras ativas.

    ``start_default()`` sobe a câmera definida no ``.env``. As demais podem ser
    adicionadas depois pela API (dashboard) ou por código.
    """

    def __init__(self) -> None:
        self._cameras: dict[int, CameraRuntime] = {}
        self._default_id: int = 1
        self._lock = threading.RLock()
        self._event_callback: Callable[[str, str, str], None] | None = None

    # ------------------------------------------------------------------ setup
    def set_event_callback(self, cb: Callable[[str, str, str], None]) -> None:
        self._event_callback = cb
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
        with self._lock:
            if camera_id in self._cameras:
                return self._cameras[camera_id]
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
            source.set_video_loop(loop_video)
            if self._event_callback is not None:
                source.set_event_callback(self._event_callback)
            runtime = CameraRuntime(source=source, camera_id=camera_id, name=name or settings.camera.camera_name)
            self._cameras[camera_id] = runtime
            if start:
                runtime.source.start()
            return runtime

    def get(self, camera_id: int) -> CameraRuntime | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def default_id(self) -> int:
        with self._lock:
            return self._default_id

    def set_default(self, camera_id: int) -> None:
        with self._lock:
            if camera_id in self._cameras:
                self._default_id = camera_id

    def list(self) -> list[CameraRuntime]:
        with self._lock:
            return list(self._cameras.values())

    def remove(self, camera_id: int) -> bool:
        with self._lock:
            runtime = self._cameras.pop(camera_id, None)
        if runtime is None:
            return False
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
                    if not runtime.source.stats.opened:
                        runtime.source.start()
                    return runtime
            return self.add(
                self._default_id,
                url=settings.camera.camera_rtsp_url,
                name=settings.camera.camera_name,
                mode=SourceMode.RTSP,
            )

    def stop_all(self) -> None:
        with self._lock:
            runtimes = list(self._cameras.values())
            self._cameras.clear()
        for runtime in runtimes:
            runtime.source.stop()

    # ----------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        runtimes = self.list()
        if not runtimes:
            return {
                "any_online": False,
                "state": CameraState.OFFLINE.value,
                "cameras": [],
            }
        default = self._cameras.get(self._default_id) or runtimes[0]
        states = [r.source.stats.state for r in runtimes]
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
            "cameras": [
                {"camera_id": r.camera_id, "name": r.name, **r.status()} for r in runtimes
            ],
        }


__all__ = ["CameraManager", "CameraRuntime"]
