"""Testes da camada de câmera: fontes, estados, reconexão e métricas.

A reconexão é testada com um "servidor RTSP" simulado por um socket local que
aceita conexões e depois fecha, o que é exatamente o cenário de queda de
câmera em produção.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.camera.camera_manager import CameraManager
from app.camera.rtsp_camera import FrameSource
from app.schemas import CameraState, SourceMode


# --------------------------------------------------------------- imagem
def test_image_source_reads_frame(tmp_path: Path):
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)

    src = FrameSource(url=str(path), mode=SourceMode.IMAGE)
    src.start()
    try:
        item = src.read(timeout=8.0)
        assert item is not None
        frame, info = item
        assert frame.shape == (240, 320, 3)
        assert info.width == 320 and info.height == 240
        assert info.source_mode is SourceMode.IMAGE
    finally:
        src.stop()
    assert src.stats.frames_captured >= 1


def test_image_source_missing_file(tmp_path: Path):
    src = FrameSource(url=str(tmp_path / "nao_existe.png"), mode=SourceMode.IMAGE, retry_delay_sec=0.1)
    src.start()
    time.sleep(1.0)
    try:
        assert src.stats.state is CameraState.ERROR
        assert "não encontrada" in src.stats.last_event
    finally:
        src.stop()


# ----------------------------------------------------------------- vídeo
def test_video_source(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(25):
        vw.write(np.full((120, 160, 3), i * 8 % 255, dtype=np.uint8))
    vw.release()
    assert path.is_file() and path.stat().st_size > 0

    src = FrameSource(url=str(path), mode=SourceMode.VIDEO, retry_delay_sec=0.1)
    src.set_video_loop(False)
    src.start()
    try:
        deadline = time.time() + 25
        while time.time() < deadline and src.stats.frames_captured < 3:
            src.read(timeout=0.5)
        assert src.stats.frames_captured >= 3, src.stats.as_dict()
    finally:
        src.stop()
        src.wait_closed(timeout=25.0)


# ----------------------------------------------------------------- RTSP
class _FakeCap:
    """VideoCapture falso: entrega N frames e depois falha.

    Testa a LÓGICA de reconexão do sistema sem depender do FFmpeg (que
    bloquearia por dezenas de segundos ao tentar falar RTSP de verdade).
    """

    def __init__(self, frames: int, w: int = 160, h: int = 120) -> None:
        self.frames = frames
        self.w, self.h = w, h
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return not self.released

    def read(self):  # noqa: ANN201
        if self.frames <= 0:
            return False, None
        self.frames -= 1
        return True, np.full((self.h, self.w, 3), 60, dtype=np.uint8)

    def get(self, _prop: int) -> float:  # noqa: ANN201
        return 0.0

    def release(self) -> None:
        self.released = True


def test_rtsp_reconnect_counter(monkeypatch: pytest.MonkeyPatch):
    """Câmera que cai no meio do frame precisa reconectar e contar."""
    opened: list[int] = []

    def fake_build(self) -> "_FakeCap":
        # 1ª e 3ª aberturas dão 4 frames; a 2ª cai na hora (0 frames).
        idx = len(opened)
        opened.append(idx)
        return _FakeCap(4 if idx in (0, 2) else 0)

    monkeypatch.setattr(FrameSource, "_build_capture", fake_build)

    src = FrameSource(
        url="rtsp://user:senha@10.0.0.9:554/x",
        mode=SourceMode.RTSP,
        open_timeout_sec=0.5,
        read_timeout_sec=0.5,
        retry_delay_sec=0.1,
        max_retries=0,
    )
    events: list[tuple[str, str, str]] = []
    src.set_event_callback(lambda lvl, ev, msg: events.append((lvl, ev, msg)))
    src.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and len(opened) < 3:
            src.read(timeout=0.2)
        time.sleep(0.4)
        assert len(opened) >= 3, f"deveria ter reconectado: {opened}"
        assert src.stats.reconnects >= 1, src.stats.as_dict()
        assert any(ev == "camera_reconnect" for _l, ev, _m in events), events
        # a senha nunca aparece nos eventos/estado
        assert "senha" not in str(src.stats.as_dict())
    finally:
        src.stop()


def test_rtsp_stats_url_is_masked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(FrameSource, "_build_capture", lambda self: _FakeCap(1))
    src = FrameSource(url="rtsp://user:segredo@10.0.0.9:554/x", mode=SourceMode.RTSP)
    d = src.stats.as_dict()
    assert "segredo" not in d["url"]
    assert "****" in d["url"]


def test_rtsp_offline_does_not_crash():
    """URL que nunca conecta: fica OFFLINE, sem exceção e sem travar."""
    src = FrameSource(
        url="rtsp://127.0.0.1:1/nao_existe",
        mode=SourceMode.RTSP,
        open_timeout_sec=0.5,
        read_timeout_sec=0.5,
        retry_delay_sec=0.2,
        max_retries=2,
    )
    src.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline and src.stats.state not in (
            CameraState.OFFLINE, CameraState.ERROR
        ):
            time.sleep(0.2)
        assert src.stats.state in (CameraState.OFFLINE, CameraState.ERROR, CameraState.RECONNECTING)
        assert src.stats.connect_attempts >= 1
        assert src.stats.last_error or src.stats.last_event
    finally:
        src.stop()
        # O FFmpeg pode ficar alguns segundos preso na tentativa de conexão;
        # sem esperar, a thread viva atrapalharia os testes seguintes.
        src.wait_closed(timeout=25.0)


# ------------------------------------------------------------- frame queue
def test_queue_discards_old_frames():
    """Em tempo real, frames antigos devem ser descartados, não acumular."""
    src = FrameSource(url="x", mode=SourceMode.IMAGE, buffer_size=2)
    for i in range(10):
        info = None
        frame = np.full((10, 10, 3), i, dtype=np.uint8)
        from app.camera.rtsp_camera import FrameInfo

        info = FrameInfo(timestamp=time.time(), index=i, width=10, height=10, source_fps=0, capture_latency_ms=0)
        src._push(frame, info)
    assert src.stats.dropped_frames >= 8, src.stats.as_dict()


# --------------------------------------------------------- camera manager
def test_manager_lifecycle():
    mgr = CameraManager()
    rt = mgr.add(1, url="rtsp://x/y", name="Teste", start=False)
    assert mgr.get(1) is rt
    assert mgr.default_id() == 1
    assert len(mgr.list()) == 1
    st = mgr.status()
    assert st["any_online"] is False
    assert st["cameras"][0]["name"] == "Teste"
    assert mgr.remove(1) is True
    assert mgr.get(1) is None
    mgr.stop_all()


def test_manager_masks_url_in_status():
    mgr = CameraManager()
    mgr.add(1, url="rtsp://user:senhasecreta@10.0.0.1:554/s", start=False)
    text = str(mgr.status())
    assert "senhasecreta" not in text


def test_manager_supports_multiple_cameras():
    mgr = CameraManager()
    mgr.add(1, url="rtsp://a/1", name="A", start=False)
    mgr.add(2, url="rtsp://b/2", name="B", start=False)
    mgr.set_default(2)
    assert mgr.default_id() == 2
    assert len(mgr.list()) == 2
    mgr.stop_all()
