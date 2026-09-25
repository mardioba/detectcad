"""Ponto de entrada do Contador de Cadeiras.

Modos de execução::

    python -m app.main                      # câmera RTSP do .env (produção)
    python -m app.main --image  teste.jpg   # imagem única
    python -m app.main --video  teste.mp4   # arquivo de vídeo
    python -m app.main --image teste.jpg --no-server   # só imprime, sem HTTP

O ``main`` é deliberadamente pequeno: ele apenas liga as peças
(câmera -> IA -> contagem -> banco -> WebSocket -> dashboard).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


from fastapi import Request  # noqa: E402  - necessário no escopo do módulo por causa de `from __future__ import annotations`

from app.config import mask_url, settings
from app.logging_config import get_logger, setup_logging
from app.schemas import SourceMode

log = get_logger("app")


# --------------------------------------------------------------------- banner
BANNER = """
====================================
   CONTADOR DE CADEIRAS
====================================

Dashboard:  http://localhost:{port}
Logs:       logs/app.log | camera.log | ai.log | training.log
Câmera:     {camera}  [{url}]
Modelo:     {model}
Device:     {device}

====================================
"""


def print_banner(mode: SourceMode, extra: str = "") -> None:
    detector = app_state.detector
    loaded = bool(detector and detector.is_loaded)
    if not loaded:
        state = "NÃO ENCONTRADO - modo de preparação"
    elif not detector.trained_model:
        state = "pré-treinado genérico (não recomendado para produção)"
    else:
        state = "carregado (modelo da empresa)"
    device = detector.device if detector else settings.resolve_device()
    # Mostra o arquivo realmente em uso. Ele pode diferir do YOLO_MODEL do
    # .env: o modelo ativo registrado no banco tem prioridade (troca de
    # modelo em produção sem editar arquivo).
    loaded_name = detector.configured_path.name if loaded else "(nenhum)"
    configured = settings.model_path.name
    if loaded and loaded_name != configured:
        model_line = f"{loaded_name}  (ativo no banco; .env aponta {configured})"
    else:
        model_line = f"{configured}  ({state})"
    print(
        BANNER.format(
            port=settings.web.port,
            camera=settings.camera.camera_name,
            url=mask_url(settings.camera.camera_rtsp_url) or "(não configurada)",
            model=model_line,
            device=device,
        )
    )
    if extra:
        print(extra)
    if not loaded:
        print('AVISO: "Modelo específico ainda não treinado."')
        print("       O dashboard vai funcionar, mas sem contagem até você treinar")
        print("       um modelo (veja README, seção 'Treinar o primeiro modelo').\n")
    elif not detector.trained_model:
        print('AVISO: modelo genérico pré-treinado em uso.')
        print("       Serve para testar a infraestrutura, NÃO para contar cadeiras")
        print("       em produção. Treine com as suas fotos (README, seção 11).\n")
    if mode is not SourceMode.RTSP:
        print(f"MODO: {mode.value} (não é produção)\n")


# ----------------------------------------------------------------- app_state
class AppState:
    """Reúne os componentes de longa duração."""

    def __init__(self) -> None:
        self.detector: Any = None
        self.models: Any = None
        self.cameras: Any = None
        self.inference: Any = None
        self.mode: SourceMode = SourceMode.RTSP
        self.source: str = ""

    def start_workers(self) -> None:
        self.inference.start()

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            if self.inference:
                self.inference.stop()
        with contextlib.suppress(Exception):
            if self.cameras:
                self.cameras.stop_all()
        with contextlib.suppress(Exception):
            if self.detector and self.detector.is_loaded:
                self.detector.unload()


app_state = AppState()


# ------------------------------------------------------------------ bootstrap
def bootstrap(mode: SourceMode, source: str, camera_id: int = 1) -> None:
    """Cria os componentes e aplica a calibração salva."""
    from app.ai.model_manager import ModelManager
    from app.ai.yolo_detector import YoloDetector
    from app.camera.camera_manager import CameraManager
    from app.database.database import init_db, session_scope
    from app.database.repository import CameraRepository, EventRepository
    from app.services.inference_service import InferenceService
    from app.services.state_store import state_store

    # 1. Banco
    init_db()
    with session_scope() as session:
        repo = CameraRepository(session)
        events = EventRepository(session)
        url = source if (mode is SourceMode.RTSP and source) else settings.camera.camera_rtsp_url
        cam = repo.get(camera_id) or repo.get_default()
        if cam is None:
            cam = repo.create(
                name=settings.camera.camera_name,
                rtsp_url=settings.camera.camera_rtsp_url,
                is_default=True,
            )
        events.log("INFO", "startup", f"Aplicação iniciada em modo {mode.value}")

    # 2. Câmera
    cameras = CameraManager()
    cameras.set_event_callback(
        lambda level, event, message: state_store.push_event(level, event, message)
    )
    if mode is SourceMode.RTSP:
        cameras.add(
            camera_id,
            url=settings.camera.camera_rtsp_url,
            name=settings.camera.camera_name,
            mode=SourceMode.RTSP,
        )
    else:
        cameras.add(camera_id, url=source, name=settings.camera.camera_name, mode=mode, start=True)
    app_state.cameras = cameras

    # 3. IA
    detector = YoloDetector()
    app_state.detector = detector
    app_state.models = ModelManager(detector)

    # Modelo ativo registrado no banco tem prioridade sobre o .env
    active = app_state.models.get_active_info()
    model_path = None
    if active.get("active") and active.get("filename"):
        candidate = settings.models_dir / str(active["filename"])
        if candidate.is_file():
            model_path = candidate
    detector.load(model_path)
    if not detector.is_loaded:
        # Tenta o modelo configurado no .env (pode ser genérico, p.ex. yolo11n.pt)
        detector.load()
    app_state.models.sync_to_db()

    # 4. Worker de inferência
    app_state.inference = InferenceService(
        detector=detector,
        model_manager=app_state.models,
        camera_manager=cameras,
        camera_id=camera_id,
        store=state_store,
    )

    # 5. Calibração salva
    from app.services.calibration_service import calibration_service

    calib = calibration_service.load(camera_id)
    app_state.inference.set_calibration(
        {
            "roi": calib.roi.__dict__,
            "roi_enabled": calib.enabled,
            "counting": {
                "chair_height_px": calib.chair_height_px,
                "min_confidence": calib.min_confidence,
                "chair_weight": calib.chair_weight,
                "count_offset": calib.count_offset,
                "counting_method": calib.counting_method,
                "stability_frames": calib.stability_frames,
                "change_min_frames": calib.change_min_frames,
                "stability_min_confidence": calib.stability_min_confidence,
                "pile_gap_px": calib.pile_gap_px,
            },
            "ai": {
                "confidence_threshold": calib.confidence_threshold,
                "chair_classes": calib.chair_classes,
                "pile_classes": calib.pile_classes,
            },
        }
    )
    state_store.set_status(
        camera_id=camera_id,
        mode=mode.value,
        source=source or settings.camera.camera_rtsp_url,
        model_loaded=detector.is_loaded,
        model_trained=detector.trained_model,
        model_filename=detector.configured_path.name,
        process_fps=0.0,
        gpu=detector.gpu_info(),
    )
    app_state.mode = mode
    app_state.source = source


# ------------------------------------------------------------- modo sem HTTP
def run_offline(frame_path: str, video: bool) -> int:
    """Processa um arquivo e imprime o resultado no terminal."""
    from app.services.state_store import state_store

    app_state.start_workers()

    print(f"Processando {'vídeo' if video else 'imagem'}: {frame_path}")
    print("-" * 70)
    deadline = time.time() + (300 if video else 30)
    seen_frames = 0
    last_total = None
    while time.time() < deadline:
        piles = state_store.get_piles()
        total = state_store.total()
        if piles and (total != last_total or seen_frames % 10 == 0):
            last_total = total
            print(f"\n[frame {seen_frames}] TOTAL: {total} cadeiras em {len(piles)} pilha(s)")
            for p in piles:
                print(
                    f"  Pilha {p.pile_id} (track {p.tracking_id}): {p.stable_count} cadeiras "
                    f"| status={p.status.value} | conf={p.confidence * 100:.0f}% "
                    f"| método={p.method} | passo={p.pitch_px:.1f}px "
                    f"| estimadores={ {k: round(v, 1) for k, v in p.candidates.items()} }"
                )
        seen_frames += 1
        if not video and state_store.frame_counter > 0:
            break
        time.sleep(0.4)

    print("-" * 70)
    status = state_store.get_status()
    if status.get("model_error"):
        print(f"Observação: {status['model_error']}")
    jpeg, _ = state_store.get_frame_jpeg(-1)
    out = settings.results_dir / "offline_overlay.jpg"
    if jpeg:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpeg)
        print(f"Frame anotado salvo em: {out}")
    app_state.stop()
    return 0


# ------------------------------------------------------------------ servidor
def build_app() -> Any:
    """Cria a aplicação FastAPI e liga as rotas."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    from app.api import routes as routes_module
    from app.api import websocket as ws_module
    from app.services.dataset_service import DatasetService
    from app.services.state_store import state_store as _store
    from app.services.training_service import training_service

    routes_module.app_state = app_state

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:  # type: ignore[no-untyped-def]
        log.info("Dashboard disponível em http://%s:%s", settings.web.host, settings.web.port)
        # Conecta os eventos do StateStore ao WebSocket
        loop = asyncio.get_running_loop()

        def on_event(kind: str, payload: Any) -> None:
            if kind != "event":
                return
            coro = ws_module.manager.broadcast_event(
                payload.get("level", "INFO"), payload.get("event", ""), payload.get("message", "")
            )
            try:
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                # O loop já foi fechado (servidor encerrando): fecha a corrotina
                # explicitamente para não vazar "never awaited".
                coro.close()

        _store.add_listener(on_event)
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await ws_module.close_all()
            app_state.stop()

    app = FastAPI(
        title="Contador de Cadeiras Plásticas",
        description=(
            "Sistema de visão computacional para contagem de cadeiras em pilhas. "
            "Câmera fixa + YOLO + análise de empilhamento + dashboard em tempo real."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.web.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_module.router)
    app.include_router(ws_module.router)

    static_dir = settings.base_dir / "app" / "static"
    templates_dir = settings.base_dir / "app" / "templates"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False, response_model=None)
    async def dashboard(request: Request) -> Any:  # type: ignore[no-untyped-def]
        return templates.TemplateResponse(request, "dashboard.html", {"port": settings.web.port})

    @app.get("/favicon.ico", include_in_schema=False, response_model=None)
    async def favicon() -> Any:  # type: ignore[no-untyped-def]
        from fastapi.responses import Response

        return Response(status_code=204)

    return app


# ---------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Contador de cadeiras plásticas por câmera IP/RTSP",
    )
    parser.add_argument("--image", help="Processa uma imagem JPG/PNG e encerra (modo teste)")
    parser.add_argument("--video", help="Processa um arquivo de vídeo e encerra (modo teste)")
    parser.add_argument(
        "--no-server", action="store_true", help="Não sobe o servidor web (imprime no terminal)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Sobe o dashboard mesmo com --image/--video (demonstração e teste)",
    )
    parser.add_argument("--host", help="Sobrescreve HOST")
    parser.add_argument("--port", type=int, help="Sobrescreve PORT")
    parser.add_argument("--fps", type=float, help="Sobrescreve PROCESS_FPS")
    parser.add_argument("--model", help="Sobrescreve YOLO_MODEL")
    parser.add_argument("--device", help="Sobrescreve DEVICE (auto|cpu|cuda|cuda:0)")
    parser.add_argument("--reload", action="store_true", help="Recarrega ao alterar código (dev)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings.ensure_dirs()
    setup_logging(settings.logs_dir, settings.web.log_level, console=True)

    if args.fps:
        settings.camera.process_fps = args.fps
    if args.model:
        settings.ai.yolo_model = args.model
    if args.device:
        settings.ai.device = args.device
    if args.host:
        settings.web.host = args.host
    if args.port:
        settings.web.port = args.port

    if args.image and args.video:
        print("Use --image OU --video, não os dois.")
        return 2

    if args.image and not args.serve:
        bootstrap(SourceMode.IMAGE, args.image)
        print_banner(SourceMode.IMAGE, extra=f"Arquivo: {args.image}\n")
        return run_offline(args.image, video=False)
    if args.video and not args.serve:
        bootstrap(SourceMode.VIDEO, args.video)
        print_banner(SourceMode.VIDEO, extra=f"Arquivo: {args.video}\n")
        return run_offline(args.video, video=True)

    if args.image:
        bootstrap(SourceMode.IMAGE, args.image)
        print_banner(SourceMode.IMAGE, extra=f"Arquivo: {args.image}\n")
    elif args.video:
        bootstrap(SourceMode.VIDEO, args.video)
        print_banner(SourceMode.VIDEO, extra=f"Arquivo: {args.video}\n")
    else:
        bootstrap(SourceMode.RTSP, settings.camera.camera_rtsp_url)
        print_banner(SourceMode.RTSP)

    if args.no_server:
        app_state.start_workers()
        print("Rodando sem servidor web (Ctrl+C para sair).")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        app_state.stop()
        return 0

    import uvicorn

    app = build_app()
    app_state.start_workers()

    def handle_signal(_sig: int, _frame: Any) -> None:  # pragma: no cover
        log.info("Encerrando...")
        app_state.stop()

    with contextlib.suppress(NotImplementedError, ValueError):
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    uvicorn.run(
        app,
        host=settings.web.host,
        port=settings.web.port,
        log_level=settings.web.log_level.lower(),
        access_log=False,
    )
    return 0


app = None  # criado sob demanda por build_app() / servidor ASGI

if __name__ == "__main__":
    sys.exit(main())
