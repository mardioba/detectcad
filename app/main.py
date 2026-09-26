"""Ponto de entrada do Contador de Cadeiras.

Modos de execução::

    python -m app.main                      # câmera RTSP do .env (produção)
    python -m app.main --image  teste.jpg   # imagem única
    python -m app.main --video  teste.mp4   # arquivo de vídeo
    python -m app.main --image teste.jpg --no-server   # só imprime, sem HTTP

O ``main`` é deliberadamente pequeno: ele apenas liga as peças
(câmera -> IA -> contagem -> banco -> WebSocket -> dashboard).
"""

# =============================================================================
# ARQUIVO / MAPA  -  app/main.py
#
# O que faz: ponto de entrada. Monta a ordem dos componentes e decide o modo
#   de execução. Nenhuma lógica de detecção mora aqui.
#
# Ordem de leitura:
#   1. BANNER / print_banner ... o que está rodando, com senha mascarada
#   2. AppState .............. guarda os componentes de vida longa (para o stop)
#   3. bootstrap ............. 1) banco 2) câmera 3) IA 4) worker 5) calibração
#   4. run_offline ........... --image/--video: imprime no terminal, sem HTTP
#   5. build_app ............. FastAPI, rotas, WebSocket, dashboard
#   6. parse_args / main ..... CLI, precedência das flags, sinais e uvicorn
#
# O banner existe porque o modo mais confuso de falha é "subiu, mas sem modelo":
# avisa logo no terminal, antes de qualquer requisição.
# =============================================================================

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
# Placeholders {port} {camera} {url} {model} {device} são preenchidos em
# print_banner. Ver o significado de cada linha logo abaixo de lá.
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
    # Os 3 estados possíveis do modelo, do pior para o melhor:
    #   1. não carregou       -> "modo de preparação": a infra sobe, mas a
    #                            contagem NÃO funciona. É o estado esperado
    #                            logo após instalar, antes do primeiro treino.
    #   2. carregou genérico   -> yolo11n.pt e afins. Serve para testar a
    #                            máquina, não para contar pilhas: o COCO não
    #                            foi treinado para cadeira empilhada.
    #   3. carregou da empresa-> best.pt treinado nas fotos da câmera real.
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
        # Divergência banco x .env: mostra as duas para ninguém ficar achando
        # que editou o arquivo errado.
        model_line = f"{loaded_name}  (ativo no banco; .env aponta {configured})"
    else:
        model_line = f"{configured}  ({state})"
    print(
        BANNER.format(
            port=settings.web.port,
            camera=settings.camera.camera_name,
            # mask_url: a senha da câmera nunca é impressa em claro, nem no
            # terminal, nem na tela de televisionamento.
            url=mask_url(settings.camera.camera_rtsp_url) or "(não configurada)",
            model=model_line,
            device=device,
        )
    )
    if extra:
        print(extra)
    # O aviso é repetido como texto, e não só no estado entre parênteses,
    # porque é a informação que o operador precisa ver primeiro.
    if not loaded:
        print('AVISO: "Modelo específico ainda não treinado."')
        print("       O dashboard vai funcionar, mas sem contagem até você treinar")
        print("       um modelo (veja README, seção 'Treinar o primeiro modelo').\n")
    elif not detector.trained_model:
        print('AVISO: modelo genérico pré-treinado em uso.')
        print("       Serve para testar a infraestrutura, NÃO para contar cadeiras")
        print("       em produção. Treine com as suas fotos (README, seção 11).\n")
    if mode is not SourceMode.RTSP:
        # Fora do modo RTSP não há produção: o sistema entra no ar sem
        # monitoramento e a contagem não representa o depósito real.
        print(f"MODO: {mode.value} (não é produção)\n")


# ----------------------------------------------------------------- app_state
class AppState:
    """Reúne os componentes de longa duração.

    Existe como objeto único e global (app_state) porque quem precisa parar
    tudo não está com acesso às variáveis: a rota de shutdown, o handler de
    SIGTERM e o modo offline estão em funções diferentes.
    """

    def __init__(self) -> None:
        # Tudo começa None: quem chama start/stop antes do bootstrap não
        # quebra (ver o contextlib.suppress em stop).
        self.detector: Any = None
        self.models: Any = None
        self.cameras: Any = None
        self.inference: Any = None
        self.mode: SourceMode = SourceMode.RTSP
        self.source: str = ""

    def start_workers(self) -> None:
        # O worker de inferência roda em thread separada e é quem faz o
        # trabalho pesado (câmera -> YOLO -> contagem) sem travar o FastAPI.
        self.inference.start()

    def stop(self) -> None:
        # Ordem inversa ao start: worker, câmeras, modelo. Cada bloco é
        # isolado para que uma falha no primeiro não impeça os outros de
        # liberar recurso (GPU, socket, handle de arquivo).
        with contextlib.suppress(Exception):
            if self.inference:
                self.inference.stop()
        with contextlib.suppress(Exception):
            if self.cameras:
                self.cameras.stop_all()
        with contextlib.suppress(Exception):
            if self.detector and self.detector.is_loaded:
                # unload() devolve a VRAM; sem isso o próximo start do
                # container tropeça em "CUDA out of memory".
                self.detector.unload()


app_state = AppState()


# ------------------------------------------------------------------ bootstrap
def bootstrap(mode: SourceMode, source: str, camera_id: int = 1) -> None:
    """Cria os componentes e aplica a calibração salva.

    Tudo é montado aqui, sempre na mesma ordem (banco -> câmera -> IA ->
    worker -> calibração). A ordem importa: o banco precisa existir antes de
    consultar o modelo ativo, e a calibração só pode ser aplicada depois que
    o InferenceService existe.
    """
    # Imports aqui dentro, e não no topo: quem importa app.main (os testes,
    # o uvicorn) não paga o custo de torch/ultralytics. Também garante que
    # os componentes só existam depois que settings foi criado.
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
        # O source explícito (--rtsp) ganha do .env; senão usa o .env.
        url = source if (mode is SourceMode.RTSP and source) else settings.camera.camera_rtsp_url
        cam = repo.get(camera_id) or repo.get_default()
        if cam is None:
            # Primeira execução: a câmera do .env vira a câmera padrão.
            cam = repo.create(
                name=settings.camera.camera_name,
                rtsp_url=settings.camera.camera_rtsp_url,
                is_default=True,
            )
        events.log("INFO", "startup", f"Aplicação iniciada em modo {mode.value}")

    # 2. Câmera
    cameras = CameraManager()
    # Eventos da câmera entram no StateStore e daí no WebSocket: é assim que
    # "câmera reconectou" aparece no dashboard sem polling.
    cameras.set_event_callback(
        lambda level, event, message: state_store.push_event(level, event, message)
    )
    if mode is SourceMode.RTSP:
        # start=False implícito: a câmera entra depois, junto com o worker.
        cameras.add(
            camera_id,
            url=settings.camera.camera_rtsp_url,
            name=settings.camera.camera_name,
            mode=SourceMode.RTSP,
        )
    else:
        # --image/--video já abre o arquivo aqui (start=True): é um arquivo
        # local, não há reconexão a fazer.
        cameras.add(camera_id, url=source, name=settings.camera.camera_name, mode=mode, start=True)
    app_state.cameras = cameras

    # 3. IA
    detector = YoloDetector()
    app_state.detector = detector
    app_state.models = ModelManager(detector)

    # Modelo ativo registrado no banco tem prioridade sobre o .env
    active = app_state.models.get_active_info()
    model_path = None
    # Regra de precedência do modelo:
    #   1. arquivo em data/models/<nome> marcado como ativo no banco
    #   2. YOLO_MODEL do .env
    #   3. nada -> "modo de preparação", sem contagem
    # O .is_file() protege do caso do banco apontar para um arquivo apagado:
    # aí cai no .env em vez de subir com o modelo quebrado.
    if active.get("active") and active.get("filename"):
        candidate = settings.models_dir / str(active["filename"])
        if candidate.is_file():
            model_path = candidate
    detector.load(model_path)
    if not detector.is_loaded:
        # Tenta o modelo configurado no .env (pode ser genérico, p.ex. yolo11n.pt)
        detector.load()
    # Persiste o que está no disco (e a versão) para o banco não divergir.
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
    # Vem do BANCO, não do .env: foi o operador que calibrou aquela câmera
    # específica (altura da cadeira em pixels só existe para ela).
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
    # Status inicial do dashboard. process_fps começa em 0.0: o valor real
    # só faz sentido depois que o primeiro frame passar pelo worker.
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
    """Processa um arquivo e imprime o resultado no terminal.

    Modo de diagnóstico: existe para responder "por que o modelo não conta
    cadeiras?" sem subir servidor, sem WebSocket e sem tocar no dashboard.
    Usa o mesmo pipeline da produção, o que garante que o resultado visto
    aqui é o mesmo que o dashboard mostraria.
    """
    from app.services.state_store import state_store

    app_state.start_workers()

    print(f"Processando {'vídeo' if video else 'imagem'}: {frame_path}")
    print("-" * 70)
    # Deadline como rede de segurança: vídeo longo ou arquivo que nunca
    # termina não podem deixar o processo pendurado para sempre.
    deadline = time.time() + (300 if video else 30)
    seen_frames = 0
    last_total = None
    while time.time() < deadline:
        piles = state_store.get_piles()
        total = state_store.total()
        # Imprime quando o total MUDA, e a cada 10 passos para mostrar a
        # evolução. Sem isso, um vídeo de 2 horas despejaria 1000 linhas.
        if piles and (total != last_total or seen_frames % 10 == 0):
            last_total = total
            print(f"\n[frame {seen_frames}] TOTAL: {total} cadeiras em {len(piles)} pilha(s)")
            for p in piles:
                # O detalhe importa no diagnóstico: status e candidatos dizem
                # QUAL método errou e por quê, não só o número final.
                print(
                    f"  Pilha {p.pile_id} (track {p.tracking_id}): {p.stable_count} cadeiras "
                    f"| status={p.status.value} | conf={p.confidence * 100:.0f}% "
                    f"| método={p.method} | passo={p.pitch_px:.1f}px "
                    f"| estimadores={ {k: round(v, 1) for k, v in p.candidates.items()} }"
                )
        seen_frames += 1
        # Imagem é um único frame: assim que ele passa, acabou.
        if not video and state_store.frame_counter > 0:
            break
        time.sleep(0.4)

    print("-" * 70)
    status = state_store.get_status()
    # model_error explica o motivo real de não ter contado (modelo ausente,
    # classe desconhecida, torch faltando). Sem ele, o silêncio parece bug.
    if status.get("model_error"):
        print(f"Observação: {status['model_error']}")
    jpeg, _ = state_store.get_frame_jpeg(-1)
    out = settings.results_dir / "offline_overlay.jpg"
    if jpeg:
        # A imagem anotada é a prova: dá para conferir a contagem com os olhos.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpeg)
        print(f"Frame anotado salvo em: {out}")
    app_state.stop()
    return 0


# ------------------------------------------------------------------ servidor
def build_app() -> Any:
    """Cria a aplicação FastAPI e liga as rotas."""
    # Imports locais: o dashboard (app/main.py em modo --no-server) nunca
    # precisa de FastAPI, templates nem dos serviços.
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

    # Injeção do app_state global nas rotas: evita import circular (routes
    # precisa do estado, main precisa das rotas).
    routes_module.app_state = app_state

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:  # type: ignore[no-untyped-def]
        # lifespan = startup/shutdown do FastAPI. O log de "dashboard
        # disponível" vai aqui (e não antes) porque a porta só está de fato
        # escutando depois que o servidor sobe.
        log.info("Dashboard disponível em http://%s:%s", settings.web.host, settings.web.port)
        # Conecta os eventos do StateStore ao WebSocket
        # O loop é capturado AGORA porque os eventos chegam de outra thread
        # (worker de inferência) e precisam voltar para cá para virar broadcast.
        loop = asyncio.get_running_loop()

        def on_event(kind: str, payload: Any) -> None:
            if kind != "event":
                return
            coro = ws_module.manager.broadcast_event(
                payload.get("level", "INFO"), payload.get("event", ""), payload.get("message", "")
            )
            try:
                # run_coroutine_threadsafe é a ponte thread -> event loop.
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                # O loop já foi fechado (servidor encerrando): fecha a corrotina
                # explicitamente para não vazar "never awaited".
                coro.close()

        _store.add_listener(on_event)
        try:
            yield
        finally:
            # Roda também no shutdown do uvicorn, e o app_state.stop() de novo
            # é inofensivo (cada bloco é suprimido).
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
        # origins vazio no .env cai em "*": o dashboard é servido pela mesma
        # origem, então a configuração permissiva não muda nada na prática.
        allow_origins=origins or ["*"],
        # Necessário porque o dashboard manda cookies/credenciais; com "*" o
        # navegador recusaria, por isso a lista de origens deve ser explícita
        # se um dia houver autenticação.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_module.router)
    app.include_router(ws_module.router)

    static_dir = settings.base_dir / "app" / "static"
    templates_dir = settings.base_dir / "app" / "templates"
    # is_dir() antes de mount: sem static, o servidor ainda sobe (a API
    # funciona; só o visual do dashboard quebra).
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))

    # Rotas do próprio dashboard. include_in_schema=False esconde do /docs.
    @app.get("/", response_class=HTMLResponse, include_in_schema=False, response_model=None)
    async def dashboard(request: Request) -> Any:  # type: ignore[no-untyped-def]
        # {port} é entregue ao template para o JS montar a URL do WebSocket.
        return templates.TemplateResponse(request, "dashboard.html", {"port": settings.web.port})

    # 204 em vez de 404: evita o "erro de favicon" no log a cada F5.
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
    # As flags de host/port/fps/model/device são SOBRESCRITAS em settings mais
    # abaixo, DEPOIS de o .env ser lido. Por isso não podem ter valor padrão
    # aqui: None é o que distingue "não passei a flag" de "passei o valor X".
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # ensure_dirs() antes do setup_logging: o setup precisa do diretório.
    settings.ensure_dirs()
    setup_logging(settings.logs_dir, settings.web.log_level, console=True)

    # Precedência: flag de linha de comando > .env > default do config.py.
    # Só sobrescreve quando a flag foi passada de fato (None = não passada).
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
        # São modos mutuamente exclusivos: um worker, uma fonte só.
        print("Use --image OU --video, não os dois.")
        return 2

    # Modo offline: arquivo + sem --serve. Imprime no terminal e sai, sem HTTP.
    if args.image and not args.serve:
        bootstrap(SourceMode.IMAGE, args.image)
        print_banner(SourceMode.IMAGE, extra=f"Arquivo: {args.image}\n")
        return run_offline(args.image, video=False)
    if args.video and not args.serve:
        bootstrap(SourceMode.VIDEO, args.video)
        print_banner(SourceMode.VIDEO, extra=f"Arquivo: {args.video}\n")
        return run_offline(args.video, video=True)

    # Daqui em diante sempre sobe servidor. --image/--video com --serve é
    # demonstração: processa o arquivo e mostra o dashboard ao mesmo tempo.
    if args.image:
        bootstrap(SourceMode.IMAGE, args.image)
        print_banner(SourceMode.IMAGE, extra=f"Arquivo: {args.image}\n")
    elif args.video:
        bootstrap(SourceMode.VIDEO, args.video)
        print_banner(SourceMode.VIDEO, extra=f"Arquivo: {args.video}\n")
    else:
        # Produção: a fonte é a URL do .env.
        bootstrap(SourceMode.RTSP, settings.camera.camera_rtsp_url)
        print_banner(SourceMode.RTSP)

    if args.no_server:
        # Só câmera e IA, sem HTTP. Mesmo worker do modo com servidor, para
        # não existir um caminho de código diferente do testado.
        app_state.start_workers()
        print("Rodando sem servidor web (Ctrl+C para sair).")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        app_state.stop()
        return 0

    # uvicorn é o último import: é pesado e desnecessário nos modos acima.
    import uvicorn

    app = build_app()
    # Worker iniciado ANTES do uvicorn: se a primeira leitura do dashboard já
    # pedir o frame, ele existe.
    app_state.start_workers()

    def handle_signal(_sig: int, _frame: Any) -> None:  # pragma: no cover
        # SIGTERM vem do `docker stop`/systemd. Sem este handler o processo
        # morre sem soltar a câmera e a VRAM, e o próximo start falha.
        log.info("Encerrando...")
        app_state.stop()

    # signal.signal só funciona na thread principal; em contexto de teste pode
    # não estar disponível, daí o suppress (não esconde falha de código).
    with contextlib.suppress(NotImplementedError, ValueError):
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    uvicorn.run(
        app,
        host=settings.web.host,
        port=settings.web.port,
        log_level=settings.web.log_level.lower(),
        # access_log desligado: uma linha por requisição (o dashboard pede
        # imagem a STREAM_FPS) lotaria o app.log e esconderia o que importa.
        # As rotas relevantes já registram o que precisa.
        access_log=False,
    )
    return 0


# Placeholder para o servidor ASGI (uvicorn app.main:app). Vale None no import
# normal de propósito: construir a app aqui carregaria torch/ultralytics.
app = None  # criado sob demanda por build_app() / servidor ASGI

if __name__ == "__main__":
    sys.exit(main())
