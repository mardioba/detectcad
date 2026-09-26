r"""Captura de frames: RTSP, arquivo de vídeo ou imagem estática.

Uma única classe (:class:`FrameSource`) cobre os três modos, para que o
pipeline de IA seja idêntico em produção e em teste.

Estados e reconexão automática:

    OFFLINE -> CONNECTING -> ONLINE
                        \-> RECONNECTING -> ONLINE
                                    \-> ERROR
"""

# ARQUIVO / MAPA - rtsp_camera.py
# O que faz: captura frames de três fontes (RTSP, arquivo de vídeo, imagem
#   fixa) e entrega por uma fila, com reconexão automática.
# Ordem de leitura:
#   1. FrameInfo / CameraStats  -> o que o dashboard mostra
#   2. FrameSource.__init__     -> fila, thread, trava, timeouts
#   3. start / stop / read      -> API usada pelo worker de IA
#   4. _build_capture           -> como abrir cada fonte (com timeout)
#   5. _run                     -> estados e política de reconexão
#   6. _read_loop               -> leitura, descarte e travamento
# Detalhe que engana: em SourceMode.IMAGE o None devolvido por
# _build_capture significa SUCESSO, porque imagem não usa VideoCapture.

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas import CameraState, SourceMode

log = logging.getLogger("camera")

# Limites de rede do FFmpeg para RTSP, aplicados uma única vez.
#
# Sem isso, ``cv2.VideoCapture(rtsp://ip-inacessivel)`` fica bloqueado na
# espera padrão do FFmpeg (dezenas de segundos) e, como o FFmpeg do OpenCV é
# praticamente serializado, uma câmera morta atrasa até a leitura de um
# arquivo local. ``timeout`` é em microssegundos.
# O operador ainda pode sobrepor pelo .env sem mexer no código.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|timeout;5000000|max_delay;5000000",
)


@dataclass
class FrameInfo:
    """Metadados do frame capturado (vistos pelo dashboard)."""

    timestamp: float
    index: int
    width: int
    height: int
    source_fps: float
    capture_latency_ms: float
    source_mode: SourceMode = SourceMode.RTSP

    def as_dict(self) -> dict[str, Any]:
        """Serializa o frame para o dashboard, com números arredondados."""
        return {
            "timestamp": self.timestamp,
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "source_fps": round(self.source_fps, 2),
            "capture_latency_ms": round(self.capture_latency_ms, 2),
            "source_mode": self.source_mode.value,
        }


@dataclass
class CameraStats:
    """Estatísticas de operação da câmera."""

    state: CameraState = CameraState.OFFLINE
    frames_captured: int = 0
    frames_delivered: int = 0
    dropped_frames: int = 0
    invalid_frames: int = 0
    reconnects: int = 0
    connect_attempts: int = 0
    last_error: str = ""
    connected_at: float | None = None
    last_frame_at: float | None = None
    source_fps: float = 0.0
    width: int = 0
    height: int = 0
    opened: bool = False
    url: str = ""
    last_event: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def uptime_sec(self) -> float:
        """Tempo online atual. Vale 0 se a fonte não está conectada."""
        if self.connected_at is None or self.state is not CameraState.ONLINE:
            return 0.0
        return max(0.0, time.time() - self.connected_at)

    def as_dict(self) -> dict[str, Any]:
        """Estado da câmera para o dashboard, sem vazar a senha da URL."""
        from app.config import mask_url

        return {
            "state": self.state.value,
            "frames_captured": self.frames_captured,
            "frames_delivered": self.frames_delivered,
            "dropped_frames": self.dropped_frames,
            "invalid_frames": self.invalid_frames,
            "reconnects": self.reconnects,
            "connect_attempts": self.connect_attempts,
            "uptime_sec": round(self.uptime_sec, 1),
            "last_error": self.last_error,
            "source_fps": round(self.source_fps, 2),
            "width": self.width,
            "height": self.height,
            "opened": self.opened,
            # Mascarado: a URL nunca sai daqui com a senha.
            "url": mask_url(self.url),
            "last_event": self.last_event,
        }


class FrameSource:
    """Fonte de frames com reconexão automática.

    Não bloqueia: a thread do chamador recebe frames por uma
    :class:`queue.Queue` de tamanho limitado (descarta os frames antigos, o
    que é o comportamento correto para visão em tempo real).
    """

    # Parâmetros com padrão pensado para câmera IP real:
    #   open_timeout_sec  8 s  - câmera morta não pode travar o sistema
    #   read_timeout_sec  10 s - câmera que para de entregar frame
    #   retry_delay_sec   3 s  - espera entre tentativas (evita martelar a rede)
    #   max_retries       0    - 0 = infinito; em produção a câmera não desiste
    #   buffer_size       2    - 1 frame em processamento + 1 aguardando
    def __init__(
        self,
        url: str = "",
        *,
        name: str = "Camera 1",
        mode: SourceMode = SourceMode.RTSP,
        open_timeout_sec: float = 8.0,
        read_timeout_sec: float = 10.0,
        retry_delay_sec: float = 3.0,
        max_retries: int = 0,
        transport: str = "tcp",
        buffer_size: int = 2,
    ) -> None:
        self.url = url
        self.name = name
        self.mode = mode
        self.open_timeout_sec = open_timeout_sec
        self.read_timeout_sec = read_timeout_sec
        self.retry_delay_sec = retry_delay_sec
        self.max_retries = max_retries
        # tcp em vez de udp: em rede industrial com Wi-Fi/cabo ruim, UDP
        # perde pacotes e o FFmpeg passa a esperar quadro que nunca chega.
        self.transport = transport
        self.stats = CameraStats(url=url)
        self.stats.state = CameraState.OFFLINE

        # Fila pequena e de tamanho FIXO. Em tempo real frame velho não serve
        # para nada: a inferência deve ver o quadro mais recente possível.
        self._queue: queue.Queue[tuple[np.ndarray, FrameInfo] | None] = queue.Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None
        # Event e não um booleano: wait() substitui sleep e devolve na hora
        # quando stop() é chamado, encurtando o encerramento.
        self._stop = threading.Event()
        # RLock e não Lock porque _emit() roda dentro da região crítica de
        # start(); se o callback do evento voltar a chamar a fonte, um Lock
        # comum travaria a própria thread.
        self._lock = threading.RLock()
        self._cap: cv2.VideoCapture | None = None
        self._last_ok_ts: float | None = None
        self._last_pushed: float = 0.0
        self._event_callback: Any = None
        self._image_repeat: bool = True
        # `stats` é escrito pela thread de captura e lido pelas threads HTTP
        # sem trava. São contadores simples e o dashboard tolera um passo de
        # defasagem; travar aqui custaria latência em todo request.

    # ------------------------------------------------------------------ utils
    def set_event_callback(self, cb: Any) -> None:
        """Registra callback ``cb(level, event, message)`` para log de eventos."""
        self._event_callback = cb

    def _emit(self, level: str, event: str, message: str) -> None:
        """Registra um evento: log, `last_event` e callback do dashboard."""
        self.stats.last_event = f"{event}: {message}"
        # level chega como "INFO"/"WARNING"/"ERROR"; o dicionário traduz para
        # o método do logger, e o default cobre algum valor novo.
        getattr(log, {"INFO": "info", "WARNING": "warning", "ERROR": "error"}.get(level, "info"))(
            "[%s] %s", event, message
        )
        if self._event_callback is not None:
            try:
                self._event_callback(level, event, message)
            except Exception:  # pragma: no cover - callback nunca deve derrubar a câmera
                log.exception("Falha no callback de evento da câmera")

    def _push(self, frame: np.ndarray, info: FrameInfo) -> None:
        """Entrega o frame; se a fila estiver cheia, descarta o mais antigo."""
        # Política "descarta o mais antigo", e não "descarta o novo": o
        # quadro que acabou de chegar é o mais recente, e inferência lenta não
        # deve reprocessar o passado. A fila é minúscula de propósito.
        try:
            self._queue.put_nowait((frame, info))
            self.stats.frames_delivered += 1
        except queue.Full:
            try:
                # Corrida com o consumidor: entre este get e o put a fila pode
                # mudar de estado. Perder o frame aqui é aceitável, falhar a
                # captura não é, então qualquer exceção aqui vira só um drop.
                self._queue.get_nowait()
                self.stats.dropped_frames += 1
                self._queue.put_nowait((frame, info))
                self.stats.frames_delivered += 1
            except (queue.Empty, queue.Full):
                self.stats.dropped_frames += 1

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        """Sobe a thread de captura. Chamar de novo não faz nada."""
        with self._lock:
            # Idempotente: o dashboard pode apertar "conectar" várias vezes.
            # Sem este guard, nasceria uma segunda thread de captura, cada uma
            # com seu próprio VideoCapture no mesmo RTSP.
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            # daemon=True: se o processo morrer, a thread morre junto em vez
            # de prender o encerramento por causa do FFmpeg bloqueado.
            self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
            self._thread.start()
            self._emit("INFO", "camera_start", f"Iniciando captura ({self.mode.value})")

    def stop(self, timeout: float = 5.0) -> None:
        """Para a captura. Não espera a thread morrer para sempre."""
        self._stop.set()
        try:
            # None é o sinal de encerramento: acorda quem estiver preso em
            # read() esperando frame. Se a fila estiver cheia, o leitor acorda
            # sozinho assim que consumir um frame.
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            # 5 s é generoso mas finito de propósito: cap.release() pode ficar
            # preso no FFmpeg, e stop() vem de request HTTP que não pode
            # ficar pendurado. wait_closed() cobre quem precisar garantir.
            thread.join(timeout=timeout)
        self._thread = None
        self._release()
        self.stats.state = CameraState.OFFLINE
        self._emit("INFO", "camera_stop", "Captura encerrada")

    def wait_closed(self, timeout: float = 20.0) -> bool:
        """Espera a thread de captura terminar de fato.

        Útil em testes e no encerramento: o ``cv2.VideoCapture`` pode ficar
        bloqueado dentro do FFmpeg por alguns segundos, e uma thread viva
        depois de ``stop()`` continua consumindo CPU e interfere em outros
        testes. Retorna ``True`` se a thread morreu dentro do tempo.
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def read(self, timeout: float = 2.0) -> tuple[np.ndarray, FrameInfo] | None:
        """Consome um frame da fila. ``None`` = fim da fonte ou vazio."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        # Timeout curto de propósito: o worker de IA chama isto em ciclo e não
        # pode ficar preso aqui enquanto a câmera está reconectando.
        return item  # None explícito = sinal de encerramento

    def latest(self) -> tuple[np.ndarray, FrameInfo] | None:
        """Pega o frame mais recente disponível sem bloquear."""
        item = None
        try:
            # Esvazia a fila e devolve só a última: quem quer o quadro mais
            # novo não deve pagar o preço de processar os que já são velhos.
            while True:
                item = self._queue.get_nowait()
        except queue.Empty:
            return item

    @property
    def is_online(self) -> bool:
        """True só quando a fonte está de fato ONLINE (não reconectando)."""
        return self.stats.state is CameraState.ONLINE

    # ------------------------------------------------------------- interno
    def _release(self) -> None:
        """Fecha o VideoCapture e anula a referência."""
        cap = self._cap
        # Zera o ponteiro ANTES de liberar: se a thread de leitura pegar
        # self._cap depois daqui, ela vê None em vez de um objeto já
        # liberado (ler de um capture liberado estoura erro no OpenCV).
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.stats.opened = False

    def _build_capture(self) -> cv2.VideoCapture | None:
        """Abre a fonte conforme o modo."""
        if self.mode is SourceMode.IMAGE:
            path = Path(self.url)
            if not path.is_file():
                self._emit("ERROR", "image_missing", f"Imagem não encontrada: {path}")
                return None
            # IMREAD_COLOR força BGR de 3 canais mesmo em PNG com alpha: o
            # detector espera sempre 3 canais.
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                self._emit("ERROR", "image_invalid", f"Não foi possível ler a imagem: {path}")
                return None
            self.stats.width, self.stats.height = img.shape[1], img.shape[0]
            # 0.0 = "não há taxa de quadros"; evita divisão por zero em quem
            # calcula FPS a partir daqui.
            self.stats.source_fps = 0.0
            self._image_frame = img
            # Convenção invertida neste ramo: None é SUCESSO (não existe
            # VideoCapture para imagem). _run() trata esse caso à parte.
            return None  # imagem não usa VideoCapture

        if self.mode is SourceMode.VIDEO:
            path = Path(self.url)
            if not path.is_file():
                self._emit("ERROR", "video_missing", f"Vídeo não encontrado: {path}")
                return None
            cap = cv2.VideoCapture(str(path))
        else:
            # A abertura COM timeout vem primeiro: sem ela, o FFmpeg bloqueia
            # na sua espera padrão (dezenas de segundos) e o
            # CAMERA_OPEN_TIMEOUT_SEC configurado seria uma mentira.
            cap = None
            # "any" fica de fora de propósito: sem transporte fixo o FFmpeg
            # negocia sozinho e não aceita receber o parametro de timeout.
            if self.transport in ("tcp", "udp"):
                try:
                    cap = cv2.VideoCapture(
                        self.url,
                        cv2.CAP_FFMPEG,
                        # open_timeout_sec está em SEGUNDOS; a propriedade do
                        # OpenCV é em MILISSEGUNDOS.
                        [int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), int(self.open_timeout_sec * 1000)],
                    )
                except Exception as exc:
                    # Build do OpenCV sem suporte a params (ex.: sem FFMPEG).
                    log.debug("Abertura com timeout não suportada: %s", exc)
                    cap = None
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                # Rede de segurança: tenta o caminho antigo, sem timeout.
                try:
                    cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                except Exception as exc:
                    log.warning("Falha ao abrir a câmera: %s", exc)
                    return None

        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None

        # Metadados do sensor vêm da câmera, não do frame: o `or 0` protege
        # contra stream que responde 0/não responde para essas propriedades.
        self.stats.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.stats.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.stats.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return cap

    def _run(self) -> None:  # pragma: no cover - laço de I/O de longa duração
        """Laço de estados: abre a fonte, lê até falhar, reabre."""
        retries = 0
        # Contador de frames que ATRAVESSA reconexões de propósito: para quem
        # consome, a mesma câmera não pode parecer um fluxo novo a cada queda.
        frame_index = 0
        # Atributo criado aqui, e não em __init__, porque _build_capture (que o
        # preenche) só roda dentro desta thread.
        self._image_frame: np.ndarray | None = None

        while not self._stop.is_set():
            # CONNECTING só na primeira conexão da vida da fonte. Qualquer
            # tentativa seguinte é RECONNECTING, que o dashboard mostra como
            # "instável" em vez de "conectando".
            self.stats.state = (
                CameraState.CONNECTING if retries == 0 and self.stats.frames_captured == 0
                else CameraState.RECONNECTING
            )
            self.stats.connect_attempts += 1

            cap = self._build_capture()
            if cap is None and self.mode is not SourceMode.IMAGE:
                self.stats.opened = False
                retries += 1
                self._emit("ERROR", "connect_failed", f"Não foi possível abrir a fonte: {self.url}")
                # max_retries == 0 (produção) nunca entra neste if: espera e
                # tenta de novo para sempre, porque queda de rede é normal.
                if self.max_retries and retries >= self.max_retries:
                    self.stats.state = CameraState.ERROR
                    self.stats.last_error = "Falha ao abrir a fonte"
                    self._emit("ERROR", "camera_error", "Limite de tentativas atingido")
                    return
                # wait() em vez de sleep: um stop() durante a espera encurta
                # o ciclo na hora.
                self._stop.wait(self.retry_delay_sec)
                continue
            if self.mode is SourceMode.IMAGE and self._image_frame is None:
                self.stats.state = CameraState.ERROR
                self.stats.last_error = "Imagem inválida"
                return

            self._cap = cap
            # Em SourceMode.IMAGE, cap é None mesmo no caminho de sucesso, e
            # por isso `opened` fica False. Quem manda no estado de verdade é
            # `stats.state` logo abaixo, não este booleano.
            self.stats.opened = cap is not None
            self.stats.connected_at = time.time()
            self.stats.state = CameraState.ONLINE
            # Só conta "reconexão" quando uma câmera real caiu. Reiniciar um
            # vídeo em loop não é falha de conexão e não deve aparecer como
            # tal no dashboard (a métrica é usada para detectar câmera instável).
            if retries > 0 and self.mode is SourceMode.RTSP:
                self.stats.reconnects += 1
                self._emit("INFO", "camera_reconnect", f"Reconectado (tentativa {retries})")
            elif retries == 0:
                self._emit("INFO", "camera_online", "Câmera conectada")
            else:
                self._emit("INFO", "source_restarted", "Fonte reiniciada")

            # Um único ponto de leitura para os três modos; devolve o contador
            # para ele continuar de onde parou.
            frame_index = self._read_loop(frame_index)
            self._release()

            if self.mode is SourceMode.IMAGE:
                self._emit("INFO", "camera_end", "Processamento da imagem concluído")
                return
            if self.mode is SourceMode.VIDEO and not self._loop_video:
                self._emit("INFO", "camera_end", "Vídeo finalizado")
                return

            retries += 1
            if self.max_retries and retries >= self.max_retries:
                self.stats.state = CameraState.OFFLINE
                self.stats.last_error = "Fonte indisponível"
                self._emit("WARNING", "camera_offline", "Fonte indisponível; encerrando")
                return
            self.stats.state = CameraState.RECONNECTING
            if self.mode is SourceMode.VIDEO and self._loop_video:
                # Vídeo em loop não é "queda de câmera": recomeça na hora.
                # Esperar o retry_delay de RTSP (3 s) faria o dashboard
                # parecer travado entre uma passada e outra.
                self._emit("INFO", "video_loop", "Vídeo reiniciado em loop")
                continue
            self._emit("WARNING", "camera_reconnect", "Fonte caiu; tentando reconectar")
            self._stop.wait(self.retry_delay_sec)

        self._release()
        self.stats.state = CameraState.OFFLINE

    # Atributo de CLASSE, e não de __init__: o padrão é loop para todo mundo e
    # set_video_loop() cria valor próprio só na instância que muda.
    _loop_video = True  # se False, o arquivo de vídeo não reinicia

    def _read_loop(self, frame_index: int) -> int:
        """Lê frames até falhar, emitir timeout de frame ou parar."""
        # Piso de 3 s: com read_timeout_sec menor que isso, uma câmera apenas
        # lenta entraria em laço de reconexão para sempre.
        stale_limit = max(self.read_timeout_sec, 3.0)
        while not self._stop.is_set():
            # Cópia local: stop()/_release() podem anular self._cap a qualquer
            # momento; a checagem abaixo evita ler de um capture já liberado.
            cap = self._cap
            if cap is None and self.mode is not SourceMode.IMAGE:
                return frame_index

            # perf_counter e não time(): mede só a duração, sem depender da
            # hora do sistema (que o NTP pode jumpingar).
            started = time.perf_counter()
            if self.mode is SourceMode.IMAGE:
                frame = self._image_frame
                if frame is None:
                    return frame_index
            else:
                ok, frame = cap.read()  # type: ignore[union-attr]
                if not ok or frame is None:
                    self.stats.invalid_frames += 1
                    # Pode ser EOF de arquivo de vídeo.
                    if self.mode is SourceMode.VIDEO and not self._loop_video:
                        return frame_index
                    break
                if frame.size == 0:
                    self.stats.invalid_frames += 1
                    # continue e não break: frame vazio é falha momentânea. Só
                    # a checagem de travamento abaixo decide quando é a hora
                    # de reabrir a fonte.
                    continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            self.stats.frames_captured += 1
            self.stats.last_frame_at = time.time()
            self._last_ok_ts = time.time()

            info = FrameInfo(
                timestamp=time.time(),
                index=frame_index,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                source_fps=self.stats.source_fps,
                capture_latency_ms=latency_ms,
                source_mode=self.mode,
            )
            self._push(frame, info)
            frame_index += 1

            if self.mode is SourceMode.IMAGE:
                # Imagem estática: reemite devagar só para alimentar o
                # dashboard em teste, sem saturar a CPU.
                if self._stop.wait(0.5):
                    break
                continue

            # Detecção de frame congelado: se nada é lido há muito tempo e o
            # modo é RTSP, força reconexão.
            # Só dispara quando read() CONTINUA voltando (com frame vazio) sem
            # nenhum frame válido chegar. Se read() travar de verdade, quem
            # derruba a thread é o timeout do FFmpeg lá no topo do arquivo, não
            # esta checagem: a thread estaria presa dentro de cap.read().
            if self._last_ok_ts is not None and (time.time() - self._last_ok_ts) > stale_limit:
                break
        return frame_index

    def set_video_loop(self, loop: bool) -> None:
        """Faz o arquivo de vídeo reiniciar em loop (útil para testes)."""
        self._loop_video = loop
