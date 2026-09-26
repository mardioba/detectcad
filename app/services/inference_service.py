"""Pipeline de inferência: o worker que faz o trabalho real.

Roda em thread própria (nunca bloqueia o FastAPI):

    frame -> ROI -> YOLO -> pilhas -> StackCounter -> estabilidade
          -> estado global -> banco (só em mudanças) -> snapshot -> overlay

Cada frame gera:

* ``CountEstimate`` por pilha (contagem + confiança + método);
* estados :class:`PileState` estabilizados;
* imagem com overlay (boxes, rótulo, confiança) para o dashboard;
* eventos de mudança (contagem/pilha/confiança) que viram registro no banco
  e snapshot.
"""

# ARQUIVO / MAPA
# Worker de inferência: 1 thread por câmera, o único que escreve o estado.
#
# Ordem de leitura:
#   1. _PileRuntime / __init__          -> o que sobrevive entre frames
#   2. start / stop / _loop             -> ciclo de vida e ritmo (fps)
#   3. set_calibration                  -> calibração quente, sem reiniciar
#   4. _process_frame                   -> o pipeline, 11 etapas numeradas
#   5. _apply_roi / _effective_roi / _crop   -> geometria em pixels
#   6. _resolve_pile_id                 -> ID estável de pilha (câmera fixa)
#   7. _draw_overlay / _draw_status_bar -> a imagem que o dashboard vê
#   8. _persist_changes / _handle_snapshots -> banco e auditoria
#   9. _evaluate_alerts / manual_count / status -> regras de negócio
#  10. helpers de texto (fonte, _draw_block) -> overlay em português com acentos
#
# Cuidado: o passo 4 é a única fonte de contagem. Nada aqui deve inventar
# número quando falta modelo (ver _use_manual_mode).

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.ai.model_manager import ModelManager
from app.ai.pile_detector import PileDetector, build_pile_detector
from app.ai.tracker import PileTracker, group_by_x, resolve_tracker_config
from app.ai.yolo_detector import ModelNotAvailable, YoloDetector
from app.camera.camera_manager import CameraManager
from app.config import settings
from app.counting.confidence import combine as combine_confidence
from app.counting.confidence import status_color_bgr, status_label
from app.counting.stability import StabilityFilter, decide_status
from app.counting.stack_counter import StackCounter
from app.database.database import session_scope
from app.database.repository import (
    CountRepository,
    EventRepository,
    PileRepository,
)
from app.schemas import CameraState, CountStatus, Detection, PileState, localnow
from app.services.snapshot_service import SnapshotService
from app.services.state_store import StateStore, state_store

log = logging.getLogger("ai")

# Fator de escala do overlay sobre o frame original.
# Resquício de uma versão anterior que redimensionava o overlay para 1280 px.
# Hoje o overlay sai no tamanho do ROI, sem resize. A constante ficou para não
# quebrar importações externas, mas não é usada pelo pipeline.
OVERLAY_WIDTH = 1280


@dataclass
class _PileRuntime:
    """Estado interno de uma pilha entre frames.

    Não vai para o banco nem para o dashboard: é a memória do worker sobre
    cada pilha, para conseguir um ``pile_id`` estável mesmo quando o tracker
    troca o ``tracking_id`` (occlusão, troca de ID do ByteTrack).
    """

    pile_id: int
    # ID que o tracker deu nesta frame. Muda; pile_id não.
    tracking_id: int
    bbox: tuple[float, float, float, float]
    first_seen_ts: float
    # Frames seguidos sem detecção. Chegando a miss_grace_frames a pilha morre.
    missed: int = 0
    # Último valor que JÁ foi gravado no banco. É o gatilho de mudança do
    # passo 9: o banco só é tocado quando stable_count != last_recorded.
    last_recorded: int | None = None
    # Travado pelo operador: a IA para de sobrescrever esta pilha.
    manual: bool = False
    # Último recorte da pilha. Guardado para diagnóstico/reuso; o recorte do
    # passo 5 é sempre refeito porque o frame muda.
    roi: np.ndarray | None = None
    last_seen: float = 0.0


class InferenceService:
    """Worker de IA de uma câmera."""

    def __init__(
        self,
        detector: YoloDetector,
        model_manager: ModelManager,
        camera_manager: CameraManager,
        camera_id: int = 1,
        store: StateStore | None = None,
    ) -> None:
        self.detector = detector
        self.models = model_manager
        self.cameras = camera_manager
        self.camera_id = camera_id
        self.store = store or state_store

        # IoU 0.30 (e não o 0.45 do NMS) porque aqui o tracker casa pilhas
        # largas e sobrepostas: com 0.45 o tracker partia pilhas ao meio.
        self.tracker = PileTracker(iou_threshold=0.30, max_age=settings.counting.miss_grace_frames)
        self.stability = StabilityFilter()
        self.counter = StackCounter(
            method=settings.counting.counting_method,
            chair_height_px=settings.counting.chair_height_px,
        )
        # Construído com preguiça no primeiro frame: só aí sabemos o tamanho
        # real da imagem e os nomes de classe do modelo carregado.
        self.pile_detector: PileDetector | None = None
        self.snapshots = SnapshotService(camera_id=camera_id, camera_name=settings.camera.camera_name)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._piles: dict[int, _PileRuntime] = {}
        # Sequência de pile_id: só cresce. Nunca reaproveita ID de pilha morta,
        # senão o banco confundiria pilhas diferentes.
        self._pile_id_seq = 1
        # Protege _piles/_pile_id_seq/calibração: a thread do worker mexe neles
        # a cada frame enquanto a API pode chamar manual_count() ao mesmo
        # tempo. RLock (e não Lock) porque set_calibration chama build_pile_detector
        # e as Operações aninhadas são normativas aqui.
        self._lock = threading.RLock()
        # Mantido por compatibilidade; o gatilho de snapshot hoje é a lista
        # `changed` devolvida por _persist_changes (ver _handle_snapshots).
        self._last_snapshot_total: int | None = None
        # Estados LOW_CONFIDENCE/UNKNOWN da frame anterior. O snapshot de
        # baixa confiança é disparado pela DIFERENÇA de conjuntos (borda de
        # subida), senão gravaria uma imagem por frame enquanto a pilha
        # continuasse instável.
        self._last_low_conf: set[int] = set()
        # Último erro de persistência, exposto em status() para o dashboard
        # mostrar que a contagem está só em memória.
        self._persist_error: str = ""
        self._calibration: dict[str, Any] = {}

        # Métricas
        self.frames_processed = 0
        self.last_process_fps = 0.0
        self.last_infer_ms = 0.0
        self.last_count_ms = 0.0
        # ROI em uso neste momento, em pixels do frame original.
        self.roi: tuple[int, int, int, int] | None = None
        # Timestamps das últimas frames, só para calcular FPS. Janela curta de
        # propósito: reflete a média de um instante, não da sessão inteira.
        self._fps_window: list[float] = []

    # ------------------------------------------------------------------ ciclo
    def start(self) -> None:
        with self._lock:
            # Início idempotente: a API pode chamar start() várias vezes.
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            # daemon=True: se o processo morrer, a thread não segura o encerramento.
            self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
            self._thread.start()
            log.info("Worker de inferência iniciado (camera_id=%s)", self.camera_id)

    def stop(self, timeout: float = 5.0) -> None:
        # _stop.set() pede a parada; o join dá tempo de a thread fechar a
        # própria iteração (no máximo um período de sleep).
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        # Mesmo se o join estourar o timeout, a referência é solta: a thread é
        # daemon e não pode impedir o shutdown do servidor.
        self._thread = None
        log.info("Worker de inferência parado")

    def set_calibration(self, calibration: dict[str, Any]) -> None:
        """Aplica calibração vinda do banco/API sem reiniciar o worker.

        Chaveado pelo _lock porque roda na thread da API enquanto o worker
        processa frames; sem o lock o contador poderia ser lido no meio da
        troca de método e gerar uma contagem com parâmetros misturados.
        """
        with self._lock:
            # Cópia rasa: o dict do chamador pode ser reutilizado/mutado lá fora.
            self._calibration = dict(calibration or {})
            count_cfg = self._calibration.get("counting", {})
            # "or 0.0" no chair_height_px: 0 significa "não calibrado" e é um
            # valor legítimo, então não pode ser trocado pelo default do .env.
            self.counter.chair_height_px = float(
                count_cfg.get("chair_height_px", settings.counting.chair_height_px) or 0.0
            )
            self.counter.min_confidence = float(
                count_cfg.get("min_confidence", settings.counting.min_confidence)
            )
            self.counter.chair_weight = float(
                count_cfg.get("chair_weight", settings.ai.chair_weight)
            )
            self.counter.count_offset = int(count_cfg.get("count_offset", 0) or 0)
            if "counting_method" in count_cfg:
                self.counter.method = str(count_cfg["counting_method"])
            self.stability.window = int(
                count_cfg.get("stability_frames", settings.counting.count_stability_frames)
            )
            self.stability.change_min_frames = int(
                count_cfg.get("change_min_frames", settings.counting.change_min_frames)
            )
            self.stability.min_agreement = float(
                count_cfg.get("stability_min_confidence", settings.counting.stability_min_confidence)
            )
            ai_cfg = self._calibration.get("ai", {})
            # Aqui sim mutamos settings.ai, que é global do processo: é o que
            # faz o limiar novo valer também no YoloDetector, sem reiniciar.
            if "confidence_threshold" in ai_cfg:
                settings.ai.confidence_threshold = float(ai_cfg["confidence_threshold"])
            if "chair_classes" in ai_cfg and ai_cfg["chair_classes"]:
                settings.ai.chair_classes = str(ai_cfg["chair_classes"])
            if "pile_classes" in ai_cfg and ai_cfg["pile_classes"]:
                settings.ai.pile_classes = str(ai_cfg["pile_classes"])
            if self.pile_detector is not None:
                # Reconstrói para pegar as classes novas. Perde o image_size,
                # que é reposto no primeiro frame por configure((w, h)).
                self.pile_detector = build_pile_detector(self.detector.class_names)
        log.info("Calibração aplicada: %s", self._calibration)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        # Periodo alvo do ciclo. process_fps é "frames por segundo processados",
        # não o FPS da câmera: a câmera pode entregar 30 fps e processamos 3.
        interval = 1.0 / max(0.1, settings.camera.process_fps)
        runtime = self.cameras.get(self.camera_id)
        if runtime is None:
            # Nenhuma câmera registrada ainda: sobe a do .env.
            runtime = self.cameras.start_default()
        self._fps_window.clear()

        while not self._stop.is_set():
            loop_start = time.perf_counter()
            try:
                item = runtime.take_latest()
            except Exception as exc:  # pragma: no cover
                log.exception("Falha ao ler frame")
                item = None

            if item is None:
                # Sem frame: reflete o estado da câmera e aguenta.
                self._update_camera_status(runtime)
                # Espera curta em vez de `time.sleep`: é o que permite parar
                # a thread imediatamente, sem esperar o período inteiro.
                if self._stop.wait(0.1):
                    break
                continue

            frame, info = item
            try:
                self._process_frame(frame, info)
            except Exception as exc:
                # Uma frame com erro não derruba o worker nem para de contar;
                # o erro aparece no status e o dashboard mostra o overlay antigo.
                log.exception("Erro ao processar frame %s", info.index)
                self.store.set_status(last_error=str(exc))

            elapsed = time.perf_counter() - loop_start
            # O sleep é o que sobra do período, medido do início do ciclo: a
            # inferência gasta parte do orçamento sem empurrar o ritmo.
            sleep_for = interval - elapsed
            if sleep_for > 0 and self._stop.wait(sleep_for):
                break

        log.info("Loop de inferência encerrado")

    def _update_camera_status(self, runtime: Any) -> None:
        st = runtime.source.stats
        self.store.set_status(
            camera_state=st.state.value,
            camera_stats=st.as_dict(),
            reconnects=st.reconnects,
        )

    # -------------------------------------------------------------- pipeline
    def _process_frame(self, frame: np.ndarray, info: Any) -> None:
        """Roda o pipeline inteiro de uma frame. Roda só na thread do worker."""
        t_start = time.perf_counter()
        h, w = frame.shape[:2]
        detections: list[Detection] = []
        # detections_ok é separado de `detections` de propósito: uma frame sem
        # detecção é resultado legítimo; uma frame com inferência quebrada não é.
        detections_ok = False
        model_error = ""

        # --- 1. inferência -------------------------------------------------
        if self.detector.is_loaded:
            try:
                # image_size é o tamanho DO FRAME CHEIO, não da ROI: é o que
                # o PileDetector usa para detectar pilha cortada pela borda.
                if self.pile_detector is None:
                    self.pile_detector = build_pile_detector(self.detector.class_names, (w, h))
                else:
                    self.pile_detector.configure((w, h))
                chairs, piles_cls = self.pile_detector.chair_classes, self.pile_detector.pile_classes
                tracker_mode, tracker_yaml = resolve_tracker_config(settings.ai.tracker)
                # O tracker do Ultralytics só entra quando o modelo devolve a
                # pilha inteira (classe "pile"). Sem isso, a inferência simples
                # é mais barata e o tracking é feito pelo PileTracker interno.
                if tracker_mode == "ultralytics" and piles_cls:
                    dets, infer_ms = self.detector.track(
                        frame, tracker=tracker_yaml or "bytetrack.yaml"
                    )
                else:
                    # "chairs | piles_cls" une os conjuntos; o "or None" é para o
                    # detector não filtrar por classe quando o modelo é genérico.
                    dets, infer_ms = self.detector.predict(frame, classes=chairs | piles_cls or None)
                detections = dets
                detections_ok = True
                self.last_infer_ms = infer_ms
            except ModelNotAvailable:
                model_error = "Modelo não carregado"
            except Exception as exc:
                # Não levanta: um erro isolado (frame corrompido, CUDA OOM)
                # não pode parar o loop.
                model_error = str(exc)
                log.exception("Falha na inferência")
        else:
            # Distingue "está usando o modelo genérico da Ultralytics" de
            # "ainda não tem o modelo da empresa": orientam o operador a ações
            # diferentes na tela de preparação.
            model_error = self.detector.load_error or (
                "Modelo específico ainda não treinado."
                if not self.detector.is_pretrained_generic()
                else "Modelo genérico pré-treinado (não recomendado para contagem)."
            )

        # --- 2. ROI ---------------------------------------------------------
        # Ordem deliberada: a inferência roda no frame inteiro e o recorte vem
        # depois, com as caixas transladadas de volta. Cortar antes faria o
        # modelo receber um recorte do qual ele nunca foi treinado, e mudaria
        # as coordenadas que o tracker usa para distinguir pilhas.
        frame_proc, offset = self._apply_roi(frame)
        if offset != (0, 0) and detections_ok:
            # -offset: volta do sistema do recorte para o do frame cheio.
            detections = [_shift(d, -offset[0], -offset[1]) for d in detections]

        # --- 3. pilhas ------------------------------------------------------
        candidates: list = []
        if detections_ok and self.pile_detector is not None:
            candidates = self.pile_detector.detect(detections)
        elif not detections_ok and self._use_manual_mode():
            # Modo sem modelo: nada é inventado; o dashboard mostra vazio.
            candidates = []

        # --- 4. tracking ----------------------------------------------------
        # Só é chamado quando há candidatos: quando a cena esvazia, deixar o
        # tracker sem envelhecer é mais barato. A pilha ainda expira, porque a
        # etapa 6 conta os `missed` no _PileRuntime.
        tracked = self.tracker.update(candidates) if candidates else []

        # --- 5. contagem por pilha -----------------------------------------
        t_count = time.perf_counter()
        pile_states: list[PileState] = []
        seen_ids: set[int] = set()

        for tracking_id, cand, track in tracked:
            # _resolve_pile_id já criou o _PileRuntime se for pilha nova, por
            # isso o self._piles[pile_id] logo abaixo nunca dá KeyError.
            pile_id = self._resolve_pile_id(tracking_id, cand)
            seen_ids.add(pile_id)
            roi = self._crop(frame_proc, cand.bbox)
            est = self.counter.count(
                roi,
                chair_detections=cand.chair_detections,
                pile=cand,
                # previous_count é só referência de consistência dentro do
                # StackCounter: usa-se o valor já gravado, não a estimativa
                # anterior, para que uma flutuação não se auto-reforce.
                previous_count=self._piles[pile_id].last_recorded,
            )
            stab = self.stability.update(pile_id, est.count)
            # A confiança exibida NÃO é a do YOLO: é a chance de a contagem
            # estar certa (ver app/counting/confidence.py).
            confidence = combine_confidence(
                # "max" e não média: se a caixa da pilha ou uma cadeira dentro
                # dela foi bem vista, a detecção é boa.
                detection_confidence=max(est.detection_confidence, cand.conf),
                stability_confidence=stab.stability_confidence,
                agreement=self._agreement_from(est),
                pile_confidence=est.confidence,
                min_confidence=settings.counting.min_confidence,
            )
            status = decide_status(
                count=stab.stable_count,
                confidence=confidence.confidence,
                stability_confidence=stab.stability_confidence,
                partial=cand.partial,
            )
            runtime_pile = self._piles[pile_id]
            runtime_pile.bbox = cand.bbox
            runtime_pile.roi = roi
            # missed zerado aqui: a pilha foi vista neste frame.
            runtime_pile.missed = 0
            runtime_pile.last_seen = time.time()

            # O que vai para o dashboard: raw_count é a leitura crua desta
            # frame; stable_count é o que a janela de estabilidade projetou.
            state = PileState(
                pile_id=pile_id,
                tracking_id=tracking_id,
                bbox=cand.bbox,
                raw_count=est.count,
                stable_count=stab.stable_count,
                confidence=confidence.confidence,
                detection_confidence=confidence.detection_confidence,
                stability_confidence=confidence.stability_confidence,
                status=status,
                method=est.method,
                candidates=est.candidates,
                pitch_px=est.pitch_px,
                chair_height_px=est.chair_height_px,
                partial=cand.partial,
                missed_frames=0,
                manual=runtime_pile.manual,
                first_seen=datetime_from(runtime_pile.first_seen_ts),
                last_seen=localnow(),
                notes=est.notes,
            )
            pile_states.append(state)

        # Tempo só da contagem (passo 5), sem inferência nem banco: é o número
        # que diz se o gargalo é o algoritmo ou o YOLO.
        self.last_count_ms = (time.perf_counter() - t_count) * 1000.0

        # --- 6. pilhas que sumiram ------------------------------------------
        appeared: list[int] = []
        disappeared: list[int] = []
        for pile_id, rt in list(self._piles.items()):
            if pile_id in seen_ids:
                continue
            rt.missed += 1
            # Tolerância a oclusão/câmera treme: some de vez só depois de
            # miss_grace_frames frames seguidos sem ver a pilha.
            if rt.missed > settings.counting.miss_grace_frames:
                self._piles.pop(pile_id, None)
                # A janela de estabilidade morre junto, senão a pilha volta
                # com uma contagem velha de N frames atrás.
                self.stability.remove(pile_id)
                disappeared.append(pile_id)

        # --- 7. totais e status ---------------------------------------------
        # O store precisa saber se as contagens deste frame são medições,
        # ANTES de calcular o total: sem modelo treinado o total oficial tem
        # de ser zero, mesmo que uma pilha tenha alcançado status STABLE (o
        # filtro de janela estabiliza um número estável, mas não um número
        # certo).
        self.store.set_counts_measurable(self._counts_are_measurable())
        self.store.set_piles(pile_states)
        total = self.store.total(only_stable=True)
        breakdown = self.store.totals_breakdown()
        # Pilha "nova" = vista há menos de 3 s.
        # Heurística de janela curta, não histórico do tracker: com
        # process_fps baixo, 3 s dão algumas frames de margem; com fps alto
        # pode disparar duas vezes para a mesma pilha, e isso é aceito
        # (o snapshot duplicado é barato, perder o evento não seria).
        now = localnow()
        appeared = [s.pile_id for s in pile_states if (now - s.first_seen).total_seconds() < 3.0]

        # --- 8. overlay ------------------------------------------------------
        # O overlay nasce do frame já recortado: o dashboard enxerga só a ROI.
        annotated = self._draw_overlay(frame_proc, pile_states, model_error, offset)
        # 82 fixo aqui (e não settings.snapshots.jpeg_quality): é o stream, que
        # vai dozens de vezes por segundo. O arquivo de auditoria usa 85.
        jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])[1].tobytes()
        self.store.set_frame(jpeg, annotated, time.time())

        # --- 9. mudanças -> banco + snapshot ---------------------------------
        changed = self._persist_changes(pile_states, total, breakdown["pile_count"])

        # --- 10. histórico ---------------------------------------------------
        # Um ponto por frame processado (não por frame da câmera). O store
        # descarta os mais antigos: 720 pontos ≈ 4 min a 3 fps.
        self.store.push_history(
            {
                "t": time.time(),
                "timestamp": localnow().isoformat(timespec="seconds"),
                "total": total,
                "piles": len(pile_states),
                "confidence": round(self.store.overall_confidence(), 4),
            }
        )

        # --- 11. métricas e status -------------------------------------------
        dt = time.perf_counter() - t_start
        self.frames_processed += 1
        # Janela deslizante de 20 timestamps: FPS medido de verdade, não o
        # valor configurado (que é só o alvo do sleep).
        self._fps_window.append(time.time())
        if len(self._fps_window) > 20:
            self._fps_window.pop(0)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                # (N-1) intervalos em N timestamps.
                self.last_process_fps = (len(self._fps_window) - 1) / span

        alerts = self._evaluate_alerts(total, pile_states)
        # set_status faz merge: campos omitidos aqui (ex.: roi_enabled) seguem
        # com o valor de quem os escreveu antes.
        self.store.set_status(
            camera_state=runtime_state(self),
            mode=self._mode_label(detections_ok),
            model_loaded=self.detector.is_loaded,
            model_trained=self.detector.trained_model,
            model_error=model_error,
            model_filename=self.detector.configured_path.name,
            process_fps=self.last_process_fps,
            inference_ms=self.last_infer_ms,
            counting_ms=self.last_count_ms,
            total_chairs=total,
            pile_count=len(pile_states),
            frame_index=info.index,
            frame_size=[w, h],
            frames_processed=self.frames_processed,
            alerts=alerts,
            roi=self.roi,
        )

        # snapshots
        # Por último: usa `annotated` (mesma imagem enviada ao dashboard) e
        # `changed`, que só existe depois do passo 9.
        self._handle_snapshots(
            annotated, pile_states, total, changed, appeared, disappeared
        )

    # ------------------------------------------------------------------ ROI
    def _apply_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Recorta a ROI principal. Fora dela, nada é processado.

        Devolve (frame_recortado, offset) em pixels do frame original.
        """
        roi = self._effective_roi(frame.shape[1], frame.shape[0])
        if roi is None:
            self.roi = None
            return frame, (0, 0)
        x, y, w, h = roi
        # ROI fora do frame (câmera mudou de resolução) não é erro fatal:
        # cai para o frame inteiro e segue contando.
        if w <= 0 or h <= 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            self.roi = None
            return frame, (0, 0)
        crop = frame[y : y + h, x : x + w]
        self.roi = roi
        # ascontiguousarray: a fatia de numpy é uma view com passo; o
        # ascontiguousarray devolve memória contígua, que o OpenCV e o YOLO
        # acessam bem mais rápido. Custo de uma cópia, ganho de alguns FPS.
        return np.ascontiguousarray(crop), (x, y)

    def _effective_roi(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
        """ROI vinda do banco; senão a do .env; senão a imagem inteira.

        Ordem de precedência: calibration_enabled decide se existe ROI;
        a do banco (dashboard) ganha sobre a do CALIBRATION_ROI do .env.
        w/h são limitados à imagem, porque a câmera pode ter mudado de
        resolução desde que o operador desenhou a ROI.
        """
        cam_roi = self._calibration.get("roi") or {}
        enabled = bool(self._calibration.get("roi_enabled", settings.calibration.enabled))
        if not enabled:
            return None
        if cam_roi and cam_roi.get("w", 0) > 0 and cam_roi.get("h", 0) > 0:
            return (
                int(cam_roi["x"]),
                int(cam_roi["y"]),
                min(int(cam_roi["w"]), frame_w - int(cam_roi["x"])),
                min(int(cam_roi["h"]), frame_h - int(cam_roi["y"])),
            )
        env = settings.calibration.roi.strip()
        if env:
            try:
                # Aceita "x,y,w,h" com espaços; o [:4] ignora sobra.
                x, y, w, h = (int(v) for v in env.replace(" ", "").split(",")[:4])
                return (x, y, min(w, frame_w - x), min(h, frame_h - y))
            except ValueError:
                # .env com typo: avisa e segue com a imagem inteira, em vez de
                # derrubar o worker.
                log.warning("CALIBRATION_ROI inválida: %r (use x,y,w,h)", env)
        return None

    def _crop(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Recorte da pilha, com coordenadas travadas dentro da imagem.

        bbox vem do YOLO em float e pode sair 1px da borda. Sem o clamp, um
        x2 menor que x1 faria o slice do Python voltar pelo indice negativo e
        devolver a faixa errada da imagem.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1i = int(max(0, min(w - 1, x1)))
        y1i = int(max(0, min(h - 1, y1)))
        # max(x1i+1, ...) garante largura mínima de 1px: um recorte vazio
        # quebraria a análise de camadas lá no StackCounter.
        x2i = int(max(x1i + 1, min(w, x2)))
        y2i = int(max(y1i + 1, min(h, y2)))
        return frame[y1i:y2i, x1i:x2i]

    # ---------------------------------------------------------------- IDs
    def _resolve_pile_id(self, tracking_id: int, cand: Any) -> int:
        """ID estável de pilha: sobrevive a-loss do tracker (câmera fixa).

        Duas passadas:
        1. o tracking_id ainda é conhecido -> devolve o pile_id de sempre;
        2. o tracker trocou o ID -> casa por sobreposição horizontal, que é
           a única coisa estável num enquadramento fixo de câmera.
        Sem casamento, é pilha nova.
        """
        for pid, rt in self._piles.items():
            if rt.tracking_id == tracking_id:
                return pid
        # Sem correspondência por track: tenta por proximidade horizontal.
        best_pid, best_score = None, 0.0
        for pid, rt in self._piles.items():
            rb = rt.bbox
            ov = min(rb[2], cand.x2) - max(rb[0], cand.x1)
            if ov <= 0:
                continue
            # Só o eixo X entra no score: pilhas são lado a lado, então o
            # quanto se cobrem na horizontal é o sinal mais forte. Normalizar
            # pela largura da candidata mantém o limiar em 0..1.
            score = ov / max(1e-6, cand.width)
            if score > best_score:
                best_pid, best_score = pid, score
        # max(0.25, pile_overlap_min) evita juntar pilhas diferentes só porque
        # encostam. O piso de 0.25 cobre o caso de .env sem valor.
        if best_pid is not None and best_score >= max(0.25, settings.counting.pile_overlap_min):
            # Rebinding: a partir de agora as duas pontas apontam para a
            # mesma pilha, então o próximo frame volta pelo caminho rápido.
            self._piles[best_pid].tracking_id = tracking_id
            return best_pid
        pid = self._pile_id_seq
        self._pile_id_seq += 1
        self._piles[pid] = _PileRuntime(
            pile_id=pid, tracking_id=tracking_id, bbox=cand.bbox, first_seen_ts=time.time()
        )
        return pid

    # ------------------------------------------------------------ overlay
    def _draw_overlay(
        self,
        frame: np.ndarray,
        piles: list[PileState],
        model_error: str,
        offset: tuple[int, int],
    ) -> np.ndarray:
        """Desenha caixas e rótulos. Recebe a imagem já recortada (ROI)."""
        # copy(): o frame original continua intacto para o próximo passo.
        out = frame.copy()
        # As caixas vivem no sistema do recorte, mas o status bar é desenhado
        # na imagem recortada também. Os nomes scale_x/scale_y vêm do
        # chamador como offset da ROI e, na prática, são o mesmo par (x, y).
        scale_x = offset[0]
        scale_y = offset[1]
        for st in piles:
            x1, y1, x2, y2 = st.bbox
            x1 += scale_x
            x2 += scale_x
            y1 += scale_y
            y2 += scale_y
            # A cor vem do status: verde estável, amarelo instável, vermelho
            # sem confiança. O operador lê o estado sem ler o texto.
            color = status_color_bgr(st.status)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            if st.status is not CountStatus.STABLE:
                # tracejado para contagem instável / baixa confiança / parcial
                # (a linha sólida de baixo fica, para a caixa não sumir)
                _dashed_rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color)
            # "(manual)" na etiqueta é a marca de que o número veio do operador:
            # sem ela não dava para saber por que a IA parou de mexer.
            # Mesmo critério do dashboard: só escreve um número quando o
            # status diz que ele é confiável. Um "7 cadeiras" desenhado em
            # cima da pilha é lido como medida mesmo quando a confiança é
            # 30%, e é exatamente esse número que o operador anota no papel.
            trusted = self._counts_are_measurable() and st.status in (
                CountStatus.STABLE,
                CountStatus.PARTIAL,
            )
            if st.manual:
                count_text = f"{st.stable_count} cadeiras (manual)"
            elif trusted:
                count_text = f"{st.stable_count} cadeiras"
            else:
                count_text = "? cadeiras"
            # Um bloco só (fundo único) para as 3 linhas: ver _draw_block.
            _draw_block(
                out,
                [
                    f"PILHA {st.pile_id}",
                    count_text,
                    f"{st.confidence * 100:.0f}% {status_label(st.status)}",
                ],
                int(x1),
                int(y1) + 2,
                color,
                # scale 0.5: metade do tamanho padrão, para caber 3 linhas
                # sem cobrir a pilha.
                scale=0.5,
                line_gap=3,
            )
        # barra de status
        self._draw_status_bar(out, piles, model_error)
        return out

    def _draw_status_bar(self, out: np.ndarray, piles: list[PileState], model_error: str) -> None:
        """Faixa fixa no topo: pilhas, total, confiança, FPS e a ROI."""
        h, w = out.shape[:2]
        bar_h = 30
        cv2.rectangle(out, (0, 0), (w, bar_h), (30, 30, 30), -1)
        # Lê do store, não dos argumentos: é o mesmo total que o dashboard
        # mostra, então a imagem nunca discorda com o número da tela.
        total = self.store.total(only_stable=True)
        conf = self.store.overall_confidence()
        parts = [
            f"PILHAS: {len(piles)}",
            f"TOTAL: {total}",
            f"CONFIANÇA: {conf * 100:.0f}%",
            f"FPS: {self.last_process_fps:.1f}",
        ]
        if self.roi:
            # Mostra o tamanho útil da ROI: se estiver pequena demais, a
            # contagem cai e a culpa costuma estar aqui.
            parts.append(f"ROI: {self.roi[2]}x{self.roi[3]}")
        x = 8
        for part in parts:
            _draw_text(out, part, x, 21, (255, 255, 255), scale=0.52)
            tw, _th = _text_size(part, 0.52)
            # 26 px de espaço: o espaçamento é fixo, não medido, porque o
            # campo seguinte já tem separadores visuais.
            x += tw + 26
        if model_error:
            # Faixa no rodapé (BGR 60,40,130 = vermelho escuro) para o erro
            # ficar visível mesmo se o operador só olhar a imagem.
            cv2.rectangle(out, (0, h - 26), (w, h), (60, 40, 130), -1)
            # Corta em 110 caracteres: msg de exceção longa viraria barra
            # gigante e cobriria as pilhas.
            _draw_text(out, f"! {model_error[:110]}", 8, h - 7, (255, 255, 255), scale=0.5)

    # -------------------------------------------------------- persistência
    def _counts_are_measurable(self) -> bool:
        """As contagens deste frame são medições, ou só diagnóstico?

        Só são medições com um modelo **treinado para este ambiente**. Com o
        modelo genérico pré-treinado, os estimadores de textura (periodic,
        extent) medem ripas e nervuras da cadeira, não cadeiras - e como todos
        erram parecido, a confiança sobe e o número sai errado com cara de
        certo. Nesse regime o dashboard mostra "?" e nada vai para o histórico.

        Uma correção manual do operador continua valendo: ele viu a pilha de
        perto, a IA não. Por isso ``manual_count`` grava sempre.
        """
        return bool(self.detector.is_loaded and self.detector.trained_model)

    def _persist_changes(
        self, piles: list[PileState], total: int, pile_count: int
    ) -> list[int]:
        """Grava no banco **somente** quando a contagem muda de fato.

        Regras:
        * uma pilha nova ou que reapareceu com valor diferente gera registro;
        * uma pilha cujo valor estabilizado mudou gera registro com o
          ``previous_count`` preenchido (auditoria antes/depois);
        * valores idênticos não geram nada - evita milhares de linhas.

        Devolve a lista de pile_ids que mudaram: ela é o gatilho do
        snapshot "contagem mudou" e chega pronta para o passo 11.
        """
        changed: list[int] = []
        try:
            with session_scope() as session:
                # A tabela piles tem chave estrangeira para cameras. Se a
                # câmera não estiver cadastrada (banco novo, linha removida),
                # toda gravação de contagem falharia com FOREIGN KEY.
                from app.database.repository import CameraRepository

                # ensure_from_env é idempotente e barato: roda toda vez que
                # há mudança justamente para o banco recém-criado não
                # derrubar a primeira contagem.
                CameraRepository(session).ensure_from_env(
                    settings.camera.camera_name, settings.camera.camera_rtsp_url
                )
                pile_repo = PileRepository(session)
                count_repo = CountRepository(session)
                for st in piles:
                    rt = self._piles.get(st.pile_id)
                    # get_or_create liga o tracking_id à linha da pilha; touch
                    # atualiza last_seen/active/stable_count. Estas duas linhas
                    # rodam sempre, mesmo sem mudança, para a lista de pilhas
                    # do banco continuar "viva".
                    db_pile = pile_repo.get_or_create(self.camera_id, st.tracking_id)
                    # `stable_count=None` preserva o último número gravado em vez
                    # de sobrescrever com uma leitura não verificada. É o mesmo
                    # portão do histórico: uma leitura que o sistema não
                    # confia não vira o número corrente da pilha no banco
                    # (que é o que /api/piles devolve ao dashboard).
                    verified = self._counts_are_measurable() and (
                        st.manual or st.status in (CountStatus.STABLE, CountStatus.PARTIAL)
                    )
                    pile_repo.touch(
                        db_pile,
                        active=True,
                        stable_count=st.stable_count if verified else None,
                    )
                    if rt is None:
                        continue
                    if st.manual:
                        # correções manuais são gravadas em count_corrections
                        # (ver manual_count); duplicar em counts poluiria a série
                        continue
                    # Só entra no histórico o que o sistema considera medido.
                    #
                    # Duas condições, e as duas importam:
                    #  1. o status precisa ser de uma pilha em que o número
                    #     foi validado (STABLE/PARTIAL);
                    #  2. o modelo precisa ser treinado para este ambiente.
                    #
                    # A (2) é a que mais importa na prática. Com o modelo
                    # genérico pré-treinado, TODOS os estimadores erram na
                    # mesma direção - eles medem a textura da cadeira (ripas do
                    # encosto, nervuras das pernas), que é uma repetição real e
                    # forte na imagem. Como todos discordam por pouco, o
                    # "acordo entre estimadores" fica alto e a confiança passa
                    # de 0.55 com um número que está o dobro do real. Nenhum
                    # limiar de confiança separa esse caso, porque o erro está
                    # no insumo, não no cálculo.
                    #
                    # Sem este portão, leituras indevidas entravam no banco
                    # como se fossem verdade - e o histórico é o que o operador
                    # usa para conferir o turno, além de alimentar o
                    # suggest_offset. Ruído confiável demais vira dado ruim com
                    # muita convicção, que é a pior combinação possível.
                    if not self._counts_are_measurable():
                        continue
                    if st.status not in (CountStatus.STABLE, CountStatus.PARTIAL):
                        continue
                    if rt.last_recorded is None:
                        # Primeira vez desta pilha: registra sem "anterior".
                        # previous_count fica nulo de propósito, para a auditoria
                        # distinguir "surgiu em N" de "mudou de A para N".
                        count_repo.add(
                            camera_id=self.camera_id,
                            pile_id=db_pile.id,
                            chair_count=st.stable_count,
                            confidence=st.confidence,
                            detection_confidence=st.detection_confidence,
                            stability_confidence=st.stability_confidence,
                            status=st.status,
                            method=st.method,
                            total_chairs=total,
                            pile_count=pile_count,
                            previous_count=None,
                        )
                        rt.last_recorded = st.stable_count
                        changed.append(st.pile_id)
                    elif st.stable_count != rt.last_recorded:
                        # Só o stable_count compara. Oscilação de confiança ou
                        # de raw_count NÃO grava: a série histórica é de
                        # contagens, não de frames.
                        count_repo.add(
                            camera_id=self.camera_id,
                            pile_id=db_pile.id,
                            chair_count=st.stable_count,
                            confidence=st.confidence,
                            detection_confidence=st.detection_confidence,
                            stability_confidence=st.stability_confidence,
                            status=st.status,
                            method=st.method,
                            total_chairs=total,
                            pile_count=pile_count,
                            previous_count=rt.last_recorded,
                        )
                        rt.last_recorded = st.stable_count
                        changed.append(st.pile_id)

                # pilhas inativas
                # Marcadas inativas (active=False) em vez de apagadas: o
                # histórico de contagens precisa continuar referenciando a
                # pilha. `self._piles` já não tem as que sumiram, então aqui
                # sobra quem não veio neste frame mas ainda existe em runtime.
                for pid in self._piles:
                    if pid not in {s.pile_id for s in piles}:
                        db_pile = pile_repo.get_by_tracking(self.camera_id, self._piles[pid].tracking_id)
                        if db_pile is not None:
                            pile_repo.touch(db_pile, active=False)
        except Exception as exc:
            # Falha de banco NÃO pode derrubar a contagem: o estado continua em
            # memória e no overlay. O texto fica guardado para o status().
            log.error("Falha ao gravar contagens (as contagens continuam em memória): %s", exc)
            self._persist_error = str(exc)
        return changed

    def _handle_snapshots(
        self,
        frame: np.ndarray,
        piles: list[PileState],
        total: int,
        changed: list[int],
        appeared: list[int],
        disappeared: list[int],
    ) -> None:
        """Dispara snapshots por evento. Cada gatilho tem sua flag no .env."""
        if not settings.snapshots.save_snapshots:
            return
        conf = self.store.overall_confidence()
        kwargs = dict(total=total, pile_count=len(piles), confidence=conf)
        if changed and settings.snapshots.on_count_change:
            self.snapshots.save(frame, reason="count_change", **kwargs)
        if appeared and settings.snapshots.on_pile_appear:
            self.snapshots.save(frame, reason="pile_appear", **kwargs)
        if disappeared and settings.snapshots.on_pile_disappear:
            self.snapshots.save(frame, reason="pile_disappear", **kwargs)
        # Dispara só na ENTRADA em baixa confiança (diferença de conjuntos),
        # não enquanto a pilha segue ruim: sem isso sairia 1 JPEG por frame.
        low_now = {p.pile_id for p in piles if p.status in (CountStatus.LOW_CONFIDENCE, CountStatus.UNKNOWN)}
        if (low_now - self._last_low_conf) and settings.snapshots.on_low_confidence:
            self.snapshots.save(frame, reason="low_confidence", **kwargs)
        # Estado guardado mesmo sem snapshot, senão o próximo frame registraria
        # a mesma transição de novo.
        self._last_low_conf = low_now
        # Cronometrado dentro do SnapshotService, com lock próprio: o
        # intervalo é independente dos eventos acima.
        self.snapshots.maybe_interval(frame, **kwargs)

    # --------------------------------------------------------------- alertas
    def _evaluate_alerts(self, total: int, piles: list[PileState]) -> list[dict[str, Any]]:
        """Regras de alerta (mostradas no dashboard; prontas p/ Telegram/e-mail).

        Convenção do .env: **0 desliga**. Por isso todo teste é `if a.total_min`
        e não `if a.total_min > 0` explícito. Um alerta não trava nada aqui;
        é só um objeto para o frontend pintar.
        """
        alerts: list[dict[str, Any]] = []
        a = settings.alerts
        if not a.enabled:
            return alerts
        if a.total_min and total < a.total_min:
            alerts.append({"type": "total_min", "level": "warning", "message": f"Total {total} abaixo do mínimo ({a.total_min})"})
        if a.total_max and total > a.total_max:
            alerts.append({"type": "total_max", "level": "warning", "message": f"Total {total} acima do máximo ({a.total_max})"})
        if a.pile_min:
            # Regra por pilha: alerta mesmo que o total geral esteja ok.
            for p in piles:
                if p.stable_count < a.pile_min:
                    alerts.append({"type": "pile_min", "level": "warning", "message": f"Pilha {p.pile_id} com apenas {p.stable_count} cadeiras"})
        low = [p for p in piles if p.confidence < a.low_confidence]
        if low:
            # Um alerta com a contagem, não um por pilha: 5 pilhas duvidosas
            # viram uma linha na tela, não cinco.
            alerts.append({"type": "low_confidence", "level": "info", "message": f"{len(low)} pilha(s) com confiança baixa"})
        runtime = self.cameras.get(self.camera_id)
        if runtime is not None and a.camera_offline and runtime.source.stats.state is not CameraState.ONLINE:
            # Único alerta com level "error": câmera fora é problema de
            # operação, não de contagem.
            alerts.append({"type": "camera_offline", "level": "error", "message": f"Câmera {runtime.source.stats.state.value}"})
        return alerts

    # ------------------------------------------------------------------ misc
    def _use_manual_mode(self) -> bool:
        """Hoje sempre ``False``.

        Ponto de decisão do projeto: sem modelo o sistema mostra "preparação"
        e não inventa contagem. Uma contagem inventada seria pior que nenhuma,
        porque o operador não teria como saber que ela era falsa. O método
        existe para deixar a decisão explícita e reversível.
        """
        return False  # sem modelo, o sistema informa "preparação" e não inventa contagem

    def _mode_label(self, detections_ok: bool) -> str:
        """Rótulo do modo de operação mostrado no cabeçalho do dashboard."""
        if not detections_ok:
            # "preparation" = modelo nem carregou; "degraded" = estava
            # funcionando e falhou agora. O operador age diferente nos dois.
            return "preparation" if not self.detector.is_loaded else "degraded"
        if not self.detector.trained_model:
            # Está contando, mas com o modelo genérico da Ultralytics. Os
            # números saem, porém não são confiáveis para a operação.
            return "test (modelo genérico)"
        return "production"

    @staticmethod
    def _agreement_from(est: Any) -> float:
        """Acordo entre os estimadores do StackCounter (0..1).

        Mede o quanto os métodos (período, picos, detecções, tamanho)
        discordam entre si: dispersão alta = os números foram "sorteados".
        Com um só estimador não há com o que comparar, então devolve 0.6
        (neutro) em vez de 0 ou 1, para não distorcer a confiança final.
        O divisor tem piso 2.0: uma dispersão de 2 cadeiras já zera o acordo,
        porque acima disso já é pilha diferente, não variação de método.
        """
        vals = list(est.candidates.values())
        if len(vals) <= 1:
            return 0.6
        v = np.array(vals, dtype=float)
        spread = float(v.std())
        return float(np.clip(1.0 - spread / max(2.0, 0.3 * max(1.0, float(v.mean()))), 0.0, 1.0))

    def manual_count(self, pile_id: int, value: int) -> dict[str, Any]:
        """Correção manual do operador. Vira registro de ``count_corrections``.

        Roda na thread da API. O lock protege a parte que mexe no estado
        compartilhado com o worker; a gravação no banco e o snapshot ficam
        fora do lock para não segurar o _process_frame por I/O.
        """
        with self._lock:
            st = self.store.get_pile(pile_id)
            if st is None:
                return {"ok": False, "message": f"Pilha {pile_id} não encontrada."}
            # ai_count/ai_confidence são salvos ANTES de sobrescrever, para
            # registrar quanto a IA errou. É esse par que alimenta a
            # sugestão de offset e as estatísticas de erro do modelo.
            ai_count = st.stable_count
            ai_conf = st.confidence
            # force() trava a janela de estabilidade no valor do operador e
            # marca como manual: a IA volta a ler, mas o número não muda.
            self.stability.force(pile_id, int(value))
            st.stable_count = int(value)
            st.raw_count = int(value)
            st.manual = True
            # Confiança 1.0: é o operador quem KNOWS. Sem isso a pilha cairia
            # para LOW_CONFIDENCE e ficaria fora do total.
            st.confidence = 1.0
            st.status = CountStatus.STABLE
            rt = self._piles.get(pile_id)
            if rt is not None:
                rt.manual = True
                # last_recorded = valor: a correção não deve gerar um novo
                # registro em `counts` no próximo frame.
                rt.last_recorded = int(value)
            # get_piles/set_piles para forçar a redescrição do dict de pilhas:
            # o store guarda referências diretas, então mutar o PileState não
            # avisa ninguém sem reindexar.
            self.store.set_piles(self.store.get_piles())
        try:
            with session_scope() as session:
                from app.database.repository import CorrectionRepository

                # pile_id=None: a linha de correção é sobre a contagem, não
                # sobre a pilha - o vínculo é feito só pela câmera. Mantido
                # assim porque a tabela foi criada sem FK para piles.
                CorrectionRepository(session).add(
                    camera_id=self.camera_id,
                    pile_id=None,
                    ai_count=int(ai_count),
                    correct_count=int(value),
                    ai_confidence=float(ai_conf),
                    author="operador",
                    note="Correção manual pelo dashboard",
                )
                # No log de eventos aparece quem mexeu e no que: é o rastro
                # de auditoria que o cliente consulta.
                EventRepository(session).log(
                    "INFO", "count_corrected",
                    f"Pilha {pile_id}: IA={ai_count} corrigido para {value}",
                )
        except Exception:
            # O estado em memória já foi corrigido; perder só o registro no
            # banco não pode fazer a API responder erro ao operador.
            log.exception("Falha ao registrar correção manual")
        # Snapshot com motivo "correction": a foto que prova qual imagem
        # gerou a contagem que o operador apontou como errada.
        self.snapshots.save(self.store.get_frame_bgr(), reason="correction", total=self.store.total())
        return {"ok": True, "pile_id": pile_id, "ai_count": int(ai_count), "correct_count": int(value)}

    def status(self) -> dict[str, Any]:
        """Retrato do worker para o endpoint de diagnóstico.

        Devolve persist_error separado: o operador precisa saber que o número
        que está vendo ao vivo não está indo para o histórico.
        """
        return {
            "camera_id": self.camera_id,
            "frames_processed": self.frames_processed,
            "process_fps": round(self.last_process_fps, 2),
            "inference_ms": round(self.last_infer_ms, 1),
            "counting_ms": round(self.last_count_ms, 1),
            "roi": self.roi,
            "piles_tracked": len(self._piles),
            "calibration": self._calibration,
            "persist_error": self._persist_error,
            "running": self._thread is not None and self._thread.is_alive(),
        }


# ----------------------------------------------------------------- texto
# ARQUIVO / MAPA (fim do pipeline) daqui para baixo é só desenho de texto:
#   _load_font -> carrega a TrueType uma vez por tamanho e memoiza
#   _ascii     -> tira acento quando só existe a Hershey
#   _text_size -> mede o texto na MESMA fonte que será desenhada
#   _draw_block-> fundo único para várias linhas
#   _draw_text -> escreve com acento, convertendo só a região do texto
#
# As fontes embutidas do OpenCV (Hershey) só sabem ASCII: "CONFIANÇA" viraria
# "CONFIAN?A". Para o overlay ficar legível em português usamos uma fonte
# TrueType (Pillow já vem com o Ultralytics). Se não houver fonte no sistema,
# caímos para o Hershey removendo os acentos - pior, mas legível.
_FONT_CANDIDATES = (
    # Ordem importa: primeiro o que tem acento em Linux; os caminhos do macOS
    # e do Windows garantem que o overlay continue legível em outra máquina.
    # A Bold vem antes porque o rótulo de "N cadeiras" precisa aparecer sobre
    # a imagem da câmera, que é ruidosa.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
)
# Cache por TAMANHO de fonte, não por arquivo: os mesmos 3 tamanhos (rótulo da
# pilha, barra de status, erro) se repetem em todas as frames. Sem isso, abrir
# o .ttf a cada rótulo custaria tempo de I/O dentro do laço de inferência.
# O valor None é cacheado de propósito: "não achou fonte" é resposta estável.
_font_cache: dict[int, Any] = {}


def _load_font(size: int):
    """Carrega (e memoriza) a fonte TrueText. ``None`` se não houver."""
    if size in _font_cache:
        return _font_cache[size]
    font = None
    try:
        from PIL import ImageFont

        # Primeira fonte existente na lista ganha: não há ranking por qualidade,
        # só "tem TrueType no sistema?". Import do PIL dentro da função para
        # não tornar o overlay uma dependência obrigatória do pacote.
        for path in _FONT_CANDIDATES:
            if Path(path).is_file():
                font = ImageFont.truetype(path, size)
                break
    except Exception as exc:  # pragma: no cover
        log.debug("Pillow/truetype indisponível: %s", exc)
    _font_cache[size] = font
    return font


def _ascii(text: str) -> str:
    """Remove acentos para as fontes que não os suportam.

    NFKD decompõe "Ç" em "C" + acento combinante; descartar as marcas
    combinantes deixa a letra base. Não é transliteração completa: "ç" vira "c",
    mas emoji e outros caracteres não-ASCII simplesmente somem.
    """
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _text_size(text: str, scale: float) -> tuple[int, int]:
    """Mede ``(largura, altura)`` do texto com a fonte que será usada.

    Importante: a medida tem que sair da MESMA fonte do desenho. Medir com a
    Hershey e desenhar com a TrueType (ou vice-versa) faz o bloco de fundo
    ficar curto ou largo demais. Por isso as duas funções repetem a fórmula
    de ``size = max(11, round(22 * scale))``.
    """
    # 22 px como base do scale 1.0 e piso de 11 px: abaixo disso a Hershey
    # fica ilegível no stream comprimido.
    size = max(11, int(round(22 * scale)))
    font = _load_font(size)
    if font is None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        # A Hershey devolve a altura da caixa; somar `base` (descida) aproxima
        # o total para poder centralizar o bloco.
        return tw, th + base
    from PIL import Image, ImageDraw

    # textbbox em vez de textlength: também pega a altura e a subida da
    # linha, necessárias para posicionar o texto na imagem.
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_block(
    img: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.55,
    line_gap: int = 4,
) -> None:
    """Desenha um bloco de várias linhas com **um** fundo só.

    Desenhar cada linha com fundo próprio fazia os retângulos se cobrirem
    quando o espaçamento era menor que a altura da fonte. Aqui o fundo é
    calculado uma vez, para o bloco inteiro.
    """
    if not lines:
        return
    # line_h é a MAIOR altura das linhas: se usarmos a menor, as linhas de
    # fonte maior do mesmo bloco se sobrepõem.
    sizes = [_text_size(line, scale) for line in lines]
    line_h = max(h for _w, h in sizes) if sizes else 12
    pad = 4
    block_h = line_h * len(lines) + line_gap * (len(lines) - 1) + 2 * pad
    block_w = max(w for w, _h in sizes) + 2 * pad

    h_img, w_img = img.shape[:2]
    # Empurra o bloco para dentro da imagem se ele estiver perto da borda.
    y = max(0, min(y, h_img - block_h))
    x = max(0, min(x, w_img - block_w))

    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        # y + pad + line_h: a coordenada do Pillow é o TOPO da linha, não a
        # linha de base como no cv2.putText. O +line_h compensa essa diferença.
        line_y = y + pad + line_h + i * (line_h + line_gap)
        # background=False: o fundo já foi desenhado acima, para o bloco todo.
        _draw_text(img, line, x + pad, line_y, color, scale=scale, background=False)


def _draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.55,
    background: bool = True,
) -> None:
    """Escreve texto, com acento quando possível.

    ``y`` é a linha de base do texto (mesma convenção do ``cv2.putText``).

    Só a região do texto é convertida para RGB (e de volta). Converter o frame
    inteiro a cada rótulo custava ~6 FPS com 3 pilhas.
    """
    # Mesma fórmula de _text_size: a medida e o desenho precisam concordar.
    size = max(11, int(round(22 * scale)))
    font = _load_font(size)
    if font is None:
        # Sem TrueType: Hershey + _ascii. O +1 em x é um deslocamento de 1px que
        # faz a Hershey parecer alinhada com a convenção de linha de base.
        cv2.putText(img, _ascii(text), (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)
        return
    try:
        from PIL import Image, ImageDraw

        h, w = img.shape[:2]
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = probe.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        # Retângulo da região do texto, já recortado à imagem. Os clamps
        # impedem que um rótulo na borda gere índice negativo.
        x0 = max(0, x - pad)
        y0 = max(0, y - th - 2 * pad)
        x1 = min(w, x + tw + 3 * pad)
        y1 = min(h, y + pad + 6)
        if x1 <= x0 or y1 <= y0:
            # Texto inteiro fora da imagem: nada a desenhar (não é erro).
            return
        # A conversão BGR<->RGB é o ponto caro do overlay. Feita só no recorte,
        # e não no frame inteiro, porque o ganho foi de ~6 FPS com 3 pilhas.
        roi = img[y0:y1, x0:x1]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        if background:
            draw.rectangle([0, 0, pil.size[0] - 1, pil.size[1] - 1], fill=(0, 0, 0))
        # Deslocamento duplo (x-x0+pad-bbox[0]): primeiro joga o texto dentro
        # do recorte, depois compensa a origem do textbbox, que tem folga
        # própria da fonte. Sem o -bbox[0]/-bbox[1] o texto sai torto.
        # fill invertido (BGR -> RGB) porque a cor chega no padrão OpenCV.
        draw.text((x - x0 + pad - bbox[0], y - y0 - pad - bbox[1]), text, font=font,
                  fill=(color[2], color[1], color[0]))
        img[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pragma: no cover
        # Um rótulo que falhou não pode derrubar a imagem inteira: cai para a
        # Hershey, que nunca falha.
        log.debug("Falha ao desenhar texto: %s", exc)
        cv2.putText(img, _ascii(text), (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------- utilidades
def _shift(det: Detection, dx: float, dy: float) -> Detection:
    """Translada uma detecção, mantendo os demais campos.

    Devolve um Detection NOVO em vez de mexer no original: as detecções do
    frame ainda podem ser lidas pelo tracker/detector mais abaixo no pipeline.
    """
    return Detection(
        x1=det.x1 + dx, y1=det.y1 + dy, x2=det.x2 + dx, y2=det.y2 + dy,
        conf=det.conf, class_id=det.class_id, class_name=det.class_name,
        track_id=det.track_id, mask=det.mask,
    )


def _dashed_rectangle(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Contorno tracejado, para marcar pilha em estado não confiável.

    Desenhado com cv2.line em vez de um "outline tracejado" do OpenCV (que não
    existe): 4 laços, um por lado. 12 px de tracejado com 12 px de vão - grande
    o bastante para se ler no stream JPEG, que borra linha fina.
    """
    x1, y1 = p1
    x2, y2 = p2
    dash = 12
    # min(x+dash, x2) evita desenhar além do canto nas bordas curtas.
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 2)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 2)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 2)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 2)


def datetime_from(ts: float):
    """Epoch -> ``datetime`` local, sem fuso.

    Inconsistente com o resto (que usa localnow() e UTC), mas é o que o
    PileState.first_seen espera. Fica assim para não mexer no schema.
    """
    from datetime import datetime

    return datetime.fromtimestamp(ts)


def runtime_state(service: "InferenceService") -> str:
    """Estado da câmera como string. "offline" se a câmera não existe mais."""
    runtime = service.cameras.get(service.camera_id)
    return runtime.source.stats.state.value if runtime else CameraState.OFFLINE.value


__all__ = ["InferenceService"]
