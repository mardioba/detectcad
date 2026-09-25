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
OVERLAY_WIDTH = 1280


@dataclass
class _PileRuntime:
    """Estado interno de uma pilha entre frames."""

    pile_id: int
    tracking_id: int
    bbox: tuple[float, float, float, float]
    first_seen_ts: float
    missed: int = 0
    last_recorded: int | None = None
    manual: bool = False
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

        self.tracker = PileTracker(iou_threshold=0.30, max_age=settings.counting.miss_grace_frames)
        self.stability = StabilityFilter()
        self.counter = StackCounter(
            method=settings.counting.counting_method,
            chair_height_px=settings.counting.chair_height_px,
        )
        self.pile_detector: PileDetector | None = None
        self.snapshots = SnapshotService(camera_id=camera_id, camera_name=settings.camera.camera_name)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._piles: dict[int, _PileRuntime] = {}
        self._pile_id_seq = 1
        self._lock = threading.RLock()
        self._last_snapshot_total: int | None = None
        self._last_low_conf: set[int] = set()
        self._persist_error: str = ""
        self._calibration: dict[str, Any] = {}

        # Métricas
        self.frames_processed = 0
        self.last_process_fps = 0.0
        self.last_infer_ms = 0.0
        self.last_count_ms = 0.0
        self.roi: tuple[int, int, int, int] | None = None
        self._fps_window: list[float] = []

    # ------------------------------------------------------------------ ciclo
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
            self._thread.start()
            log.info("Worker de inferência iniciado (camera_id=%s)", self.camera_id)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        log.info("Worker de inferência parado")

    def set_calibration(self, calibration: dict[str, Any]) -> None:
        """Aplica calibração vinda do banco/API sem reiniciar o worker."""
        with self._lock:
            self._calibration = dict(calibration or {})
            count_cfg = self._calibration.get("counting", {})
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
            if "confidence_threshold" in ai_cfg:
                settings.ai.confidence_threshold = float(ai_cfg["confidence_threshold"])
            if "chair_classes" in ai_cfg and ai_cfg["chair_classes"]:
                settings.ai.chair_classes = str(ai_cfg["chair_classes"])
            if "pile_classes" in ai_cfg and ai_cfg["pile_classes"]:
                settings.ai.pile_classes = str(ai_cfg["pile_classes"])
            if self.pile_detector is not None:
                self.pile_detector = build_pile_detector(self.detector.class_names)
        log.info("Calibração aplicada: %s", self._calibration)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        interval = 1.0 / max(0.1, settings.camera.process_fps)
        runtime = self.cameras.get(self.camera_id)
        if runtime is None:
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
                if self._stop.wait(0.1):
                    break
                continue

            frame, info = item
            try:
                self._process_frame(frame, info)
            except Exception as exc:
                log.exception("Erro ao processar frame %s", info.index)
                self.store.set_status(last_error=str(exc))

            elapsed = time.perf_counter() - loop_start
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
        t_start = time.perf_counter()
        h, w = frame.shape[:2]
        detections: list[Detection] = []
        detections_ok = False
        model_error = ""

        # --- 1. inferência -------------------------------------------------
        if self.detector.is_loaded:
            try:
                if self.pile_detector is None:
                    self.pile_detector = build_pile_detector(self.detector.class_names, (w, h))
                else:
                    self.pile_detector.configure((w, h))
                chairs, piles_cls = self.pile_detector.chair_classes, self.pile_detector.pile_classes
                tracker_mode, tracker_yaml = resolve_tracker_config(settings.ai.tracker)
                if tracker_mode == "ultralytics" and piles_cls:
                    dets, infer_ms = self.detector.track(
                        frame, tracker=tracker_yaml or "bytetrack.yaml"
                    )
                else:
                    dets, infer_ms = self.detector.predict(frame, classes=chairs | piles_cls or None)
                detections = dets
                detections_ok = True
                self.last_infer_ms = infer_ms
            except ModelNotAvailable:
                model_error = "Modelo não carregado"
            except Exception as exc:
                model_error = str(exc)
                log.exception("Falha na inferência")
        else:
            model_error = self.detector.load_error or (
                "Modelo específico ainda não treinado."
                if not self.detector.is_pretrained_generic()
                else "Modelo genérico pré-treinado (não recomendado para contagem)."
            )

        # --- 2. ROI ---------------------------------------------------------
        frame_proc, offset = self._apply_roi(frame)
        if offset != (0, 0) and detections_ok:
            detections = [_shift(d, -offset[0], -offset[1]) for d in detections]

        # --- 3. pilhas ------------------------------------------------------
        candidates: list = []
        if detections_ok and self.pile_detector is not None:
            candidates = self.pile_detector.detect(detections)
        elif not detections_ok and self._use_manual_mode():
            # Modo sem modelo: nada é inventado; o dashboard mostra vazio.
            candidates = []

        # --- 4. tracking ----------------------------------------------------
        tracked = self.tracker.update(candidates) if candidates else []

        # --- 5. contagem por pilha -----------------------------------------
        t_count = time.perf_counter()
        pile_states: list[PileState] = []
        seen_ids: set[int] = set()

        for tracking_id, cand, track in tracked:
            pile_id = self._resolve_pile_id(tracking_id, cand)
            seen_ids.add(pile_id)
            roi = self._crop(frame_proc, cand.bbox)
            est = self.counter.count(
                roi,
                chair_detections=cand.chair_detections,
                pile=cand,
                previous_count=self._piles[pile_id].last_recorded,
            )
            stab = self.stability.update(pile_id, est.count)
            confidence = combine_confidence(
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
            runtime_pile.missed = 0
            runtime_pile.last_seen = time.time()

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

        self.last_count_ms = (time.perf_counter() - t_count) * 1000.0

        # --- 6. pilhas que sumiram ------------------------------------------
        appeared: list[int] = []
        disappeared: list[int] = []
        for pile_id, rt in list(self._piles.items()):
            if pile_id in seen_ids:
                continue
            rt.missed += 1
            if rt.missed > settings.counting.miss_grace_frames:
                self._piles.pop(pile_id, None)
                self.stability.remove(pile_id)
                disappeared.append(pile_id)

        # --- 7. totais e status ---------------------------------------------
        self.store.set_piles(pile_states)
        total = self.store.total(only_stable=True)
        breakdown = self.store.totals_breakdown()
        # Pilha "nova" = vista há menos de 3 s.
        now = localnow()
        appeared = [s.pile_id for s in pile_states if (now - s.first_seen).total_seconds() < 3.0]

        # --- 8. overlay ------------------------------------------------------
        annotated = self._draw_overlay(frame_proc, pile_states, model_error, offset)
        jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])[1].tobytes()
        self.store.set_frame(jpeg, annotated, time.time())

        # --- 9. mudanças -> banco + snapshot ---------------------------------
        changed = self._persist_changes(pile_states, total, breakdown["pile_count"])

        # --- 10. histórico ---------------------------------------------------
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
        self._fps_window.append(time.time())
        if len(self._fps_window) > 20:
            self._fps_window.pop(0)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self.last_process_fps = (len(self._fps_window) - 1) / span

        alerts = self._evaluate_alerts(total, pile_states)
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
        self._handle_snapshots(
            annotated, pile_states, total, changed, appeared, disappeared
        )

    # ------------------------------------------------------------------ ROI
    def _apply_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Recorta a ROI principal. Fora dela, nada é processado."""
        roi = self._effective_roi(frame.shape[1], frame.shape[0])
        if roi is None:
            self.roi = None
            return frame, (0, 0)
        x, y, w, h = roi
        if w <= 0 or h <= 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            self.roi = None
            return frame, (0, 0)
        crop = frame[y : y + h, x : x + w]
        self.roi = roi
        return np.ascontiguousarray(crop), (x, y)

    def _effective_roi(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
        """ROI vinda do banco; senão a do .env; senão a imagem inteira."""
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
                x, y, w, h = (int(v) for v in env.replace(" ", "").split(",")[:4])
                return (x, y, min(w, frame_w - x), min(h, frame_h - y))
            except ValueError:
                log.warning("CALIBRATION_ROI inválida: %r (use x,y,w,h)", env)
        return None

    def _crop(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1i = int(max(0, min(w - 1, x1)))
        y1i = int(max(0, min(h - 1, y1)))
        x2i = int(max(x1i + 1, min(w, x2)))
        y2i = int(max(y1i + 1, min(h, y2)))
        return frame[y1i:y2i, x1i:x2i]

    # ---------------------------------------------------------------- IDs
    def _resolve_pile_id(self, tracking_id: int, cand: Any) -> int:
        """ID estável de pilha: sobrevive a-loss do tracker (câmera fixa)."""
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
            score = ov / max(1e-6, cand.width)
            if score > best_score:
                best_pid, best_score = pid, score
        if best_pid is not None and best_score >= max(0.25, settings.counting.pile_overlap_min):
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
        out = frame.copy()
        scale_x = offset[0]
        scale_y = offset[1]
        for st in piles:
            x1, y1, x2, y2 = st.bbox
            x1 += scale_x
            x2 += scale_x
            y1 += scale_y
            y2 += scale_y
            color = status_color_bgr(st.status)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            if st.status is not CountStatus.STABLE:
                # tracejado para contagem instável / baixa confiança / parcial
                _dashed_rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color)
            count_text = (
                f"{st.stable_count} cadeiras" if not st.manual else f"{st.stable_count} cadeiras (manual)"
            )
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
                scale=0.5,
                line_gap=3,
            )
        # barra de status
        self._draw_status_bar(out, piles, model_error)
        return out

    def _draw_status_bar(self, out: np.ndarray, piles: list[PileState], model_error: str) -> None:
        h, w = out.shape[:2]
        bar_h = 30
        cv2.rectangle(out, (0, 0), (w, bar_h), (30, 30, 30), -1)
        total = self.store.total(only_stable=True)
        conf = self.store.overall_confidence()
        parts = [
            f"PILHAS: {len(piles)}",
            f"TOTAL: {total}",
            f"CONFIANÇA: {conf * 100:.0f}%",
            f"FPS: {self.last_process_fps:.1f}",
        ]
        if self.roi:
            parts.append(f"ROI: {self.roi[2]}x{self.roi[3]}")
        x = 8
        for part in parts:
            _draw_text(out, part, x, 21, (255, 255, 255), scale=0.52)
            tw, _th = _text_size(part, 0.52)
            x += tw + 26
        if model_error:
            cv2.rectangle(out, (0, h - 26), (w, h), (60, 40, 130), -1)
            _draw_text(out, f"! {model_error[:110]}", 8, h - 7, (255, 255, 255), scale=0.5)

    # -------------------------------------------------------- persistência
    def _persist_changes(
        self, piles: list[PileState], total: int, pile_count: int
    ) -> list[int]:
        """Grava no banco **somente** quando a contagem muda de fato.

        Regras:
        * uma pilha nova ou que reapareceu com valor diferente gera registro;
        * uma pilha cujo valor estabilizado mudou gera registro com o
          ``previous_count`` preenchido (auditoria antes/depois);
        * valores idênticos não geram nada - evita milhares de linhas.
        """
        changed: list[int] = []
        try:
            with session_scope() as session:
                # A tabela piles tem chave estrangeira para cameras. Se a
                # câmera não estiver cadastrada (banco novo, linha removida),
                # toda gravação de contagem falharia com FOREIGN KEY.
                from app.database.repository import CameraRepository

                CameraRepository(session).ensure_from_env(
                    settings.camera.camera_name, settings.camera.camera_rtsp_url
                )
                pile_repo = PileRepository(session)
                count_repo = CountRepository(session)
                for st in piles:
                    rt = self._piles.get(st.pile_id)
                    db_pile = pile_repo.get_or_create(self.camera_id, st.tracking_id)
                    pile_repo.touch(db_pile, active=True, stable_count=st.stable_count)
                    if rt is None:
                        continue
                    if st.manual:
                        continue  # correções manuais são gravadas em count_corrections
                    if rt.last_recorded is None:
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
                for pid in self._piles:
                    if pid not in {s.pile_id for s in piles}:
                        db_pile = pile_repo.get_by_tracking(self.camera_id, self._piles[pid].tracking_id)
                        if db_pile is not None:
                            pile_repo.touch(db_pile, active=False)
        except Exception as exc:
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
        low_now = {p.pile_id for p in piles if p.status in (CountStatus.LOW_CONFIDENCE, CountStatus.UNKNOWN)}
        if (low_now - self._last_low_conf) and settings.snapshots.on_low_confidence:
            self.snapshots.save(frame, reason="low_confidence", **kwargs)
        self._last_low_conf = low_now
        self.snapshots.maybe_interval(frame, **kwargs)

    # --------------------------------------------------------------- alertas
    def _evaluate_alerts(self, total: int, piles: list[PileState]) -> list[dict[str, Any]]:
        """Regras de alerta (mostradas no dashboard; prontas p/ Telegram/e-mail)."""
        alerts: list[dict[str, Any]] = []
        a = settings.alerts
        if not a.enabled:
            return alerts
        if a.total_min and total < a.total_min:
            alerts.append({"type": "total_min", "level": "warning", "message": f"Total {total} abaixo do mínimo ({a.total_min})"})
        if a.total_max and total > a.total_max:
            alerts.append({"type": "total_max", "level": "warning", "message": f"Total {total} acima do máximo ({a.total_max})"})
        if a.pile_min:
            for p in piles:
                if p.stable_count < a.pile_min:
                    alerts.append({"type": "pile_min", "level": "warning", "message": f"Pilha {p.pile_id} com apenas {p.stable_count} cadeiras"})
        low = [p for p in piles if p.confidence < a.low_confidence]
        if low:
            alerts.append({"type": "low_confidence", "level": "info", "message": f"{len(low)} pilha(s) com confiança baixa"})
        runtime = self.cameras.get(self.camera_id)
        if runtime is not None and a.camera_offline and runtime.source.stats.state is not CameraState.ONLINE:
            alerts.append({"type": "camera_offline", "level": "error", "message": f"Câmera {runtime.source.stats.state.value}"})
        return alerts

    # ------------------------------------------------------------------ misc
    def _use_manual_mode(self) -> bool:
        return False  # sem modelo, o sistema informa "preparação" e não inventa contagem

    def _mode_label(self, detections_ok: bool) -> str:
        if not detections_ok:
            return "preparation" if not self.detector.is_loaded else "degraded"
        if not self.detector.trained_model:
            return "test (modelo genérico)"
        return "production"

    @staticmethod
    def _agreement_from(est: Any) -> float:
        """Acordo entre os estimadores do StackCounter (0..1)."""
        vals = list(est.candidates.values())
        if len(vals) <= 1:
            return 0.6
        v = np.array(vals, dtype=float)
        spread = float(v.std())
        return float(np.clip(1.0 - spread / max(2.0, 0.3 * max(1.0, float(v.mean()))), 0.0, 1.0))

    def manual_count(self, pile_id: int, value: int) -> dict[str, Any]:
        """Correção manual do operador. Vira registro de ``count_corrections``."""
        with self._lock:
            st = self.store.get_pile(pile_id)
            if st is None:
                return {"ok": False, "message": f"Pilha {pile_id} não encontrada."}
            ai_count = st.stable_count
            ai_conf = st.confidence
            self.stability.force(pile_id, int(value))
            st.stable_count = int(value)
            st.raw_count = int(value)
            st.manual = True
            st.confidence = 1.0
            st.status = CountStatus.STABLE
            rt = self._piles.get(pile_id)
            if rt is not None:
                rt.manual = True
                rt.last_recorded = int(value)
            self.store.set_piles(self.store.get_piles())
        try:
            with session_scope() as session:
                from app.database.repository import CorrectionRepository

                CorrectionRepository(session).add(
                    camera_id=self.camera_id,
                    pile_id=None,
                    ai_count=int(ai_count),
                    correct_count=int(value),
                    ai_confidence=float(ai_conf),
                    author="operador",
                    note="Correção manual pelo dashboard",
                )
                EventRepository(session).log(
                    "INFO", "count_corrected",
                    f"Pilha {pile_id}: IA={ai_count} corrigido para {value}",
                )
        except Exception:
            log.exception("Falha ao registrar correção manual")
        self.snapshots.save(self.store.get_frame_bgr(), reason="correction", total=self.store.total())
        return {"ok": True, "pile_id": pile_id, "ai_count": int(ai_count), "correct_count": int(value)}

    def status(self) -> dict[str, Any]:
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
# As fontes embutidas do OpenCV (Hershey) só sabem ASCII: "CONFIANÇA" viraria
# "CONFIAN?A". Para o overlay ficar legível em português usamos uma fonte
# TrueType (Pillow já vem com o Ultralytics). Se não houver fonte no sistema,
# caímos para o Hershey removendo os acentos - pior, mas legível.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
)
_font_cache: dict[int, Any] = {}


def _load_font(size: int):
    """Carrega (e memoriza) a fonte TrueText. ``None`` se não houver."""
    if size in _font_cache:
        return _font_cache[size]
    font = None
    try:
        from PIL import ImageFont

        for path in _FONT_CANDIDATES:
            if Path(path).is_file():
                font = ImageFont.truetype(path, size)
                break
    except Exception as exc:  # pragma: no cover
        log.debug("Pillow/truetype indisponível: %s", exc)
    _font_cache[size] = font
    return font


def _ascii(text: str) -> str:
    """Remove acentos para as fontes que não os suportam."""
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _text_size(text: str, scale: float) -> tuple[int, int]:
    """Mede ``(largura, altura)`` do texto com a fonte que será usada."""
    size = max(11, int(round(22 * scale)))
    font = _load_font(size)
    if font is None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        return tw, th + base
    from PIL import Image, ImageDraw

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
        line_y = y + pad + line_h + i * (line_h + line_gap)
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
    size = max(11, int(round(22 * scale)))
    font = _load_font(size)
    if font is None:
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
        x0 = max(0, x - pad)
        y0 = max(0, y - th - 2 * pad)
        x1 = min(w, x + tw + 3 * pad)
        y1 = min(h, y + pad + 6)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        if background:
            draw.rectangle([0, 0, pil.size[0] - 1, pil.size[1] - 1], fill=(0, 0, 0))
        draw.text((x - x0 + pad - bbox[0], y - y0 - pad - bbox[1]), text, font=font,
                  fill=(color[2], color[1], color[0]))
        img[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pragma: no cover
        log.debug("Falha ao desenhar texto: %s", exc)
        cv2.putText(img, _ascii(text), (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------- utilidades
def _shift(det: Detection, dx: float, dy: float) -> Detection:
    return Detection(
        x1=det.x1 + dx, y1=det.y1 + dy, x2=det.x2 + dx, y2=det.y2 + dy,
        conf=det.conf, class_id=det.class_id, class_name=det.class_name,
        track_id=det.track_id, mask=det.mask,
    )


def _dashed_rectangle(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dash = 12
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 2)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 2)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 2)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 2)


def datetime_from(ts: float):
    from datetime import datetime

    return datetime.fromtimestamp(ts)


def runtime_state(service: "InferenceService") -> str:
    runtime = service.cameras.get(service.camera_id)
    return runtime.source.stats.state.value if runtime else CameraState.OFFLINE.value


__all__ = ["InferenceService"]
