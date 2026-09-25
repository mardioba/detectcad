"""Serviço de treinamento de modelos YOLO.

O treinamento roda em **processo separado** (``subprocess``) para que o
dashboard continue respondendo normalmente - requisito explícito do projeto.
O progresso é lido de ``results.csv``, que o Ultralytics atualiza a cada
época.

Fluxo:

    POST /api/training/start -> sobe o processo
    GET  /api/training/status -> lê o progresso do run
    (ao terminar) -> registra o best.pt em model_versions
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.state_store import state_store

log = logging.getLogger("training")


@dataclass
class TrainingJob:
    """Estado de um treinamento em andamento."""

    id: str
    name: str
    config: dict[str, Any]
    started_at: float
    process: subprocess.Popen | None = None
    run_dir: Path | None = None
    finished: bool = False
    returncode: int | None = None
    error: str = ""
    log_path: Path | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    epochs_done: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "started_at": self.started_at,
            "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "elapsed_sec": round(time.time() - self.started_at, 1),
            "running": bool(self.process and self.process.poll() is None),
            "finished": self.finished,
            "returncode": self.returncode,
            "error": self.error,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "progress": self.progress,
            "epochs_done": self.epochs_done,
            "epochs_total": self.config.get("epochs"),
        }


class TrainingService:
    """Gerencia (no máximo) um treinamento por vez."""

    def __init__(self) -> None:
        self._job: TrainingJob | None = None
        self._lock = threading.RLock()
        self._history: list[dict[str, Any]] = []
        self._poll_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ info
    def gpu_report(self) -> dict[str, Any]:
        """Mostra GPU/VRAM reais antes de começar (pedido do projeto)."""
        from app.ai.yolo_detector import YoloDetector

        info = YoloDetector.gpu_info()
        if info.get("cuda_available"):
            lines = ["GPU detectada:"]
            for gpu in info["gpus"]:
                lines.append(
                    f"  {gpu['name']}  VRAM {gpu['vram_total_gb']:.1f} GB "
                    f"(em uso {gpu['vram_used_gb']:.1f} GB)"
                )
            lines.append(f"Device: cuda:0")
            lines.append(f"Torch: {info.get('torch_version', '?')}")
            return {"cuda_available": True, "text": "\n".join(lines), **info}
        return {
            "cuda_available": False,
            "text": (
                "GPU não detectada (CUDA indisponível).\n"
                "O treinamento vai rodar na CPU - funciona, mas bem mais devagar.\n"
                "Device: cpu"
            ),
            **info,
        }

    def resolve_device(self, requested: str = "") -> str:
        dev = (requested or settings.training.device or "").strip().lower()
        if dev:
            if dev == "auto":
                return settings.resolve_device()
            if dev == "cuda":
                return "cuda:0"
            return dev
        return settings.resolve_device()

    # ------------------------------------------------------------------ start
    def start(
        self,
        *,
        model: str | None = None,
        epochs: int | None = None,
        image_size: int | None = None,
        batch: int | None = None,
        device: str | None = None,
        workers: int | None = None,
        patience: int | None = None,
        seed: int | None = None,
        name: str | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """Inicia o treinamento em processo separado."""
        with self._lock:
            if self._job and self._job.process and self._job.process.poll() is None:
                return {
                    "ok": False,
                    "message": (
                        f"Já existe um treinamento em andamento ({self._job.name}, "
                        f"epoch {self._job.epochs_done}/{self._job.config.get('epochs')})."
                    ),
                }

            cfg = {
                "model": model or settings.training.model,
                "epochs": int(epochs or settings.training.epochs),
                "image_size": int(image_size or settings.training.image_size),
                "batch": int(batch if batch is not None else settings.training.batch),
                "device": self.resolve_device(device or ""),
                "workers": int(workers if workers is not None else settings.training.workers),
                "patience": int(patience if patience is not None else settings.training.patience),
                "seed": int(seed if seed is not None else settings.training.seed),
                "name": name or f"{settings.training.name}_{time.strftime('%Y%m%d_%H%M%S')}",
            }
            data_yaml = data or str(settings.base_dir / "training" / "configs" / "chairs.yaml")
            if not Path(data_yaml).is_file():
                return {"ok": False, "message": f"Dataset YAML não encontrado: {data_yaml}"}
            # Conta só imagens de verdade: a pasta pode conter .gitkeep e afins.
            from app.services.dataset_service import IMAGE_EXTS

            train_dir = settings.dataset_dir / "images" / "train"
            n_images = sum(
                1
                for p in train_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ) if train_dir.is_dir() else 0
            if n_images == 0:
                return {
                    "ok": False,
                    "message": (
                        f"Nenhuma imagem de treino em {train_dir}. Adicione imagens, "
                        "anote e execute a divisão do dataset antes de treinar "
                        "(aba Dataset → Dividir dataset)."
                    ),
                }

            job = TrainingJob(
                id=f"job_{int(time.time())}",
                name=cfg["name"],
                config=cfg,
                started_at=time.time(),
            )
            project = settings.base_dir / settings.training.project
            run_dir = project / cfg["name"]
            job.run_dir = run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            log_dir = settings.logs_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            job.log_path = log_dir / "training_latest.log"
            self._job = job

            cmd = self._build_command(cfg, data_yaml)
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            # Ultralytics/Ultralytics RTX usa a HOME do usuário; mantemos.
            try:
                handle = job.log_path.open("w", encoding="utf-8")
                job.process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(settings.base_dir),
                    env=env,
                )
            except Exception as exc:
                job.error = str(exc)
                job.finished = True
                return {"ok": False, "message": f"Não foi possível iniciar o treino: {exc}"}

            state_store.push_event(
                "INFO",
                "training_started",
                f"Treinamento '{cfg['name']}' iniciado ({cfg['epochs']} épocas, device={cfg['device']})",
            )
            self._start_poller()
            return {
                "ok": True,
                "job": job.as_dict(),
                "command": " ".join(cmd),
                "message": (
                    f"Treinamento iniciado. Acompanhe o progresso na aba Treinamento "
                    f"ou em logs/training.log."
                ),
            }

    def _build_command(self, cfg: dict[str, Any], data_yaml: str) -> list[str]:
        """Monta o comando ``python -m training.scripts.train``."""
        return [
            sys.executable,
            "-m",
            "training.scripts.train",
            "--data", data_yaml,
            "--model", str(cfg["model"]),
            "--epochs", str(cfg["epochs"]),
            "--imgsz", str(cfg["image_size"]),
            "--batch", str(cfg["batch"]),
            "--device", str(cfg["device"]),
            "--workers", str(cfg["workers"]),
            "--patience", str(cfg["patience"]),
            "--seed", str(cfg["seed"]),
            "--name", str(cfg["name"]),
        ]

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {
                    "running": False,
                    "message": "Nenhum treinamento iniciado nesta sessão.",
                    "history": self._history[-10:],
                    "gpu": self.gpu_report(),
                }
            self._refresh(job)
            return {
                "running": bool(job.process and job.process.poll() is None),
                "job": job.as_dict(),
                "history": self._history[-10:],
                "gpu": self.gpu_report(),
            }

    def _start_poller(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_thread = threading.Thread(target=self._poll_loop, name="training-poll", daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        while True:
            with self._lock:
                job = self._job
                alive = bool(job and job.process and job.process.poll() is None)
            if not alive:
                return
            time.sleep(3.0)

    def _refresh(self, job: TrainingJob) -> None:
        """Lê o progresso atual de ``results.csv``."""
        if job.run_dir is None:
            return
        csv_path = job.run_dir / "results.csv"
        if not csv_path.is_file():
            return
        try:
            import csv as _csv

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
        except Exception:
            return
        if not rows:
            return
        last = rows[-1]

        def pick(prefix: str) -> float | None:
            for key, val in last.items():
                if key and key.strip().lower().startswith(prefix):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
            return None

        job.epochs_done = len(rows)
        job.progress = {
            "epoch": len(rows),
            "total": job.config.get("epochs"),
            "percent": round(100.0 * len(rows) / max(1, job.config.get("epochs", 1)), 1),
            "loss": pick("train/box_loss") or pick("train/seg_loss") or pick("train/cls_loss"),
            "box_loss": pick("train/box_loss"),
            "cls_loss": pick("train/cls_loss"),
            "precision": pick("metrics/precision"),
            "recall": pick("metrics/recall"),
            "map50": pick("metrics/mAP50(B)") or pick("metrics/mAP50"),
            "map5095": pick("metrics/mAP50-95(B)") or pick("metrics/mAP50-95"),
        }

    # ------------------------------------------------------------------ stop
    def stop(self) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None or job.process is None or job.process.poll() is not None:
                return {"ok": False, "message": "Nenhum treinamento em andamento."}
            try:
                job.process.terminate()
                job.process.wait(timeout=10)
            except Exception:
                try:
                    job.process.kill()
                except Exception:
                    pass
            job.finished = True
            job.returncode = job.process.returncode
            state_store.push_event("WARNING", "training_stopped", f"Treinamento '{job.name}' interrompido.")
            return {"ok": True, "message": "Treinamento interrompido."}

    def _finalize(self, job: TrainingJob) -> None:
        """Registra o modelo treinado quando o processo termina com sucesso."""
        job.finished = True
        if job.process is not None:
            job.returncode = job.process.returncode
        if job.returncode not in (0, None) or job.run_dir is None:
            msg = f"Treinamento terminou com código {job.returncode}. Veja logs/training_latest.log."
            job.error = msg
            state_store.push_event("ERROR", "training_failed", msg)
            self._history.append({"name": job.name, "result": "erro", "when": time.time()})
            return

        # Registra o best.pt e as métricas reais
        try:
            from app.ai.model_manager import ModelManager
            from app.ai.yolo_detector import YoloDetector

            mm = ModelManager(YoloDetector())
            stats = 0
            from app.services.dataset_service import DatasetService

            stats = DatasetService().stats().annotated
            res = mm.register_trained(job.run_dir, make_active=False, dataset_size=stats)
            msg = res.get("message", "Modelo registrado.")
            state_store.push_event("INFO", "training_finished", msg)
            self._history.append(
                {
                    "name": job.name,
                    "result": "ok",
                    "when": time.time(),
                    "model": res.get("filename"),
                    "metrics": res.get("metrics"),
                }
            )
        except Exception as exc:
            log.exception("Falha ao registrar o modelo treinado")
            job.error = str(exc)
            self._history.append({"name": job.name, "result": "erro_registro", "when": time.time()})

    def check_finished(self) -> None:
        """Verifica se o processo terminou e finaliza o job."""
        with self._lock:
            job = self._job
            if job is None or job.finished or job.process is None:
                return
            if job.process.poll() is None:
                return
            self._finalize(job)

    def tail_log(self, lines: int = 60) -> str:
        with self._lock:
            job = self._job
        if job is None or job.log_path is None or not job.log_path.is_file():
            return ""
        try:
            content = job.log_path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(content.splitlines()[-lines:])
        except OSError:
            return ""


training_service = TrainingService()

__all__ = ["TrainingService", "TrainingJob", "training_service"]
