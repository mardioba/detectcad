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

# ARQUIVO / MAPA
# Treinamento YOLO. Tudo aqui existe para respeitar UMA exigência: o dashboard
# tem que continuar respondendo enquanto o modelo treina.
#
# Como isso é garantido: o treino é um `subprocess` separado, nunca uma
# chamada a YOLO() dentro do processo do FastAPI. O progresso é lido de um
# arquivo (results.csv), nunca de um pipe.
#
# Ordem de leitura:
#   1. TrainingJob       -> o estado de UM treino, serializável para a UI
#   2. gpu_report/resolve_device -> o que a máquina aguenta
#   3. start             -> valida, monta o comando, sobe o processo
#   4. _build_command    -> a linha de comando exata do treino
#   5. _refresh          -> lê o progresso do results.csv
#   6. stop / _finalize  -> encerrar e registrar o best.pt
#   7. check_finished/tail_log -> o que a API consulta

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
    """Estado de um treinamento em andamento.

    É o objeto que a API serializa. `process` e `run_dir` ficam nele porque o
   .job precisa responder "ainda está rodando?" sem consultar o SO de novo.
    """

    id: str
    name: str
    # Parâmetros efetivos do treino. Guardar o que foi realmente usado (e não
    # só o que o .env diz) é o que permite comparar dois experimentos depois.
    config: dict[str, Any]
    started_at: float
    process: subprocess.Popen | None = None
    run_dir: Path | None = None
    finished: bool = False
    # returncode != 0 do Ultralytics = treino morreu. Só o processo sabe.
    returncode: int | None = None
    error: str = ""
    log_path: Path | None = None
    # Último estado lido do results.csv (loss, mAP, epoch).
    progress: dict[str, Any] = field(default_factory=dict)
    # 1 linha do results.csv = 1 época. O contador vem do arquivo, não do log.
    epochs_done: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "started_at": self.started_at,
            "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            # Calculado na serialização, não guardado: se ninguém pedir, não
            # custa nada.
            "elapsed_sec": round(time.time() - self.started_at, 1),
            # poll() retorna None enquanto o processo vive. É a única forma
            # barata de saber se ainda está rodando, sem bloquear.
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
        # Um job só, não uma lista. Treinar dois modelos ao mesmo tempo na
        # mesma GPU daria OOM e os dois resultados seriam ruins.
        self._job: TrainingJob | None = None
        self._lock = threading.RLock()
        # Histórico só em memória: perde-se no restart, e tudo bem. O que
        # importa (o .pt e as métricas) foi para o ModelManager.
        self._history: list[dict[str, Any]] = []
        self._poll_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ info
    def gpu_report(self) -> dict[str, Any]:
        """Mostra GPU/VRAM reais antes de começar (pedido do projeto).

        Import tardio de YoloDetector: importar o Ultralytics carrega o torch e
        leva segundos. Fazer isso num GET simples de status seria lento demais.
        """
        from app.ai.yolo_detector import YoloDetector

        info = YoloDetector.gpu_info()
        if info.get("cuda_available"):
            lines = ["GPU detectada:"]
            for gpu in info["gpus"]:
                # VRAM em GB com 1 casa: o operador precisa saber se cabe o
                # batch, não o valor exato ao byte.
                lines.append(
                    f"  {gpu['name']}  VRAM {gpu['vram_total_gb']:.1f} GB "
                    f"(em uso {gpu['vram_used_gb']:.1f} GB)"
                )
            lines.append(f"Device: cuda:0")
            lines.append(f"Torch: {info.get('torch_version', '?')}")
            # O texto é pronto para tela; o resto são os dados crus, que o
            # backend pode usar para sugerir um batch.
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
        """Decide entre cpu e cuda, com "auto" = detecção.

        Precedência: o que veio da request > TRAINING.DEVICE do .env >
        autodetecção. "cuda" é normalizado para "cuda:0" porque é o formato
        que o Ultralytics espera.
        """
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
        """Inicia o treinamento em processo separado.

        Tudo que pode dar errado é validado ANTES de subir o processo: um
        treino que morre em 2 s por falta de imagens seria confuso, e o
        operador acharia que o problema é o modelo.
        """
        with self._lock:
            # Um treino por vez. A mensagem diz qual e em que época, para o
            # operador entender que o botão não está quebrado.
            if self._job and self._job.process and self._job.process.poll() is None:
                return {
                    "ok": False,
                    "message": (
                        f"Já existe um treinamento em andamento ({self._job.name}, "
                        f"epoch {self._job.epochs_done}/{self._job.config.get('epochs')})."
                    ),
                }

            # `or settings.X` para valores onde 0 não faz sentido (epochs,
            # image_size) e `is not None` onde 0 é legítimo (batch=0, seed=0).
            # Ler isso errado já mandou batch=-16 para o Ultralytics numa
            # versão anterior do projeto.
            cfg = {
                "model": model or settings.training.model,
                "epochs": int(epochs or settings.training.epochs),
                "image_size": int(image_size or settings.training.image_size),
                "batch": int(batch if batch is not None else settings.training.batch),
                "device": self.resolve_device(device or ""),
                "workers": int(workers if workers is not None else settings.training.workers),
                "patience": int(patience if patience is not None else settings.training.patience),
                "seed": int(seed if seed is not None else settings.training.seed),
                # Nome com timestamp: dois treinos no mesmo dia não colidem de
                # pasta em training/runs/.
                "name": name or f"{settings.training.name}_{time.strftime('%Y%m%d_%H%M%S')}",
            }
            data_yaml = data or str(settings.base_dir / "training" / "configs" / "chairs.yaml")
            if not Path(data_yaml).is_file():
                # Sem YAML não há como treinar. Falhar aqui é mais claro que o
                # erro do Ultralytics 2 minutos depois.
                return {"ok": False, "message": f"Dataset YAML não encontrado: {data_yaml}"}
            # Conta só imagens de verdade: a pasta pode conter .gitkeep e afins.
            from app.services.dataset_service import IMAGE_EXTS

            train_dir = settings.dataset_dir / "images" / "train"
            # Filtro por extensão em vez de len(listdir): um .gitkeep ou um
            # .DS_Store na pasta não pode fazer o operador acreditar que tem
            # imagens de treino.
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
                # id só com o segundo: dois start() no mesmo segundo não
                # acontecem porque o lock já barrou o segundo.
                id=f"job_{int(time.time())}",
                name=cfg["name"],
                config=cfg,
                started_at=time.time(),
            )
            # O run_dir é ONDE o Ultralytics vai escrever results.csv,
            # weights/best.pt e os gráficos. Criado antes: o processo filho
            # não espera por ele.
            project = settings.base_dir / settings.training.project
            run_dir = project / cfg["name"]
            job.run_dir = run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            log_dir = settings.logs_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            # Nome fixo ("latest"): tail_log() é a porta de entrada do
            # operador quando algo dá errado, e ele não sabe qual o número do
            # run. O histórico de stdout do treino anterior se perde de propósito.
            job.log_path = log_dir / "training_latest.log"
            self._job = job

            cmd = self._build_command(cfg, data_yaml)
            env = os.environ.copy()
            # Sem isso, o output do Ultralytics fica preso no buffer do
            # processo filho e o log em disco fica vazio durante o treino todo.
            env.setdefault("PYTHONUNBUFFERED", "1")
            # Ultralytics/Ultralytics RTX usa a HOME do usuário; mantemos.
            try:
                # handle fica aberto de propósito: é o stdout do filho. Fechá-lo
                # aqui derrubaria a escrita.
                handle = job.log_path.open("w", encoding="utf-8")
                job.process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=handle,
                    # stderr no mesmo arquivo: a causa da falha quase sempre
                    # vem no stderr, e um log só é mais fácil de ler no log.
                    stderr=subprocess.STDOUT,
                    cwd=str(settings.base_dir),
                    env=env,
                )
            except Exception as exc:
                # Exceção no Popen = python inexistente, permissão, etc. O job
                # fica marcado como finished para não bloquear o próximo start.
                job.error = str(exc)
                job.finished = True
                return {"ok": False, "message": f"Não foi possível iniciar o treino: {exc}"}

            # Evento global: aparece no painel de eventos e no WebSocket, sem
            # precisar que o navegador faça polling.
            state_store.push_event(
                "INFO",
                "training_started",
                f"Treinamento '{cfg['name']}' iniciado ({cfg['epochs']} épocas, device={cfg['device']})",
            )
            self._start_poller()
            return {
                "ok": True,
                "job": job.as_dict(),
                # A comando volta para a UI de propósito: o operador precisa
                # poder conferir/reproduzir exatamente o que rodou.
                "command": " ".join(cmd),
                "message": (
                    f"Treinamento iniciado. Acompanhe o progresso na aba Treinamento "
                    f"ou em logs/training.log."
                ),
            }

    def _build_command(self, cfg: dict[str, Any], data_yaml: str) -> list[str]:
        """Monta o comando ``python -m training.scripts.train``.

        Tudo em lista (nunca string com shell=True): nome de treino e caminho
        do YAML vêm do usuário, e uma string interpretada abriria espaço para
        injeção de comando.
        """
        return [
            # sys.executable e não "python": garante o mesmo interpretador do
            # servidor, com as mesmas dependências do venv.
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
        """Estado do treino + GPU. É o que o polling da aba Treinamento chama."""
        with self._lock:
            job = self._job
            if job is None:
                # Ainda nada iniciado nesta sessão (típico depois de restart).
                return {
                    "running": False,
                    "message": "Nenhum treinamento iniciado nesta sessão.",
                    "history": self._history[-10:],
                    "gpu": self.gpu_report(),
                }
            # _refresh antes de responder: o endpoint é quem empurra o
            # progresso para a tela, então é aqui que results.csv é lido.
            self._refresh(job)
            return {
                "running": bool(job.process and job.process.poll() is None),
                "job": job.as_dict(),
                "history": self._history[-10:],
                "gpu": self.gpu_report(),
            }

    def _start_poller(self) -> None:
        # Uma poller só. Duas fariam _finalize em duplicata e registrariam
        # o mesmo best.pt duas vezes.
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_thread = threading.Thread(target=self._poll_loop, name="training-poll", daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        """Acorda a cada 3 s só para checar se o processo morreu.

        Não faz mais nada: o trabalho pesado (_refresh) fica por conta do
        endpoint de status. A thread existe para que o _finalize aconteça
        mesmo com ninguém olhando a tela.
        """
        while True:
            with self._lock:
                job = self._job
                alive = bool(job and job.process and job.process.poll() is None)
            if not alive:
                # Sai e deixa check_finished() (chamado pelo status) fazer o
                # _finalize, que é a parte que escreve no banco.
                return
            time.sleep(3.0)

    def _refresh(self, job: TrainingJob) -> None:
        """Lê o progresso atual de ``results.csv``.

        results.csv é a única fonte de progresso confiável: o stdout do
        Ultralytics é livre e muda de formato entre versões. Uma linha por
        época, reescrita do zero a cada época.
        """
        if job.run_dir is None:
            return
        csv_path = job.run_dir / "results.csv"
        if not csv_path.is_file():
            # Ainda não existe: a primeira época não terminou. A UI mostra
            # "iniciando" em vez de erro.
            return
        try:
            import csv as _csv

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
        except Exception:
            # O Ultralytics pode estar reescrevendo o arquivo neste instante.
            # Silêncio é melhor que erro: o próximo request tenta de novo.
            return
        if not rows:
            return
        # Última linha = última época concluída.
        last = rows[-1]

        def pick(prefix: str) -> float | None:
            """Primeira coluna cujo nome COMEÇA com ``prefixo``, como float.

            Feito por prefixo e não por nome exato porque o Ultralytics muda
            os cabeçalhos entre versões ("train/box_loss" vs "train/box_loss"
            vs "metrics/mAP50(B)") e quebrar a cada upgrade seria pior que
            aceitar uma coluna parecida.
            """
            for key, val in last.items():
                if key and key.strip().lower().startswith(prefix):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
            return None

        # Contagem de linhas, não o campo "epoch": o CSV sempre tem uma linha
        # por época, mesmo se o cabeçalho mudar.
        job.epochs_done = len(rows)
        job.progress = {
            "epoch": len(rows),
            "total": job.config.get("epochs"),
            # max(1, ...) evita divisão por zero se epochs vier vazio do .env.
            "percent": round(100.0 * len(rows) / max(1, job.config.get("epochs", 1)), 1),
            # Loss de segmentação como fallback do de detecção: um modelo
            # segmentador não escreve train/box_loss.
            "loss": pick("train/box_loss") or pick("train/seg_loss") or pick("train/cls_loss"),
            "box_loss": pick("train/box_loss"),
            "cls_loss": pick("train/cls_loss"),
            "precision": pick("metrics/precision"),
            "recall": pick("metrics/recall"),
            # mAP50 primeiro, depois o nome sem o sufixo: o (B) muda entre
            # versões.
            "map50": pick("metrics/mAP50(B)") or pick("metrics/mAP50"),
            # mAP50-95 é a métrica que realmente indica generalização: um
            # modelo com mAP50 alto e mAP50-95 baixo decorou a cena.
            "map5095": pick("metrics/mAP50-95(B)") or pick("metrics/mAP50-95"),
        }

    # ------------------------------------------------------------------ stop
    def stop(self) -> dict[str, Any]:
        """Interrompe o treino. Tenta terminar, depois mata."""
        with self._lock:
            job = self._job
            if job is None or job.process is None or job.process.poll() is not None:
                return {"ok": False, "message": "Nenhum treinamento em andamento."}
            try:
                # terminate() = SIGTERM: o Ultralytics tem chance de salvar o
                # estado e fechar os arquivos.
                job.process.terminate()
                job.process.wait(timeout=10)
            except Exception:
                # TimeoutExpired (ou processo travado): kill() = SIGKILL.
                # best.pt parcial se perde, mas a GPU é liberada.
                try:
                    job.process.kill()
                except Exception:
                    pass
            job.finished = True
            # returncode negativo = morto por sinal. Guardar para o operador
            # ver que não foi um erro do script.
            job.returncode = job.process.returncode
            state_store.push_event("WARNING", "training_stopped", f"Treinamento '{job.name}' interrompido.")
            return {"ok": True, "message": "Treinamento interrompido."}

    def _finalize(self, job: TrainingJob) -> None:
        """Registra o modelo treinado quando o processo termina com sucesso."""
        job.finished = True
        if job.process is not None:
            job.returncode = job.process.returncode
        # returncode None = nunca rodou de fato. Qualquer valor diferente de 0
        # (incluindo sinais) é falha.
        if job.returncode not in (0, None) or job.run_dir is None:
            # A causa está no stdout, que foi para training_latest.log. A
            # mensagem aponta para lá em vez de tentar resumir a traceback.
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

            # dataset_size = nº de imagens ANOTADAS, não o total de arquivos:
            # é o que diz se o modelo foi treinado com material suficiente.
            stats = DatasetService().stats().annotated
            # make_active=False: o modelo novo entra na lista de versões, mas
            # só vira o modelo em uso quando alguém validar as métricas e
            # promover na tela. Trocar no automático poderia piorar a
            # contagem sem aviso.
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
            # O treino DEU CERTO, só a bookkeeping falhou. O .pt está em disco
            # e pode ser registrado à mão; o operador precisa saber disso.
            log.exception("Falha ao registrar o modelo treinado")
            job.error = str(exc)
            self._history.append({"name": job.name, "result": "erro_registro", "when": time.time()})

    def check_finished(self) -> None:
        """Verifica se o processo terminou e finaliza o job.

        Idempotente: o guard `job.finished` garante que _finalize rode uma vez
        só, mesmo que a API chame este método várias vezes por segundo.
        """
        with self._lock:
            job = self._job
            if job is None or job.finished or job.process is None:
                return
            # poll() != None significa que o processo já saiu. Não bloqueia.
            if job.process.poll() is None:
                return
            self._finalize(job)

    def tail_log(self, lines: int = 60) -> str:
        """Últimas linhas do stdout do treino, para a tela de diagnóstico."""
        with self._lock:
            job = self._job
        if job is None or job.log_path is None or not job.log_path.is_file():
            return ""
        try:
            # Lê o arquivo INTEIRO e corta no fim. O arquivo de um treino longo
            # passa de MB, mas ler as últimas N linhas exigiria.seek, que é
            # mais frágil num arquivo que o processo ainda está escrevendo.
            content = job.log_path.read_text(encoding="utf-8", errors="replace")
            # errors="replace": byte quebrado no meio do log não pode impedir
            # o operador de ver a mensagem de erro.
            return "\n".join(content.splitlines()[-lines:])
        except OSError:
            return ""


# Instância única: a API de treinamento e o painel precisam do mesmo estado de
# job, senão um "stop" poderia não achar o processo que o "start" criou.
training_service = TrainingService()

__all__ = ["TrainingService", "TrainingJob", "training_service"]
