"""Detector YOLO.

Responsabilidades:

* carregar o modelo configurado em ``.env`` (YOLO_MODEL);
* escolher o device (CUDA se existir, senão CPU) automaticamente;
* normalizar a saída do Ultralytics em objetos :class:`Detection` independentes
  de biblioteca;
* funcionar em **modo de preparação** quando o modelo específico ainda não
  foi treinado, informando isso claramente em vez de fingir uma contagem.
"""

# ARQUIVO / MAPA
# O que faz: embrulha o ``ultralytics.YOLO`` e é a única ponte do projeto
# com a biblioteca. Carrega o .pt, escolhe CPU/GPU, roda inferência e
# traduz a saída da Ultralytics em objetos ``Detection`` próprios (assim o
# resto do sistema não depende da lib).
# Modo de preparação: sem modelo da empresa treinado, ``load()`` devolve
# ``False`` e ``load_error`` explica o motivo. O sistema segue vivo e mostra
# "sem modelo" — nunca inventa contagem com um yolo11n genérico.
# Ordem de leitura:
#   1. PRETRAINED_STEMS / ModelNotAvailable  - o que é modelo genérico
#   2. YoloDetector.__init__                  - device, lock, telemetria
#   3. gpu_info                               - telemetria de GPU
#   4. _classify_model / is_available         - classificar o .pt do .env
#   5. load / unload / hot_swap               - ciclo de vida + rollback
#   6. is_loaded / class_names / info         - estado para a página /modelo
#   7. predict                                - inferência normal
#   8. _parse                                 - normalização do Ultralytics
#   9. track                                  - inferência com ByteTrack/BoT-SORT

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.schemas import Detection

log = logging.getLogger("ai")

# Nomes de arquivo de modelos pré-treinados genéricos do Ultralytics.
# Não são adequados para contar cadeiras empilhadas: servem apenas para
# validar o pipeline enquanto o modelo da empresa não existe.
PRETRAINED_STEMS = {
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
    "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x",
    "rtdetr", "rtdetr-l", "mobile_sam", "sam_b", "yolo26n",
}


class ModelNotAvailable(RuntimeError):
    """Levantada quando não há modelo carregado.

    Existe para o loop de inferência diferenciar "modelo faltando" de
    "inferência quebrou": o primeiro é esperado em modo de preparação e
    vira uma mensagem na tela, sem log de erro.
    """


class YoloDetector:
    """Wrapper fino sobre ``ultralytics.YOLO`` com hot-swap seguro.

    Uma instância é compartilhada entre a thread de inferência, a API e a
    tela de modelos, por isso todos os acessos ao modelo passam pelo lock.
    """

    def __init__(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        self.configured_path = Path(model_path) if model_path else settings.model_path
        self.device = device or settings.resolve_device()
        self._model: Any = None
        # RLock (e não Lock) porque load() é chamado por hot_swap(), que já
        # segura este mesmo lock: com Lock simples isso travaria.
        self._lock = threading.RLock()
        self._class_names: dict[int, str] = {}
        self._task = "detect"
        self.loaded_at: float | None = None
        self.load_error: str = ""
        self.trained_model: bool = False  # True só para modelo da empresa (.pt em data/models)
        # Métricas de telemetria: são atualizadas fora do lock de propósito.
        # Uma corrida aqui só distorce um número de dashboard, nunca o
        # resultado da contagem.
        self.last_infer_ms: float = 0.0
        self.inference_count: int = 0

    # ------------------------------------------------------------- device/gpu
    @staticmethod
    def gpu_info() -> dict[str, Any]:
        """Informações reais da GPU (vazio se não houver CUDA).

        Só alimenta a tela de diagnóstico. Por isso engole qualquer erro:
        falta do torch ou driver quebrado não pode derrubar o sistema.
        """
        info: dict[str, Any] = {"cuda_available": False, "gpus": []}
        try:
            import torch
        except Exception:
            return info
        try:
            info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["cuda_version"] = torch.version.cuda
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_gb = props.total_memory / (1024**3)
                    # free vem do driver, não do PyTorch: mede a VRAM real
                    # ocupada, inclusive por outro processo na mesma GPU.
                    free, _total = torch.cuda.mem_get_info(i)
                    info["gpus"].append(
                        {
                            "index": i,
                            "name": props.name,
                            "vram_total_gb": round(total_gb, 2),
                            "vram_used_gb": round(total_gb - free / (1024**3), 2),
                            "capability": f"{props.major}.{props.minor}",
                            "multi_processor_count": props.multi_processor_count,
                        }
                    )
        except Exception as exc:  # pragma: no cover
            log.debug("gpu_info falhou: %s", exc)
        return info

    # ------------------------------------------------------------------ model
    def _classify_model(self, path: Path) -> tuple[bool, bool]:
        """Retorna (existe, é modelo pré-treinado genérico).

        Olha só o nome do arquivo, não o conteúdo: nomes genéricos do
        Ultralytics começam com yolo11/yolov8/yolov5/yolov10/rtdetr e
        carregam um sufixo de tamanho (n/s/m/l/x). Qualquer outra coisa é
        tratada como modelo da empresa.
        """
        stem = path.stem.lower()
        is_pretrained = any(stem.startswith(s) or stem == s for s in PRETRAINED_STEMS)
        return path.is_file(), is_pretrained

    def is_available(self) -> bool:
        """O arquivo configurado existe no disco?"""
        exists, _ = self._classify_model(self.configured_path)
        return exists

    def is_pretrained_generic(self) -> bool:
        """O arquivo configurado é um modelo genérico (não serve para contar)?"""
        _, pre = self._classify_model(self.configured_path)
        return pre

    def load(self, path: str | Path | None = None) -> bool:
        """Carrega (ou recarrega) o modelo. Retorna ``True`` se carregou.

        Chamada direta zera o modelo atual se falhar (comportamento de
        "recarregamento"). Quem precisa preservar o anterior usa
        :meth:`hot_swap`, que trata o rollback.

        Carregar o .pt é lento (dezenas de MB). O lock fica segurado durante
        toda a operação para que ninguém veja um modelo pela metade.
        """
        with self._lock:
            target = Path(path) if path else self.configured_path
            # Caminho relativo do .env é sempre relativo à raiz do projeto,
            # nunca ao diretório atual do processo.
            if not target.is_absolute():
                target = settings.base_dir / target
            if not target.is_file():
                self._model = None
                self.load_error = f"Modelo não encontrado: {target}"
                self.trained_model = False
                log.warning("Modelo não encontrado: %s (sistema em modo de preparação)", target)
                return False
            try:
                from ultralytics import YOLO
            except Exception as exc:
                self.load_error = f"Ultralytics indisponível: {exc}"
                log.error("Ultralytics não importável: %s", exc)
                return False

            try:
                model = YOLO(str(target))
                # Ultralytics escolhe o device no predict(); forçamos aqui.
                try:
                    model.to(self.device)
                except Exception:
                    # Sem CUDA (ou modelo sem pesos) o .to() estoura. Não é
                    # fatal aqui: o predict ainda pode resolver, e se não
                    # resolver a mensagem de erro sai mais clara dali.
                    pass
                names = getattr(model, "names", {}) or {}
                # A Ultralytics devolve dict {id: nome} ou list conforme a
                # versão; o projeto só quer o dict.
                if isinstance(names, dict):
                    self._class_names = {int(k): str(v) for k, v in names.items()}
                else:
                    self._class_names = {i: str(n) for i, n in enumerate(names)}
                task = getattr(model, "task", "detect")
                self._task = task
            except Exception as exc:
                self._model = None
                self.load_error = f"Falha ao carregar o modelo: {exc}"
                log.exception("Falha ao carregar o modelo %s", target)
                return False

            # Só publica o modelo novo no final: até aqui nada mudou para
            # quem já está chamando predict().
            self._model = model
            self.configured_path = target
            _, pre = self._classify_model(target)
            self.trained_model = not pre
            self.loaded_at = time.time()
            self.load_error = ""
            log.info(
                "Modelo carregado: %s | device=%s | task=%s | classes=%s | treinado_pela_empresa=%s",
                target.name, self.device, self._task, self._class_names, self.trained_model,
            )
            return True

    def unload(self) -> None:
        """Solta o modelo e devolve a memória (RAM e VRAM) ao sistema."""
        with self._lock:
            self._model = None
            self.loaded_at = None
            try:
                import gc

                import torch

                # Só o gc.collect() não basta na CUDA: o cache do PyTorch
                # segura a VRAM mesmo sem tensor vivo.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            log.info("Modelo descarregado da memória")

    def hot_swap(self, path: str | Path) -> tuple[bool, str]:
        """Troca o modelo em produção sem derrubar o pipeline.

        Se o novo modelo não carregar, o anterior continua ativo.

        O segredo é que :meth:`predict` copia a referência do modelo
        dentro do lock e solta antes de inferir. Então um predict que já
        começou termina com o modelo velho, enquanto o próximo já usa o
        novo: nunca os dois ao mesmo tempo, e a thread não fica parada
        esperando o carregamento.
        """
        with self._lock:
            previous = self._model
            ok = self.load(path)
            if not ok:
                error = self.load_error
                self._model = previous  # rollback
                self.load_error = error
                return False, error
            log.info("Modelo trocado em produção: %s", path)
            return True, ""

    @property
    def is_loaded(self) -> bool:
        """Há modelo em memória pronto para inferir?"""
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        """Cópia do mapa {id: nome} do modelo carregado (evita corrida)."""
        return dict(self._class_names)

    @property
    def task(self) -> str:
        """Tarefa do modelo: ``detect``, ``segment`` etc."""
        return self._task

    def class_indices(self) -> tuple[set[int], set[int]]:
        """Índices de (chair, pile) conforme o modelo carregado e o .env.

        Os índices saem dos NOMES de classe, então um .env errado devolve
        conjunto vazio em vez de erro — o filtro no predict() simplesmente
        não vai separar nada.
        """
        return settings.class_indices(self._class_names)

    def info(self) -> dict[str, Any]:
        """Metadados do modelo para a página /modelo."""
        return {
            "path": str(self.configured_path),
            "filename": self.configured_path.name,
            "exists": self.configured_path.is_file(),
            "loaded": self.is_loaded,
            "device": self.device,
            "task": self._task,
            "trained_model": self.trained_model,
            "is_pretrained_generic": self.is_pretrained_generic(),
            "classes": self._class_names,
            "loaded_at": self.loaded_at,
            "load_error": self.load_error,
            "last_infer_ms": round(self.last_infer_ms, 1),
            "inference_count": self.inference_count,
            "gpu": self.gpu_info(),
            "using_gpu": self.device.startswith("cuda"),
        }

    # -------------------------------------------------------------- predict
    def predict(
        self,
        frame: np.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        classes: set[int] | None = None,
    ) -> tuple[list[Detection], float]:
        """Executa a inferência e devolve ``(deteções, latência_ms)``.

        Lança :class:`ModelNotAvailable` se não houver modelo carregado.
        """
        # Só a leitura da referência é protegida. Segurar o lock durante a
        # inferência inteira travaria o hot-swap (e a API) por segundos.
        with self._lock:
            model = self._model
        if model is None:
            raise ModelNotAvailable(self.load_error or "Nenhum modelo carregado")

        # Parâmetros vêm do .env quando o chamador não manda, para que a
        # calibração por câmera consiga sobrescrever só o que precisa.
        conf = settings.ai.confidence_threshold if conf is None else conf
        iou = settings.ai.iou_threshold if iou is None else iou
        imgsz = settings.ai.imgsz if imgsz is None else imgsz

        started = time.perf_counter()
        try:
            results = model.predict(
                source=frame,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                device=self.device,
                # Filtrar classes aqui (e não depois) economiza NMS e memória:
                # a classe que não é usada nem precisa existir no retorno.
                classes=sorted(classes) if classes else None,
                # Teto de detecções por frame: protege RAM/latência quando a
                # cena tem muito ruído (cadeiras fora do ROI, por exemplo).
                max_det=settings.ai.max_det,
                # fp16 só faz sentido na GPU; na CPU a meia precisão quebra.
                half=bool(settings.ai.half_precision) and self.device.startswith("cuda"),
                verbose=False,
            )
        except Exception as exc:
            log.exception("Falha na inferência")
            raise RuntimeError(f"Falha na inferência: {exc}") from exc

        # Mede só a chamada ao modelo (a conversão para Detection vem
        # depois), que é o número que interessa para saber se o FPS dá.
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_infer_ms = latency_ms
        self.inference_count += 1

        detections = self._parse(results, conf)
        return detections, latency_ms

    def _parse(self, results: list[Any], conf_threshold: float) -> list[Detection]:
        """Converte ``Results`` do Ultralytics em :class:`Detection`.

        É aqui que o projeto se separa da biblioteca: depois deste ponto
        nada mais importa se a Ultralytics mudou a API ou a versão.
        """
        detections: list[Detection] = []
        if not results:
            return detections
        # predict() devolve uma lista com um item por imagem; aqui é 1:1.
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return detections

        names = self._class_names
        n = len(boxes)
        # As saídas vêm como tensor na GPU; o try/except cobre o caso em que
        # já são numpy (modelo exportado, mock de teste, backend ONNX).
        try:
            xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            xyxy = np.asarray(boxes.xyxy)
        try:
            confs = boxes.conf.cpu().numpy()
        except Exception:
            confs = np.asarray(boxes.conf)
        try:
            clss = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            clss = np.asarray(boxes.cls).astype(int)
        # id só existe quando a inferência veio de track(); em predict() puro
        # fica None e a identidade da pilha é responsabilidade do tracker.py.
        try:
            tids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * n
        except Exception:
            tids = [None] * n

        # Máscaras de segmentação (se o modelo for de segmentação)
        masks: list[np.ndarray | None] = [None] * n
        seg = getattr(res, "masks", None)
        if seg is not None and getattr(seg, "data", None) is not None:
            try:
                mask_data = seg.data.cpu().numpy()
                for i in range(min(n, mask_data.shape[0])):
                    # 0/255 uint8: é o formato que o OpenCV e a sobreposição
                    # do dashboard esperam, não booleano.
                    masks[i] = (mask_data[i] > 0.5).astype(np.uint8) * 255
            except Exception:
                log.debug("Não foi possível extrair máscaras")

        # orig_shape é (altura, largura, canais) da imagem original, já que o
        # modelo redimensiona internamente para imgsz.
        h_img, w_img = (res.orig_shape if getattr(res, "orig_shape", None) else frame_shape(res))[:2]

        for i in range(n):
            c = float(confs[i])
            # Filtro final de confiança: o predict() já filtrou, mas o
            # chamador pode ter enviado um conf menor que o do .env.
            if c < conf_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            # As coordenadas são recortadas na imagem: uma caixa só meio fora
            # do quadro tem área enganosa e quebraria o agrupamento no eixo X.
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w_img), x2), min(float(h_img), y2)
            # Caixa degenerada (só uma borda, ou totalmente fora) não vira
            # Detection: largura/área zero quebram o agrupamento no eixo X.
            if x2 <= x1 or y2 <= y1:
                continue
            cid = int(clss[i])
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=c,
                    class_id=cid,
                    # Nome desconhecido vira o próprio id: melhor mostrar "3"
                    # do que um None que quebra a tela.
                    class_name=names.get(cid, str(cid)),
                    track_id=int(tids[i]) if tids[i] is not None else None,
                    mask=masks[i],
                )
            )
        return detections

    def track(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack.yaml",
        **kwargs: Any,
    ) -> tuple[list[Detection], float]:
        """Tracking via Ultralytics (ByteTrack / BoT-SORT). Retorna track_id.

        Alternativa ao tracker próprio de ``tracker.py``, para quando as
        pilhas se movem muito (cobrindo e descobrindo pilhas).
        """
        with self._lock:
            model = self._model
        if model is None:
            raise ModelNotAvailable(self.load_error or "Nenhum modelo carregado")
        started = time.perf_counter()
        try:
            results = model.track(
                source=frame,
                # persist=True mantém o estado do tracker entre chamadas: sem
                # isso cada frame seria um rastreamento do zero e os IDs
                # nunca casariam de um frame para o outro.
                persist=True,
                tracker=tracker,
                conf=kwargs.get("conf", settings.ai.confidence_threshold),
                iou=kwargs.get("iou", settings.ai.iou_threshold),
                imgsz=kwargs.get("imgsz", settings.ai.imgsz),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            log.exception("Falha no tracking")
            raise RuntimeError(f"Falha no tracking: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        # O mesmo _parse() do predict(): assim as duas rotas devolvem o
        # mesmo tipo de dado e o resto do pipeline não precisa saber qual
        # delas foi usada.
        return self._parse(results, kwargs.get("conf", settings.ai.confidence_threshold)), latency_ms


def frame_shape(res: Any) -> tuple[int, int, int]:
    """Tamanho original (altura, largura, canais) com um padrão de reserva."""
    # 480x640 é o fallback: sem ele, um Results sem orig_shape derrubaria o
    # recorte das caixas ao recortar pela imagem.
    shape = getattr(res, "orig_shape", (480, 640, 3))
    return tuple(shape)  # type: ignore[return-value]


__all__ = ["YoloDetector", "ModelNotAvailable", "PRETRAINED_STEMS"]
