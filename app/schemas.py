"""Enums e estruturas de dados compartilhadas (sem dependência de ORM)."""

# ===========================================================================
# ARQUIVO / MAPA  -  schemas.py
#
# O que faz: o CONTRATO entre as camadas. Câmera, IA, contador, estabilidade e
# banco falam todos através destes enums e dataclasses. Não importa SQLAlchemy
# nem FastAPI de propósito: quem define o formato não pode depender de quem
# consome.
#
# Ordem de leitura:
#   1. utcnow / localnow - relógios (o sistema conta em UTC)
#   2. CameraState       - vida da câmera
#   3. CountStatus       - o quanto se confia no número de uma pilha
#   4. SourceMode        - de onde vem o frame
#   5. EventLevel        - gravidade dos eventos
#   6. Detection         - uma caixa do YOLO, já normalizada
#   7. iou_xyxy / x_overlap_ratio - geometria entre detecções e pilhas
#   8. PileCandidate     - pilha vista neste frame (ainda sem identidade)
#   9. CountEstimate     - resposta do StackCounter para uma pilha
#  10. PileState         - o que atravessa os frames (identidade + estabilidade)
#
# Três enums são str, Enum de propósito: o valor gravado no banco e o que
# viaja no JSON/WebSocket é a STRING ("STABLE", "rtsp"), não o objeto. Por
# isso o repository faz status.value e o banco tem coluna de texto.
#
# Convenção de todo o arquivo: coordenadas em PIXELS da imagem, bbox no
# formato xyxy (x1,y1 = canto superior esquerdo). OpenCV entrega BGR; quem
# converte para o YOLO é a camada de IA, não este arquivo.
# ===========================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


def utcnow() -> datetime:
    """Agora em UTC COM fuso - para marca de tempo em log e API."""
    return datetime.now(timezone.utc)


def localnow() -> datetime:
    """Agora no fuso da máquina - usado como padrão dos dataclasses em memória.

    O banco usa sempre UTC (models._utcnow, sem fuso). A diferença é
    proposital: em memória o datetime vai direto para o JSON da tela, e hora
    local é o que o operador reconhece.
    """
    return datetime.now()


class CameraState(str, Enum):
    """Estados possíveis da câmera.

    A máquina de estados é: OFFLINE -> CONNECTING -> ONLINE, e de ONLINE para
    RECONNECTING quando o stream cai, voltando a ONLINE se voltar. ERROR é
    terminal até uma nova tentativa (reinício/recarga de RTSP).
    """

    OFFLINE = "OFFLINE"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class CountStatus(str, Enum):
    """Estado da contagem de uma pilha.

    Não é decoração: cada valor muda o que o sistema faz.

    * ``STABLE``        - número aprovado, aparece em tela e pode ser gravado.
    * ``UNSTABLE``      - a janela de frames não concorda (pilha sendo mexida).
    * ``LOW_CONFIDENCE``- os estimadores não bateram acima do mínimo.
    * ``PARTIAL``       - pilha cortada pela borda da imagem; conta é o piso.
    * ``UNKNOWN``       - sem leitura útil (ROI vazia, falha de inferência).

    A prioridade quando mais de um problema existe é
    UNKNOWN > LOW_CONFIDENCE > UNSTABLE > PARTIAL > STABLE: o pior motivo
    vence, senão o painel esconderia justamente o caso que exige atenção.
    """

    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class SourceMode(str, Enum):
    """Origem dos frames - permite testar sem câmera."""

    # rtsp  = câmera real, o modo de produção
    # video = arquivo .mp4, para reproducir um caso ruim
    # image = foto única, para calibrar sem vídeo
    # none  = nada conectado (a tela mostra "sem fonte")
    RTSP = "rtsp"
    VIDEO = "video"
    IMAGE = "image"
    NONE = "none"


class EventLevel(str, Enum):
    """Gravidade do evento, espelhando os níveis do logging do Python.

    DEBUG fica fora do banco por padrão: encheria a tabela de auditoria sem
    acrescentar informação.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class Detection:
    """Uma detecção YOLO já normalizada (coordenadas em pixels da imagem)."""

    # slots=True (acima): estas detecções são criadas milhares de vezes por
    # segundo; sem __dict__ o objeto fica menor e mais rápido de montar.

    # bbox xyxy em pixels (não normalizado 0..1): o ROI, o tracker e o
    # desenho do painel trabalham todos na mesma escala.
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str
    # track_id só existe depois do tracker: a detecção crua vem sem ele.
    track_id: int | None = None
    mask: Any = None  # ndarray binário (uint8 0/255) ou None

    @property
    def width(self) -> float:
        # max(0.0, ...): o tracker pode devolver caixa invertida num frame
        # ruim. Área negativa contaminaria IoU e separação de pilhas.
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def iou(self, other: "Detection") -> float:
        return iou_xyxy(self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2)

    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_dict(self) -> dict[str, Any]:
        """Forma enviada ao painel.

        Arredonda por casa (não no consumer) porque o dashboard só mostra 1
        casa, e o WebSocket manda isso a cada frame. A máscara NÃO vai aqui:
        é grande demais para trafegar, e só interessa ao contador.
        """
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "conf": round(self.conf, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


def iou_xyxy(
    ax1: float, ay1: float, ax2: float, ay2: float, bx1: float, by1: float, bx2: float, by2: float
) -> float:
    """IoU entre dois retângulos ``xyxy``."""
    # Interseção: interseção das projeções em x e em y (o retângulo que sobra).
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    # max(0.0, ...): caixas que não se tocam dariam largura/altura negativas
    # e um produto positivo falso - o clamp zera isso.
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    # União = A + B - interseção (a parte contada duas vezes sai uma vez).
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def x_overlap_ratio(a: "Detection", b: "Detection") -> float:
    """Fração da largura de ``a`` que se sobrepõe horizontalmente com ``b``.

    Como as pilhas estão lado a lado e nunca atrás umas das outras, a
    sobreposição horizontal é o sinal mais confiável para separar pilhas.
    """
    inter = min(a.x2, b.x2) - max(a.x1, b.x1)
    if inter <= 0:
        return 0.0
    # Normaliza pela largura de "a", não pela união: aqui a pergunta é
    # "quanto de a está dentro de b", e o resultado fica em 0..1.
    # O piso de 1e-6 evita divisão por zero com caixa de largura ~0.
    return inter / max(1e-6, a.width)


@dataclass(slots=True)
class PileCandidate:
    """Pilha detectada no frame, antes da estabilização de ID."""

    # bbox guardado como tupla (e não quatro campos) porque é assim que a
    # lista inteira é desenhada e comparada no painel.
    bbox: tuple[float, float, float, float]
    chair_detections: list[Detection] = field(default_factory=list)
    conf: float = 0.0
    # "pile_class"  = o YOLO tem uma classe "pilha" e recortou a região.
    # "chair_group" = não há classe de pilha; a pilha foi montada agrupando
    #                 cadeiras por sobreposição horizontal (o modo padrão).
    # A diferença importa na auditoria: os dois falham de jeitos distintos.
    source: Literal["pile_class", "chair_group"] = "chair_group"
    # partial=True quando a pilha encosta na borda da imagem. O número é um
    # PISO, não a verdade - por isso o status vai para PARTIAL e o alerta
    # avisa, em vez de mostrar uma contagem errada como se fosse boa.
    partial: bool = False

    # As properties abaixo existem para PileCandidate falar como Detection:
    # o resto do código (x_overlap_ratio, desenho, pilha de detecções) já sabe
    # ler .width/.cx e não ganhou um "se for PileCandidate...".

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass(slots=True)
class CountEstimate:
    """Resultado do StackCounter para uma pilha, em um único frame."""

    # count é a resposta (ainda instável: pode mudar no frame seguinte).
    count: int
    # confidence é o valor fundido de todos os sinais; detecção e geometria
    # entram com peso menor. Guardar os dois permite reavaliar a fusão sem
    # reprocessar a imagem.
    confidence: float
    detection_confidence: float
    # "periodic" = conta pelas camadas do perfil vertical
    # "extent"   = conta pela altura útil da pilha
    # "detections" = uma cadeira por caixa
    # "size"     = estima pela área
    # Guardar qual método venceu é o que permite ver que método funciona.
    method: str
    # candidates = cada estimador e o número que ele proposed (diagnóstico)
    # candidate_weights = o quanto cada um pesou na média ponderada
    candidates: dict[str, float] = field(default_factory=dict)
    candidate_weights: dict[str, float] = field(default_factory=dict)
    # chair_height_px e pitch_px são a escala da pilha: altura de uma cadeira
    # e distância entre camadas. Toda a contagem por período depende deles.
    chair_height_px: float = 0.0
    pitch_px: float = 0.0
    # notes são avisos livres (ex.: "recorte toca a borda"). Aparecem no painel
    # e ajudam a entender por que o número não é o esperado.
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "method": self.method,
            "candidates": {k: round(v, 2) for k, v in self.candidates.items()},
            "weights": {k: round(v, 3) for k, v in self.candidate_weights.items()},
            "chair_height_px": round(self.chair_height_px, 1),
            "pitch_px": round(self.pitch_px, 1),
            "notes": self.notes,
        }


@dataclass(slots=True)
class PileState:
    """Estado estável de uma pilha através do tempo."""

    # pile_id é interno e estável (sobrevive ao tracker renumerar);
    # tracking_id é o do tracker e PODE mudar. Manter os dois separados é o
    # que impede o histórico de se partir quando o tracker reatribui um ID.
    pile_id: int
    tracking_id: int
    bbox: tuple[float, float, float, float]
    # raw_count = o que este frame viu. stable_count = o valor aprovado após
    # o filtro de janela. A tela mostra o stable; a diferença entre os dois é
    # justamente o que a colagem ainda não estabilizou.
    raw_count: int
    stable_count: int
    confidence: float
    detection_confidence: float
    stability_confidence: float
    status: CountStatus
    method: str = "auto"
    candidates: dict[str, float] = field(default_factory=dict)
    pitch_px: float = 0.0
    chair_height_px: float = 0.0
    partial: bool = False
    # Quantos frames seguidos a pilha não apareceu. Não desliga a pilha de
    # uma vez: o tracker pode perder a detecção por alguns frames sem que a
    # pilha tenha sumido de verdade.
    missed_frames: int = 0
    # manual=True quando o operador corrigiu o valor. A pilha volta a ser
    # contada pela IA, mas a correção não é sobrescrita.
    manual: bool = False
    first_seen: datetime = field(default_factory=localnow)
    last_seen: datetime = field(default_factory=localnow)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """O objeto que o dashboard recebe (via WebSocket) e que o snapshot
        de estado do banco recebe.

        Espelha o formato do registro de contagem: "count" é o stable_count
        aqui, e a coluna chair_count lá. Mesma ideia em dois lugares.
        """
        return {
            "pile_id": self.pile_id,
            "tracking_id": self.tracking_id,
            "bbox": [round(v, 1) for v in self.bbox],
            "raw_count": self.raw_count,
            "count": self.stable_count,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "stability_confidence": round(self.stability_confidence, 4),
            # .value e não o enum: o que trafega é a string, e o mesmo formato
            # que vai para a coluna de status.
            "status": self.status.value,
            "method": self.method,
            "candidates": {k: round(v, 2) for k, v in self.candidates.items()},
            "pitch_px": round(self.pitch_px, 1),
            "chair_height_px": round(self.chair_height_px, 1),
            "partial": self.partial,
            "manual": self.manual,
            "missed_frames": self.missed_frames,
            # ISO com segundos: legível no painel e curto para o JSON.
            "first_seen": self.first_seen.isoformat(timespec="seconds"),
            "last_seen": self.last_seen.isoformat(timespec="seconds"),
            "notes": self.notes,
        }


__all__ = [
    "CameraState",
    "CountStatus",
    "SourceMode",
    "EventLevel",
    "Detection",
    "PileCandidate",
    "CountEstimate",
    "PileState",
    "iou_xyxy",
    "x_overlap_ratio",
    "localnow",
    "utcnow",
]
