"""Tracking de pilhas.

Duas opções, selecionáveis por ``TRACKER=`` no .env:

* ``internal`` (padrão) - tracker próprio baseado em IoU horizontal + vertical,
  adequado ao cenário do projeto (câmera fixa, pilhas lado a lado, sem
  sobreposição em profundidade). É determinístico e não depende de arquivos
  externos do Ultralytics.
* ``bytetrack`` / ``botsort`` - delegam ao Ultralytics, útil se as pilhas
  se movem muito (cobrindo/descobrindo pilhas durante a operação).

A identidade da pilha é o que permite mostrar "Pilha 1 / Pilha 2" de forma
estável no dashboard e no banco de dados.
"""

# ARQUIVO / MAPA
# O que faz: dá um NÚMERO ESTÁVEL para cada pilha, frame a frame. A câmera
# é fixa e as pilhas ficam paradas, então um tracker simples por IoU já
# resolve: não vale a pena pagar o custo de ByteTrack aqui.
# Por que o ID importa: sem ele, a cada frame o sistema renomearia as
# pilhas e o histórico do banco (uma linha por contagem) viraria lixo.
# Ordem de leitura:
#   1. _TRACK_SETTINGS              - apelidos de tracker do .env
#   2. Track                        - estado de uma pilha (hits/velocidade)
#   3. PileTracker                  - a máquina de associar
#   4. PileTracker._similarity      - quão parecidas duas caixas são
#   5. PileTracker.update           - associação e criação/expurgo
#   6. active_tracks / reset        - consultas e reinício
#   7. resolve_tracker_config       - TRACKER= -> (modo, yaml)
#   8. group_by_x                   - agrupamento 1D auxiliar

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from app.schemas import Detection, PileCandidate, iou_xyxy

log = logging.getLogger("ai")

_TRACK_SETTINGS = {
    # Aceita o apelido curto e o nome do arquivo: o .env e a tela podem
    # escrever "bytetrack" ou "bytetrack.yaml" e cair no mesmo lugar.
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "botsort.yaml": "botsort.yaml",
    "bytetrack.yaml": "bytetrack.yaml",
}


@dataclass
class Track:
    """Rastro persistente de uma pilha.

    ``hits`` conta frames confirmados, ``misses`` conta frames em que a
    pilha não apareceu (pode ser oclusão ou ruído do modelo).
    """

    tracking_id: int
    bbox: tuple[float, float, float, float]
    hits: int = 1
    misses: int = 0
    age: int = 1
    # (vx, vy, reservado). O terceiro slot não é usado: fica sempre 0.0
    # para não mexer no formato depois.
    velocity: tuple[float, float, field] = field(default=(0.0, 0.0, 0.0))
    # Últimos 30 centros (cx, cy, timestamp): janela curta de propósito,
    # para o histórico não crescer sem limite com a câmera rodando dias.
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    pile_id: int = 0

    @property
    def x1(self) -> float:
        """Borda esquerda em pixels."""
        return self.bbox[0]

    @property
    def y1(self) -> float:
        """Borda superior em pixels."""
        return self.bbox[1]

    @property
    def x2(self) -> float:
        """Borda direita em pixels."""
        return self.bbox[2]

    @property
    def y2(self) -> float:
        """Borda inferior em pixels."""
        return self.bbox[3]

    @property
    def cx(self) -> float:
        """Centro horizontal, em pixels."""
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        """Centro vertical, em pixels."""
        return (self.bbox[1] + self.bbox[3]) / 2.0

    def predict(self) -> tuple[float, float, float, float]:
        """BBox prevista um passo à frente (velocidade simples).

        Desloca a caixa pela velocidade, sem interpolar Kalman. Para uma
        pilha parada a velocidade é ~0 e a previsão é a própria caixa.
        """
        vx, vy = self.velocity[0], self.velocity[1]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return (self.bbox[0] + vx, self.bbox[1] + vy, self.bbox[0] + vx + w, self.bbox[1] + vy + h)

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        """Aplica a caixa vista neste frame e zera a contagem de faltas.

        A bbox nova é a do YOLO, não a prevista: corrigir a detecção evita
        que o erro da velocidade accumulate a cada frame.
        """
        prev_cx, prev_cy = self.cx, self.cy
        self.bbox = bbox
        new_cx, new_cy = self.cx, self.cy
        # Média móvel da velocidade para suavizar.
        # 50/50: com o mesmo peso do valor antigo, um único frame de detecção
        # errada (salto grande) só mexe pela metade na velocidade.
        self.velocity = (
            0.5 * self.velocity[0] + 0.5 * (new_cx - prev_cx),
            0.5 * self.velocity[1] + 0.5 * (new_cy - prev_cy),
            0.0,
        )
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.history.append((new_cx, new_cy, time_stamp()))

    def mark_missed(self) -> None:
        """Marca um frame sem detecção, mantendo a track viva por ``max_age``."""
        self.misses += 1
        self.age += 1


def time_stamp() -> float:
    """Epoch em segundos (usado só no histórico da track)."""
    import time

    return time.time()


class PileTracker:
    """Tracker IoU com predição de velocidade e IDs estáveis.

    Lógica de association (greedy por maior similaridade):

    1. Prediz a posição de cada track existente.
    2. Calcula a similaridade de cada par (candidato, track).
    3. Associa de cima para baixo enquanto a similaridade passar do limiar.
    4. Tracks não associados contam ``misses``; são removidos após o limite.
    5. Candidatos sem track recebem um novo ID incremental (nunca reaproveita
       um ID já encerrado, evitando "pilha 1" reaparecer com outro sentido).
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 1,
        center_weight: float = 0.6,
    ) -> None:
        # iou_threshold aqui é na verdade o limiar de SIMILARIDADE (IoU
        # misturada com distância de centro), não IoU puro.
        self.iou_threshold = iou_threshold
        # Frames que uma pilha pode ficar invisível antes de o ID morrer.
        # 15 frames cobrem um piscar de detecção sem trocar a identidade.
        self.max_age = max_age
        # Guardado para compatibilidade de assinatura; com min_hits=1 (padrão)
        # qualquer detecção já vira pilha na tela, sem esperar confirmação.
        self.min_hits = min_hits
        # 0.6 = mais peso para proximidade de centro do que para IoU: pilhas
        # mudam de altura quando alguém retira cadeiras do topo, e o centro
        # horizontal é o que realmente identifica a pilha.
        self.center_weight = center_weight
        # ID -> track. Dicionário porque a ordem de inserção não importa
        # e o acesso por ID é o mais frequente.
        self._tracks: dict[int, Track] = {}
        # Contador que só cresce. Nunca reutiliza um ID encerrado: se a
        # "Pilha 3" reaparecesse com outro sentido, o histórico do banco
        # misturaria duas pilhas diferentes.
        self._next_id = 1
        # Reservado e hoje sem uso: sobrou de uma versão que só publicava a
        # pilha depois de N confirmações. Fica para não quebrar chamadores
        # que já passam min_hits.
        self._confirmed: dict[int, int] = {}

    # ------------------------------------------------------------------ utils
    def _similarity(self, track: Track, cand: PileCandidate) -> float:
        """Quanto a caixa prevista da track combina com o candidato (0 a 1).

        A comparação é feita contra a caixa PREVISTA (posição futura), não
        contra a última vista: com a câmera fixa, isso absorve o atraso de
        um frame sem precisar de Kalman.
        """
        px1, py1, px2, py2 = track.predict()
        iou = iou_xyxy(px1, py1, px2, py2, cand.x1, cand.y1, cand.x2, cand.y2)
        # Distância de centro normalizada pelo tamanho médio da pilha.
        # Normalizar pela largura torna a medida adimensional: assim o mesmo
        # número de pixels de erro vale para uma pilha pequena e uma grande.
        w = max(1.0, (cand.width + (track.bbox[2] - track.bbox[0])) / 2.0)
        h = max(1.0, (cand.height + (track.bbox[3] - track.bbox[1])) / 2.0)
        dcx = (track.cx - cand.cx) / w
        dcy = (track.cy - cand.cy) / h
        # exp(-distância): decai rápido, então 0 = sobreposição perfeita e
        # cai perto de 0 quando os centros passam de ~1 tamanho de distância.
        center_sim = math.exp(-math.hypot(dcx, dcy))
        return (1.0 - self.center_weight) * iou + self.center_weight * center_sim

    # ------------------------------------------------------------------- API
    def update(self, candidates: list[PileCandidate]) -> list[tuple[int, PileCandidate, Track]]:
        """Associa candidatos às tracks. Retorna ``(tracking_id, cand, track)``.

        Sem lock: chamado só pela thread de inferência. A API lê o estado
        pelas cópias em :meth:`active_tracks`.
        """
        # Monta TODOS os pares acima do limiar antes de escolher. Com poucas
        # pilhas na cena (o caso normal) a lista é minúscula; o custo é
        # O(tracks x candidatos), o que é irrelevante aqui.
        pairs: list[tuple[float, int, int]] = []
        for ti, track in self._tracks.items():
            for ci, cand in enumerate(candidates):
                sim = self._similarity(track, cand)
                if sim >= self.iou_threshold:
                    pairs.append((sim, ti, ci))
        # Maior similaridade primeiro (greedy matching).
        # Greedy (e não Hungarian) é suficiente: com a câmera fixa quase
        # nunca há ambiguidade real, e o custo do Hungarian é bem maior.
        pairs.sort(key=lambda p: p[0], reverse=True)

        # Cada track e cada candidato podem ser usados uma vez só.
        used_tracks: set[int] = set()
        used_cands: set[int] = set()
        # Índice do candidato -> ID da track, montado neste frame.
        self._assigned: dict[int, int] = {}
        matches: list[tuple[int, int]] = []
        for _sim, ti, ci in pairs:
            if ti in used_tracks or ci in used_cands:
                continue
            used_tracks.add(ti)
            used_cands.add(ci)
            matches.append((ti, ci))

        for ti, ci in matches:
            # Caixa vista agora, não a prevista: a detecção manda.
            self._tracks[ti].update(candidates[ci].bbox)
            self._assigned[ci] = ti

        # Tracks sem par neste frame: contam uma falta, mas continuam vivas
        # (max_age frames de tolerância).
        for ti, track in self._tracks.items():
            if ti not in used_tracks:
                track.mark_missed()

        for cand_idx, cand in enumerate(candidates):
            if cand_idx in used_cands:
                continue
            # Pilha nova (ou que voltou depois de expurgada): ID novo.
            new_id = self._next_id
            self._next_id += 1
            track = Track(tracking_id=new_id, bbox=cand.bbox)
            self._tracks[new_id] = track
            # Candidatos sem track anterior também recebem track agora: sem
            # isso a primeira detecção de uma pilha nunca apareceria na saída.
            self._assigned[cand_idx] = new_id

        # Expurgo no FIM, e não no início: uma track que sobreviveu dentro
        # do max_age ainda pode ter sido reassociada acima.
        expired = [tid for tid, t in self._tracks.items() if t.misses > self.max_age]
        for tid in expired:
            log.debug("Track %d expirou após %d frames perdidos", tid, self._tracks[tid].misses)
            # pop em vez de remover da lista: some do tracker mas o ID já
            # foi gasto e não volta.
            self._tracks.pop(tid, None)

        results: list[tuple[int, PileCandidate, Track]] = []
        for ci, cand in enumerate(candidates):
            ti = self._assigned.get(ci)
            # Filtro final: um candidato cuja track expirou no mesmo frame
            # não volta na saída.
            if ti is None or ti not in self._tracks:
                continue
            results.append((ti, cand, self._tracks[ti]))
        return results

    def active_tracks(self) -> list[Track]:
        """Tracks vistas no frame atual (sem faltas)."""
        return [t for t in self._tracks.values() if t.misses == 0]

    def all_tracks(self) -> list[Track]:
        """Todas as tracks vivas, inclusive as que faltaram neste frame."""
        return list(self._tracks.values())

    def reset(self) -> None:
        """Esquece tudo e reinicia a numeração.

        Usado quando a câmera reinicia: as pilhas da cena antiga não
        têm relação com as novas.
        """
        self._tracks.clear()
        self._next_id = 1

    @property
    def track_count(self) -> int:
        """Quantas tracks estão vivas (contando as que faltaram)."""
        return len(self._tracks)


def resolve_tracker_config(name: str) -> tuple[str, str | None]:
    """Traduz ``TRACKER=`` para ``(modo, arquivo_yaml)``.

    ``("internal", None)`` ou ``("ultralytics", "bytetrack.yaml")``.

    Valor desconhecido cai em ``internal`` de propósito: é o tracker sem
    dependência externa, então um .env com erro de digitação não quebra
    a contagem.
    """
    key = (name or "internal").strip().lower()
    if key in ("internal", "none", "custom", ""):
        return "internal", None
    if key in _TRACK_SETTINGS:
        return "ultralytics", _TRACK_SETTINGS[key]
    # Aceita caminho customizado para um YAML de tracker.
    if key.endswith(".yaml"):
        return "ultralytics", name
    return "internal", None


def group_by_x(
    detections: Iterable[Detection],
    *,
    overlap_min: float = 0.30,
    gap_px: float = 40.0,
) -> list[list[Detection]]:
    """Agrupa detecções de cadeira em pilhas usando a posição horizontal.

    Premissa do projeto: pilhas lado a lado, nunca uma atrás da outra nem
    sobrepostas. Logo, duas cadeiras pertencem à mesma pilha se suas caixas
    se sobrepõem horizontalmente acima de ``overlap_min`` da largura; caso
    contrário, se a distância horizontal for maior que ``gap_px``, começam
    pilhas diferentes.

    Mesma ideia do ``PileDetector._cluster``, mas aqui o anchor é o ponto
    mais à direita do grupo (``current_right``), e não a última cadeira.
    Serve para testes e utilitários; o pipeline usa o _cluster.
    """
    dets = sorted(detections, key=lambda d: d.x1)
    if not dets:
        return []

    groups: list[list[Detection]] = [[dets[0]]]
    # "current_right" é a borda mais à direita do grupo: como as cadeiras
    # vêm ordenadas por x1, ela nunca precisa voltar para trás.
    current_right = dets[0].x2
    for det in dets[1:]:
        # interseção horizontal medida contra a última cadeira do grupo,
        # mas comparada com a borda mais à direita para não "escorregar".
        inter = min(current_right, det.x2) - max(groups[-1][-1].x1, det.x1)
        # 1e-6 evita divisão por zero numa caixa de largura 0.
        same_pile = inter > 0 and (inter / max(1e-6, det.width)) >= overlap_min
        gap = det.x1 - current_right
        if same_pile and gap <= gap_px:
            groups[-1].append(det)
            current_right = max(current_right, det.x2)
        else:
            groups.append([det])
            current_right = det.x2
    return groups


__all__ = [
    "Track",
    "PileTracker",
    "resolve_tracker_config",
    "group_by_x",
]
