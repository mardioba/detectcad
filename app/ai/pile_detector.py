"""Detecção de pilhas a partir das detecções do YOLO.

Suporta as duas estratégias pedidas:

* **classe ``pile``** - o modelo devolve a pilha inteira; as cadeiras
  individuais são usadas só como apoio.
* **classe ``chair`` apenas** - as cadeiras são agrupadas por posição
  horizontal para formar a pilha (funciona porque as pilhas ficam lado a
  lado).

A escolha é automática: se o modelo tem classe ``pile``, usa; senão agrupa.
Também é forçável pelo .env (``CHAIR_CLASSES`` / ``PILE_CLASSES``).
"""

# ARQUIVO / MAPA
# O que faz: recebe as detecções de cadeiras do YOLO e devolve PILHAS.
# Por que agrupar no eixo X: a câmera é fixa e as pilhas ficam lado a
# lado, nunca uma atrás da outra. Então a sobreposição horizontal é um
# sinal mais confiável que a vertical para dizer "é a mesma pilha".
# Duas estratégias (detect() escolhe):
#   A. classe `pile`     - o modelo já desenhou a pilha (melhor caso)
#   B. só classe `chair` - agrupa cadeiras por proximidade horizontal
# Saída: PileCandidate, ainda SEM id estável (isso é do tracker.py).
# Ordem de leitura:
#   1. PileDetector.__init__ / configure  - parâmetros e tamanho do frame
#   2. detect                            - escolhe a estratégia
#   3. _from_pile_class                  - estratégia A
#   4. _from_chair_group / _cluster      - estratégia B e o agrupamento 1D
#   5. _pile_conf / _group_conf          - confiança de cada pilha
#   6. _is_partial                       - pilha cortada pela borda
#   7. _inside                           - cadeira dentro da pilha
#   8. build_pile_detector               - liga com .env e nomes do modelo

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.schemas import Detection, PileCandidate, x_overlap_ratio

log = logging.getLogger("ai")


class PileDetector:
    """Transforma detecções YOLO em :class:`PileCandidate`.

    Sem estado entre frames: cada ``detect()`` é independente. Quem
    estabiliza a identidade da pilha é o ``tracker.py``.
    """

    def __init__(
        self,
        chair_classes: set[int],
        pile_classes: set[int],
        *,
        overlap_min: float = 0.30,
        gap_px: float = 40.0,
        partial_ratio: float = 0.12,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        # Sets de ÍNDICE de classe (não nome): o YOLO devolve class_id, e o
        # .env traduz nome -> indice via settings.class_indices().
        self.chair_classes = chair_classes
        self.pile_classes = pile_classes
        # overlap_min: fração da largura que duas cadeiras precisam
        # compartilhar para serem da mesma pilha.
        self.overlap_min = overlap_min
        # gap_px: folga em PIXELS entre pilhas vizinhas. Em pixels porque
        # a distância física entre pilhas é o que separa, e a câmera é fixa.
        self.gap_px = gap_px
        # partial_ratio: fração da largura/altura que, encostada na borda,
        # marca a pilha como cortada (contagem subestimada).
        self.partial_ratio = partial_ratio
        # (largura, altura) do frame, em pixels do ROI aplicado. Sem isso
        # _is_partial não consegue saber onde estão as bordas.
        self.image_size = image_size

    def configure(self, image_size: tuple[int, int] | None = None) -> None:
        """Ajusta o detector com o tamanho real do frame (ROI applied)."""
        if image_size is not None:
            self.image_size = image_size

    # ------------------------------------------------------------------ core
    def detect(
        self,
        detections: list[Detection],
        *,
        chair_weight: float | None = None,
    ) -> list[PileCandidate]:
        """Retorna as pilhas do frame.

        ``chair_weight``: quantas cadeiras cada detecção de ``chair``
        representa (1.0 = uma detecção é uma cadeira). Se o modelo foi
        treinado com ``pile``, o peso não se aplica às pilhas.
        """
        if not detections:
            return []

        # Separar por classe logo no começo: o resto do arquivo só trabalha
        # com as duas listas, sem repetir o filtro de classe.
        chairs = [d for d in detections if d.class_id in self.chair_classes]
        piles_cls = [d for d in detections if d.class_id in self.pile_classes]
        # weight só é repassado adiante (a contagem fina é do stack_counter);
        # aqui entra para manter a assinatura igual entre as estratégias.
        weight = settings.ai.chair_weight if chair_weight is None else chair_weight

        if piles_cls:
            # Estratégia A: o modelo já delimita a pilha.
            # Tem prioridade: se o modelo foi treinado com `pile`, a caixa
            # dele vale mais do que qualquer agrupamento inventado aqui.
            return self._from_pile_class(piles_cls, chairs, weight)
        if chairs:
            # Estratégia B: agrupa as cadeiras por posição horizontal.
            return self._from_chair_group(chairs, weight)
        return []

    # ------------------------------------------------------------- estratégias
    def _from_pile_class(
        self, piles: list[Detection], chairs: list[Detection], weight: float
    ) -> list[PileCandidate]:
        """Estratégia A: uma pilha por detecção de classe ``pile``.

        A caixa do YOLO é usada como está (é a melhor estimativa que
        temos) e as cadeiras dentro dela viram apoio de confiança.
        """
        cands: list[PileCandidate] = []
        for pile_det in piles:
            # Cadeiras que caem dentro da pilha. As de fora são
            # descartadas: contariam cadeiras de outra pilha.
            inside = [c for c in chairs if _inside(c, pile_det)]
            conf = self._pile_conf(pile_det, inside, weight)
            cands.append(
                PileCandidate(
                    bbox=(pile_det.x1, pile_det.y1, pile_det.x2, pile_det.y2),
                    chair_detections=inside,
                    conf=conf,
                    source="pile_class",
                    partial=self._is_partial(pile_det),
                )
            )
        return cands

    def _from_chair_group(self, chairs: list[Detection], weight: float) -> list[PileCandidate]:
        """Estratégia B: a pilha é a caixa que abraça o grupo de cadeiras.

        Não há informação da pilha no modelo, então o retângulo é
        reconstruído aqui pelas detecções de cadeira.
        """
        cands: list[PileCandidate] = []
        for group in self._cluster(chairs):
            if not group:
                continue
            # União (envelope) das caixas do grupo.
            x1 = min(d.x1 for d in group)
            y1 = min(d.y1 for d in group)
            x2 = max(d.x2 for d in group)
            y2 = max(d.y2 for d in group)
            # Padding leve para não cortar a base/encaixe da pilha.
            # Proporcional à altura (4%), com um mínimo de 2 px para
            # grupos pequenos não sumirem no ruído do tracker.
            pad = max(2.0, 0.04 * (y2 - y1))
            bbox = (x1, max(0.0, y1 - pad), x2, y2 + pad)
            conf = self._group_conf(group, weight)
            cands.append(
                PileCandidate(
                    bbox=bbox,
                    # Cópia da lista: o candidato é guardado pelo pipeline
                    # (contagem, histórico) e não deve ficar preso à lista
                    # interna do cluster.
                    chair_detections=list(group),
                    conf=conf,
                    source="chair_group",
                    # Só o bbox importa para _is_partial, por isso o
                    # PileCandidate temporário em vez de passar a tupla.
                    partial=self._is_partial(PileCandidate(bbox=bbox)),
                )
            )
        return cands

    def _cluster(self, chairs: list[Detection]) -> list[list[Detection]]:
        """Clustering 1D no eixo X, respeitando a premissa "lado a lado".

        É single-link: cada cadeira é comparada só com a última do grupo
        atual (a "âncora"). Barato e suficiente porque, numa pilha, as
        cadeiras se sobrepõem em cadeia vertical.
        """
        ordered = sorted(chairs, key=lambda d: d.x1)
        if not ordered:
            return []
        groups: list[list[Detection]] = [[ordered[0]]]
        for det in ordered[1:]:
            anchor = groups[-1][-1]
            overlap = x_overlap_ratio(det, anchor)
            gap = det.x1 - max(g.x2 for g in groups[-1])
            # Mesma pilha: sobreposição horizontal relevante OU encostadas
            # dentro da folga configurada.
            # O 0.05 evita juntar duas pilhas que quase se tocam: um
            # resíduo de sobreposição de poucos pixels não é a mesma pilha.
            if overlap >= self.overlap_min or (-self.gap_px <= gap <= self.gap_px and overlap > 0.05):
                groups[-1].append(det)
            else:
                groups.append([det])
        return groups

    # ------------------------------------------------------------- confiança
    def _pile_conf(self, pile: Detection, inside: list[Detection], weight: float) -> float:
        """Confiança da detecção da pilha, com reforço das cadeiras internas."""
        if not inside:
            return pile.conf * 0.6  # pilha sem cadeira dentro: confiança reduzida
        mean_conf = sum(d.conf for d in inside) / len(inside)
        # Muitas cadeiras detectadas dentro da pilha reforçam a confiança.
        # log1p faz o reforço crescer devagar: a partir de ~5 cadeiras o
        # suporte já satura em 1.0, então um monte não infla a confiança.
        support = min(1.0, 0.6 + 0.1 * math_log1p(len(inside)))
        # Mistura 60% caixa da pilha + 40% cadeiras de apoio.
        return min(1.0, 0.6 * pile.conf + 0.4 * mean_conf * support)

    def _group_conf(self, group: list[Detection], weight: float) -> float:
        """Confiança de uma pilha formada por agrupamento de cadeiras.

        Média das confidências com penalidade para grupos muito pequenos
        (1-2 cadeiras podem ser falso positivo) e para grupos enormes
        (podem ser várias pilhas coladas).
        """
        if not group:
            return 0.0
        mean_conf = sum(d.conf for d in group) / len(group)
        n = len(group)
        # 60 é a faixa de uma pilha real neste projeto; acima disso a
        # suspeita é serem várias pilhas unidas num só grupo.
        if n == 1:
            size_penalty = 0.7
        elif n == 2:
            size_penalty = 0.85
        elif n <= 60:
            size_penalty = 1.0
        else:
            # Cai 1.0 até 0.5 ao longo de 120 cadeiras a mais, e nunca
            # abaixo de 0.5 (senão a pilha sumiria do dashboard).
            size_penalty = max(0.5, 1.0 - (n - 60) / 120.0)
        return max(0.0, min(1.0, mean_conf * size_penalty))

    def _is_partial(self, pile: PileCandidate) -> bool:
        """Pilha cortada pelas bordas da imagem (contagem subestimada).

        Só é marcada parcial se encostar numa BORDA LATERAL (ou de topo) E
        na base: pilha no chão, com o pé encostando na borda de baixo, é
        normal e não significa corte.
        """
        if self.image_size is None:
            # Sem o tamanho do frame não dá para saber onde é a borda:
            # melhor não marcar nada do que marcar errado.
            return False
        w, h = self.image_size
        left = pile.x1 <= w * self.partial_ratio
        right = pile.x2 >= w * (1.0 - self.partial_ratio)
        top = pile.y1 <= h * self.partial_ratio
        # 2% de folga na base: o rodapé da cadeira encosta quase sempre.
        bottom = pile.y2 >= h * (1.0 - 0.02)
        return bool((left or right or top) and bottom)


def _inside(inner: Detection, outer: Detection) -> bool:
    """A cadeira está contida (pelo menos 60% da área) dentro da pilha?"""
    # Interseção retangular entre as duas caixas, com área negativa zerada
    # (caixas que não se cruzam).
    ix1 = max(inner.x1, outer.x1)
    iy1 = max(inner.y1, outer.y1)
    ix2 = min(inner.x2, outer.x2)
    iy2 = min(inner.y2, outer.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner.area <= 0:
        return False
    # Razão sobre a área da CADEIRA, não da pilha: uma cadeira que só
    # arranha a borda da pilha (grande) não deve contar como "dentro".
    # 60% tolera as imprecisões de caixa entre YOLO e o desenho da pilha.
    return inter / inner.area >= 0.6


def math_log1p(x: float) -> float:
    """``log(1 + x)`` com import local (só este ponto usa matemática)."""
    import math

    return math.log1p(max(0.0, float(x)))


def build_pile_detector(
    class_names: dict[int, str], image_size: tuple[int, int] | None = None
) -> PileDetector:
    """Cria o detector lendo os nomes de classe do modelo e o .env.

    Precisa ser reconstruído quando o modelo muda (hot-swap), porque os
    índices de classe do modelo novo podem ser diferentes dos antigos.
    """
    chairs, piles = settings.class_indices(class_names)
    if not chairs and not piles:
        # Modelo com classes desconhecidas: assume que todas são cadeira.
        # Melhor agrupar tudo e contar do que devolver zero pilhas.
        chairs = set(class_names.keys()) if class_names else set()
        if chairs:
            # O aviso importa: aqui a contagem é só uma estimativa, e o
            # operador precisa saber que o modelo não foi treinado para
            # este projeto.
            log.warning(
                "Nenhuma classe de cadeira/pilha reconhecida. Usando todas as classes (%s) como cadeira.",
                class_names,
            )
    return PileDetector(
        chair_classes=chairs,
        pile_classes=piles,
        # Os parâmetros vêm da seção de contagem do .env, não da de IA,
        # porque são regras da cena, não do modelo.
        overlap_min=settings.counting.pile_overlap_min,
        gap_px=settings.counting.pile_gap_px,
        partial_ratio=settings.counting.partial_ratio,
        image_size=image_size,
    )


__all__ = ["PileDetector", "build_pile_detector"]
