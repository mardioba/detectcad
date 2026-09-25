"""Gerador de pilhas sintéticas de cadeiras para TESTE do algoritmo.

Isto **não** é o modelo de IA. É uma herramienta de testes que desenha
pilhas com estrutura repetida (a mesma assinatura visual de uma pilha real
de cadeiras: linhas horizontais repetidas com passo constante) para
verificar o :class:`~app.counting.stack_counter.StackCounter` de forma
determinística e mensurável.

Serve para:

* validar o algoritmo de periodicidade com um número *conhecido* de camadas;
* gerar um vídeo sintético para testar o pipeline completo sem câmera;
* criar imagens para o dashboard em modo demonstração.

Uso::

    python -m tools.make_synthetic_stack --chairs 20 --out data/results/test20.png
    python -m tools.make_synthetic_stack --chairs 8,15,32 --out /tmp/cenario.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_stack(
    img: np.ndarray,
    cx: int,
    base_y: int,
    n_chairs: int,
    chair_h: int = 34,
    width: int = 130,
    color: tuple[int, int, int] = (70, 130, 200),
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, int]:
    """Desenha ``n_chairs`` cadeiras ENCAIXADAS e devolve a bbox da pilha.

    Modelagem fiel a uma pilha real de cadeiras plásticas:

    * cada cadeira revela **uma** borda horizontal forte — a frente do
      assento. As outras partes ficam escondidas dentro da cadeira de baixo.
      (Esta é a razão de o gerador anterior, que desenhava 3-4 linhas por
      cadeira, ser irrealista e enganar a análise de empilhamento.)
    * apenas a cadeira **de cima** mostra o encosto inteiro, e uma vez só.
    * as **pernas** aparecem só na região inferior da pilha.
    """
    rng = rng or np.random.default_rng(7)
    top_y = base_y - n_chairs * chair_h
    x1 = cx - width // 2
    x2 = cx + width // 2

    # Sombra suave no chão
    cv2.rectangle(img, (x1 - 8, base_y - 3), (x2 + 8, base_y + 5), (40, 40, 40), -1)

    dark = tuple(int(c * 0.75) for c in color)
    lighter = tuple(min(255, int(c * 1.15) + 20) for c in color)
    leg_zone = base_y - 2 * chair_h

    for i in range(n_chairs):
        top = base_y - (i + 1) * chair_h
        # Frente do assento: UMA linha por cadeira. Uma faixa preenchida
        # criaria duas arestas e mudaria o espaçamento medido.
        seat_y = top + chair_h - 8
        cv2.line(img, (x1, seat_y), (x2, seat_y), color, 2)

        # Pernas: só na parte de baixo da pilha
        if top > leg_zone:
            cv2.rectangle(img, (x1 + 4, top + chair_h - 6), (x1 + 12, top + chair_h), lighter, -1)
            cv2.rectangle(img, (x2 - 12, top + chair_h - 6), (x2 - 4, top + chair_h), lighter, -1)

    # Encosto: só a cadeira de cima
    top_chair = base_y - chair_h
    cv2.rectangle(img, (cx - 34, top_chair - chair_h + 6), (cx + 34, top_chair), dark, -1)
    for k in range(5):  # ripas do encosto (verticais, não afetam o perfil)
        sx = cx - 26 + k * 13
        cv2.line(img, (sx, top_chair - chair_h + 10), (sx, top_chair - 6), lighter, 2)

    if noise > 0:
        y0, y1 = max(0, top_y - 5), base_y + 5
        x0, x1n = max(0, x1 - 8), x2 + 8
        region = img[y0:y1, x0:x1n]
        if region.size:
            noisy = region.astype(np.float32) + rng.normal(0.0, noise * 255.0, region.shape[:2])[:, :, None]
            img[y0:y1, x0:x1n] = np.clip(noisy, 0, 255).astype(np.uint8)

    return x1, top_y, x2, base_y


def make_scene(
    counts: list[int],
    width: int = 1280,
    height: int = 720,
    chair_h: int = 34,
    n_piles: int = 3,
    seed: int = 7,
) -> np.ndarray:
    """Cria uma cena com ``n_piles`` pilhas de alturas variadas."""
    img = np.full((height, width, 3), 195, dtype=np.uint8)
    # Gradiente de fundo (iluminação não uniforme - o detrend precisa lidar com isso)
    grad = np.linspace(150, 215, height, dtype=np.float32)[:, None]
    img[:] = np.clip(grad[:, :, None].repeat(3, axis=2), 0, 255).astype(np.uint8)
    # Chão
    base_y = int(height * 0.88)
    cv2.rectangle(img, (0, base_y), (width, height), (110, 110, 110), -1)

    rng = np.random.default_rng(seed)
    step = width / (n_piles + 1)
    for i, n in enumerate(counts):
        cx = int(step * (i + 1))
        draw_stack(img, cx, base_y, n, chair_h=chair_h, rng=rng)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera pilhas sintéticas de cadeiras para teste")
    parser.add_argument("--chairs", required=True, help="Ex.: 20 ou 8,15,32")
    parser.add_argument("--out", required=True, help="Arquivo de saída (.png)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--chair-h", type=int, default=34)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    counts = [int(v) for v in args.chairs.replace(" ", "").split(",") if v]
    img = make_scene(counts, args.width, args.height, args.chair_h, len(counts), args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"Imagem gerada: {out} | pilhas={counts} | cadeira={args.chair_h}px")


if __name__ == "__main__":
    main()
