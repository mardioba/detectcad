"""Extrai frames de um vídeo para virar dataset de treinamento.

Este é o caminho mais prático para ter imagens suficientes: em vez de
fotografar centenas de vezes, você grava um vídeo do galpão e extrai os
frames.

    python -m tools.extract_frames --video Cadeiras_Video.mp4
    python -m tools.extract_frames --video filmed.mp4 --fps 2 --crop 0,0,1920,1080

O que ele faz:

1. lê o vídeo (arquivo ou URL RTSP, se passado em --rtsp-env);
2. recorta a **moldura do player** (barra de título, botões, timeline) e as
   tarjas pretas das laterais, quando existem;
3. opcionalmente recorta uma ROI (--crop x,y,w,h) para pegar só a área das
   cadeiras, aumentando a resolução efetiva das cadeiras;
4. descarta frames quase duplicados (quadro parado não ensina nada);
5. salva em ``training/datasets/raw/`` e já escreve o YAML.

Dicas de uso real:

* Grave o vídeo **com a câmera no lugar definitivo**. A contagem depende da
  altura da cadeira em pixels, que muda com a posição da câmera.
* Pegueilona: dia, noite, com e sem pessoas na frente.
* Inclua pilhas de 3 a 40 cadeiras. É a variedade de quantidade que faz o
  modelo generalizar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402


def detect_content(img: "np.ndarray", min_band: int = 18, max_std: float = 3.0) -> tuple[int, int, int, int]:
    """Encontra o conteúdo real, descartando moldura do player.

    Remove, a partir de cada borda, **faixas quase uniformes** (desvio
    padrão baixo demais para serem imagem) com pelo menos ``min_band``
    pixels. É conservador de propósito: errar por não cortar é melhor do que
    cortar parte da pilha de cadeiras.

    Na prática, a melhor opção é não usar este recorte: grave os frames
    direto da câmera (--rtsp-env) ou passe --crop com a ROI exata.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_std = gray.std(axis=1)
    col_std = gray.std(axis=0)

    def uniform_band(std: "np.ndarray", reverse: bool = False) -> int:
        """Tamanho da faixa uniforme contígua a partir da borda."""
        n = 0
        rng = range(len(std) - 1, -1, -1) if reverse else range(len(std))
        for i in rng:
            if std[i] <= max_std:
                n += 1
            else:
                break
        return n if n >= min_band else 0

    top = uniform_band(row_std)
    bottom = uniform_band(row_std, reverse=True)
    left = uniform_band(col_std)
    right = uniform_band(col_std, reverse=True)

    # Não corta mais de 25% de cada dimensão: imagem muito "chapada" indica
    # que a heurística falhou, e é melhor não cortar nada.
    if top > h * 0.25 or bottom > h * 0.25 or left > w * 0.25 or right > w * 0.25:
        return 0, 0, w, h
    return left, top, max(1, w - left - right), max(1, h - top - bottom)


def parse_crop(spec: str) -> tuple[int, int, int, int] | None:
    if not spec.strip():
        return None
    parts = [int(v) for v in spec.replace(" ", "").split(",")[:4]]
    if len(parts) != 4:
        raise SystemExit("--crop deve ser x,y,w,h (em pixels)")
    return parts[0], parts[1], parts[2], parts[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extrai frames de vídeo para o dataset")
    p.add_argument("--video", default="", help="Arquivo de vídeo local")
    p.add_argument("--rtsp-env", default="", help="Variável de ambiente com a URL RTSP")
    p.add_argument("--seconds", type=float, default=0,
                   help="Duração da captura em SEGUNDOS de relógio (0 = vídeo inteiro)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Teto de imagens salvas (0 = sem teto). Use com --seconds como rede de segurança")
    p.add_argument("--fps", type=float, default=1.0, help="Frames por segundo a extrair")
    p.add_argument("--crop", default="", help="ROI manual x,y,w,h (opcional)")
    p.add_argument("--out", default="", help="Destino (padrão: training/datasets/raw)")
    p.add_argument("--prefix", default="frame", help="Prefixo do nome do arquivo")
    p.add_argument("--no-trim", action="store_true", help="Não remover a moldura do player")
    p.add_argument("--min-diff", type=float, default=3.0,
                   help="Diferença mínima entre frames (0-100) para não salvar duplicatas")
    args = p.parse_args(argv)

    import os

    source = args.video
    if args.rtsp_env:
        # Primeiro o ambiente real; depois o .env, que o pydantic já leu.
        # Assim `python -m tools.extract_frames --rtsp-env CAMERA_RTSP_URL`
        # funciona mesmo sem exportar a variável no shell.
        source = os.environ.get(args.rtsp_env, "")
        if not source and args.rtsp_env.upper() == "CAMERA_RTSP_URL":
            source = settings.camera.camera_rtsp_url
    if not source:
        print("Informe --video ARQUIVO.mp4 ou --rtsp-env NOME_DA_VARIAVEL")
        print("  Ex.: python -m tools.extract_frames --video gravacao.mp4")
        print("  Ex.: python -m tools.extract_frames --rtsp-env CAMERA_RTSP_URL")
        return 2
    if source.startswith("rtsp://") and "@" in source:
        # Nunca imprime a senha, mesmo em log de erro.
        from app.config import mask_url

        print("(credenciais omitidas na saída)")
        source_for_msg = mask_url(source)
    else:
        source_for_msg = source

    out_dir = Path(args.out) if args.out else settings.source_images_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Não consegui abrir: {source_for_msg}")
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(src_fps / max(0.05, args.fps))))

    print("=" * 68)
    print("EXTRAÇÃO DE FRAMES")
    print("=" * 68)
    print(f"Fonte:     {source_for_msg}")
    print(f"Resolução: {w}x{h} @ {src_fps:.1f} fps ({total} frames)")
    limite = f"{args.seconds:.0f}s de relógio" if args.seconds > 0 else "vídeo inteiro"
    teto = f", teto de {args.max_frames} imagens" if args.max_frames else ""
    print(f"Extraindo: 1 a cada {step} frames ({args.fps} fps) durante {limite}{teto}")
    print(f"Destino:   {out_dir}")
    print("=" * 68)

    crop_manual = parse_crop(args.crop)
    prev_gray: np.ndarray | None = None
    saved = 0
    skipped_dup = 0
    idx = 0
    import time

    t0 = time.time()
    # --seconds é tempo de RELÓGIO: com a cena parada, quase todo frame vira
    # duplicata e o limite de quantidade nunca seria alcançado.
    deadline = t0 + args.seconds if args.seconds > 0 else None

    while True:
        if deadline is not None and time.time() > deadline:
            break
        if args.max_frames and saved >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue

        # recorte
        if crop_manual:
            x, y, cw, ch = crop_manual
            fh, fw = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            frame = frame[y : min(fh, y + ch), x : min(fw, x + cw)]
        elif not args.no_trim:
            x, y, cw, ch = detect_content(frame)
            if (x, y, cw, ch) != (0, 0, w, h):
                frame = frame[y : y + ch, x : x + cw]

        if frame is None or frame.size == 0 or min(frame.shape[:2]) < 32:
            idx += 1
            continue

        # descarta quase-duplicados
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and gray.shape == prev_gray.shape:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if diff < args.min_diff:
                skipped_dup += 1
                idx += 1
                continue
        prev_gray = gray

        name = f"{args.prefix}_{saved:05d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1
        if saved % 50 == 0:
            print(f"  {saved} frames salvos…")
        idx += 1

    cap.release()
    dt = time.time() - t0
    print("-" * 68)
    print(f"Salvos:    {saved} imagens em {out_dir}")
    print(f"Descartados por repetição: {skipped_dup}")
    print(f"Tempo:     {dt:.1f}s de relógio")
    if saved and skipped_dup > saved * 3:
        print()
        print("AVISO: a cena ficou parada — quase todos os frames eram iguais.")
        print("  Para o dataset variar, mova as cadeiras enquanto grava, ou grave em")
        print("  horários diferentes (dia/noite). Frames repetidos não ensinam nada.")
    if saved:
        print(f"Resolução final: {frame.shape[1]}x{frame.shape[0]}")
    print("=" * 68)
    if saved == 0:
        print("Nenhuma imagem extraída. Tente --fps maior ou --no-trim.")
        return 1
    print()
    print("PRÓXIMOS PASSOS:")
    print("  1. Confira as imagens: .venv/bin/python -m app.main --image "
          f"{out_dir}/{args.prefix}_00000.jpg")
    print(f"  2. Anote (CVAT/Label Studio). {saved} imagens ainda é pouco se forem")
    print("     todas do mesmo segundo: grave mais vídeos de situações diferentes.")
    print(f"  3. Divida: aba Dataset -> Dividir dataset (seed 42)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
