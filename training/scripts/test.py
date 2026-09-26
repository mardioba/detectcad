"""Testa um modelo treinado em imagem, vídeo ou câmera RTSP.

    python -m training.scripts.test --image foto.jpg
    python -m training.scripts.test --video gravacao.mp4
    python -m training.scripts.test --rtsp rtsp://user:pass@ip:554/stream
    python -m training.scripts.test --weights training/runs/chair/weights/best.pt

Salva as imagens anotadas em ``data/results/``.

Para a URL RTSP use a variável de ambiente CAMERA_RTSP_URL (nunca no
histórico do shell); este script aceita ``--rtsp-env CAMERA_RTSP_URL``.
"""

# =============================================================================
# ARQUIVO / MAPA  -  training/scripts/test.py
#
# O que faz: rodar um modelo treinado em imagem, vídeo ou câmera RTSP e salvar
#   o resultado anotado. É o "ver com os olhos antes de colocar no ar".
#   Não mexe em banco, não sobe servidor: só lê e escreve em data/results/.
#
# Ordem de leitura:
#   1. resolve_weights .. qual arquivo abrir (--weights, run recente, .env)
#   2. test_image ...... 1 foto
#   3. test_video ...... amostra alguns frames do vídeo
#   4. test_rtsp ...... N segundos da câmera, com reconexão
#   5. main() .......... monta a URL da câmera e escolhe o modo
#
# Segurança: a URL RTSP tem usuário e senha. Prefira --rtsp-env; a URL é
#   mascarada (mask_url) antes de aparecer no terminal e nos logs.
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from app.config import mask_url, settings  # noqa: E402


def resolve_weights(weights: str) -> Path | None:
    """Achata o --weights em um caminho de arquivo .pt."""
    p = Path(weights)
    if p.is_file():
        return p
    if p.is_dir():
        # Aceita a pasta do run ou a pasta "weights/" dele.
        for cand in (p / "weights" / "best.pt", p / "best.pt"):
            if cand.is_file():
                return cand
    # procura o best.pt mais recente nos runs
    runs = ROOT / settings.training.project
    cands = [d / "weights" / "best.pt" for d in runs.glob("*") if d.is_dir() and (d / "weights" / "best.pt").is_file()]
    if cands:
        return max(cands, key=lambda x: x.stat().st_mtime)
    # Último recurso: o YOLO_MODEL do .env (pode ser o genérico pré-treinado).
    configured = settings.model_path
    return configured if configured.is_file() else None


def test_image(model, path: Path, conf: float, out_dir: Path) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Não consegui ler {path}")
        return
    res = model.predict(img, conf=conf, verbose=False)[0]
    n = len(res.boxes) if res.boxes is not None else 0
    out = out_dir / f"test_{path.stem}.jpg"
    # res.plot() devolve a imagem com as caixas desenhadas em cima.
    cv2.imwrite(str(out), res.plot())
    print(f"{path.name}: {n} detecção(ões) -> {out}")


def test_video(model, path: Path, conf: float, out_dir: Path, max_frames: int) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Não consegui abrir {path}")
        return
    # CAP_PROP_FRAME_COUNT de MP4 às vezes vem 0 ou errado; daí o fallback
    # de "passo" abaixo e do deadline implícito em max_frames.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"{path.name}: {total} frames @ {fps:.1f} fps")
    idx = 0
    saved = 0
    # Espaça as amostras ao longo do vídeo todo, em vez de olhar só o começo.
    step = max(1, total // 8) if total else 30
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0 and saved < 8:
            res = model.predict(frame, conf=conf, verbose=False)[0]
            out = out_dir / f"test_{path.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(out), res.plot())
            saved += 1
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()
    print(f"{idx} frames processados, {saved} imagem(ns) salva(s) em {out_dir}")


def test_rtsp(model, url: str, conf: float, out_dir: Path, seconds: float) -> None:
    # Impressão segura: a senha vira ****, mas o resto da URL fica visível
    # para dar pra saber qual câmera foi testada.
    print(f"Conectando em {mask_url(url)} por {seconds:.0f}s ...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return
    import time

    t0 = time.time()
    idx = 0
    saved = 0
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            # Câmera IP derruba a conexão sozinha (Wi-Fi, reinício do NVR).
            # Em produção quem reconecta é o CameraManager; aqui é um retry
            # simples para o teste não morrer no primeiro{frame perdido.
            print("Frame inválido; tentando reconectar...")
            cap.release()
            time.sleep(2.0)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                break
            continue
        if idx % 50 == 0 and saved < 6:
            res = model.predict(frame, conf=conf, verbose=False)[0]
            # O nome usa o relógio, não o índice: cada foto mostra a hora real
            # em que a câmera estava daquele jeito.
            out = out_dir / f"test_rtsp_{int(time.time())}.jpg"
            cv2.imwrite(str(out), res.plot())
            saved += 1
        idx += 1
    cap.release()
    print(f"{idx} frames lidos, {saved} imagem(ns) salva(s) em {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Testa um modelo YOLO de cadeiras")
    p.add_argument("--weights", default="", help="best.pt (padrão: procura nos runs)")
    p.add_argument("--image", default="")
    p.add_argument("--video", default="")
    p.add_argument("--rtsp", default="")
    p.add_argument("--rtsp-env", default="", help="Nome da variável com a URL RTSP")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--seconds", type=float, default=20.0, help="Duração do teste RTSP")
    p.add_argument("--max-frames", type=int, default=0, help="Limite de frames do vídeo")
    p.add_argument("--source-dir", default="", help="Testa todas as imagens de uma pasta")
    args = p.parse_args(argv)

    weights = resolve_weights(args.weights)
    if weights is None:
        print("Nenhum modelo encontrado.")
        print("  Treine primeiro (python -m training.scripts.train) ou")
        print("  aponte --weights para o best.pt.")
        return 1
    print(f"Modelo: {weights}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics indisponível: {exc}")
        return 1
    model = YOLO(str(weights))

    out_dir = settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precedência da URL: --rtsp-env (nome de variável) > --rtsp > .env.
    # O --rtsp-env existe para a senha NÃO ficar no histórico do shell.
    url = args.rtsp
    if args.rtsp_env:
        url = os.environ.get(args.rtsp_env, "")
    if not url and not args.image and not args.video:
        url = settings.camera.camera_rtsp_url

    did = False
    if args.image:
        test_image(model, Path(args.image), args.conf, out_dir)
        did = True
    if args.video:
        test_video(model, Path(args.video), args.conf, out_dir, args.max_frames)
        did = True
    if args.source_dir:
        for f in sorted(Path(args.source_dir).glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                test_image(model, f, args.conf, out_dir)
                did = True
    # RTSP só por último: é o modo mais lento e o único que depende da rede.
    if not did and url:
        test_rtsp(model, url, args.conf, out_dir, args.seconds)
        did = True
    if not did:
        print("\nNada para testar. Use --image, --video, --source-dir ou --rtsp.")
        return 2  # 2 = erro de uso (nada a fazer), diferente de "falhou"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
