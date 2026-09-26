"""Exporta o modelo treinado para outros formatos.

    python -m training.scripts.export
    python -m training.scripts.export --format onnx --imgsz 640

Formatos: onnx (CPU/mobile), coreml/tflite/etc. via ``--format <nome>``.

O ONNX é o mais útil: roda em qualquer máquina, inclusive sem GPU e sem
PyTorch instalado no servidor de produção.
"""

# =============================================================================
# ARQUIVO / MAPA  -  training/scripts/export.py
#
# O que faz: converte o best.pt (.pt do PyTorch) em outro formato (onnx,
#   coreml, tflite...). É OPCIONAL: o sistema atual roda o .pt direto.
#   Serve para quem precisa rodar em máquina sem PyTorch, ou quer NVIDIA
#   TensorRT (engine) para inference mais rápida na GPU.
#
# Ordem de leitura:
#   1. find_best ..... qual arquivo exportar (--weights, run mais recente, .env)
#   2. main() ........ CLI -> export() -> moves o arquivo -> mostra o tamanho
#
# Aviso: exportar não melhora o modelo. Mesmos pesos, outro formato.
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def find_best() -> Path | None:
    """Escolhe o que exportar quando --weights não foi informado."""
    runs = ROOT / settings.training.project
    cands = [
        d / "weights" / "best.pt"
        for d in runs.glob("*")
        if d.is_dir() and (d / "weights" / "best.pt").is_file()
    ]
    if cands:
        # O mais recente por mtime. Ver validate.py: "mais recente" não é
        # necessariamente "melhor".
        return max(cands, key=lambda x: x.stat().st_mtime)
    # Sem runs, tenta o modelo do .env (pode ser o genérico yolo11n.pt, que
    # exporta, mas não serve para contar pilhas).
    if settings.model_path.is_file():
        return settings.model_path
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exporta um modelo YOLO")
    p.add_argument("--weights", default="")
    p.add_argument("--format", default="onnx", help="onnx, coreml, tflite, engine, saved_model")
    p.add_argument("--imgsz", type=int, default=settings.training.image_size)
    p.add_argument("--half", action="store_true", help="FP16 (somente GPU)")
    p.add_argument("--out", default="", help="Arquivo de saída")
    args = p.parse_args(argv)

    weights = Path(args.weights) if args.weights else find_best()
    if weights is None or not Path(weights).is_file():
        print("Nenhum best.pt encontrado. Treine primeiro:")
        print("  python -m training.scripts.train")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics indisponível: {exc}")
        return 1

    print(f"Exportando {weights} -> {args.format} (imgsz={args.imgsz})")
    print("Isso pode demorar e baixar dependências extras (onnx, onnxslim...).")
    model = YOLO(str(weights))
    try:
        out = model.export(
            format=args.format,
            # imgsz fica FIXADO na exportação: um ONNX 640 não aceita entrada
            # 1280 depois. Se a câmera mudar de resolução, reexporte.
            imgsz=args.imgsz,
            # FP16 só faz sentido em GPU: em CPU não há ganho e ainda quebra
            # alguns runtimes. Por isso o `and` com o device resolvido.
            half=args.half and settings.resolve_device().startswith("cuda"),
            # simplify passa o grafo pelo onnxslim: arquivo menor e mais rápido.
            simplify=True,
        )
    except Exception as exc:
        print(f"Falha na exportação: {exc}")
        print()
        print("Dicas:")
        print("  pip install onnx onnxruntime onnxslim")
        print("  ou exporte apenas 'onnx' que é o caso mais útil.")
        return 1

    # O Ultralytics devolve o caminho e sempre escreve do lado do .pt.
    dest = Path(str(out))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        # replace() move o arquivo; se for o mesmo caminho, não faz nada
        # (substituir por si mesmo levantaria erro).
        dest.replace(target) if dest != target else None
        dest = target
    print(f"\nExportado com sucesso: {dest}")
    print(f"Tamanho: {dest.stat().st_size / 1024**2:.1f} MB")
    print()
    print("Para usar o ONNX em produção seria preciso trocar o backend de")
    print("inferência (Ultralytics já suporta, ou use OpenCV DNN). O sistema")
    print("atual usa o .pt do PyTorch — a exportação é opcional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
