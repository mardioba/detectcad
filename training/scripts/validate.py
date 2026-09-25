"""Valida um modelo treinado e mostra as métricas REAIS.

    python -m training.scripts.validate
    python -m training.scripts.validate --weights training/runs/chair_counter/weights/best.pt

Mostra Precision, Recall, mAP50 e mAP50-95 lidos do ``results.csv`` do run
(números inventados não servem para nada) e salva algumas detecções de
exemplo para inspeção visual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.model_manager import read_metrics_from_run  # noqa: E402
from app.config import settings  # noqa: E402


def find_run_dir(weights: str | None) -> Path | None:
    """Localiza o run a partir do peso ou do diretório de runs."""
    if weights:
        p = Path(weights)
        if p.parent.name == "weights":
            return p.parent.parent
        if p.is_dir():
            return p
    runs = ROOT / settings.training.project
    if not runs.is_dir():
        return None
    cands = [d for d in runs.iterdir() if d.is_dir() and (d / "weights" / "best.pt").is_file()]
    if not cands:
        return None
    return max(cands, key=lambda d: (d / "weights" / "best.pt").stat().st_mtime)


def show_metrics(run_dir: Path) -> dict:
    metrics = read_metrics_from_run(run_dir)
    print("=" * 64)
    print("MÉTRICAS REAIS DO MODELO")
    print("=" * 64)
    print(f"Run: {run_dir}")
    def fmt(v: float | None, nd: int = 4) -> str:
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

    print(f"  Épocas treinadas : {metrics.get('epochs') or '—'}")
    print(f"  Precision        : {fmt(metrics.get('precision'))}")
    print(f"  Recall           : {fmt(metrics.get('recall'))}")
    print(f"  mAP50            : {fmt(metrics.get('map50'))}")
    print(f"  mAP50-95         : {fmt(metrics.get('map5095'))}")
    if metrics.get("base_model"):
        print(f"  Modelo base      : {metrics['base_model']}")
    print("=" * 64)

    if metrics.get("precision") is None:
        print("AVISO: results.csv não encontrado. As métricas acima ficam vazias")
        print("       de propósito — o sistema não inventa números.")
    else:
        p, r = metrics.get("precision", 0), metrics.get("recall", 0)
        m = metrics.get("map50", 0)
        print("INTERPRETAÇÃO (referência, não promessa):")
        if m >= 0.9:
            print("  mAP50 alto. Confira no teste com as fotos reais do seu depósito.")
        elif m >= 0.7:
            print("  mAP50 razoável. Funciona, mas vale revisar os erros de anotação.")
        else:
            print("  mAP50 baixo. O modelo ainda NÃO deve ir para produção.")
            print("  Causas comuns: poucas imagens, anotações inconsistentes,")
            print("  imagens sem diversidade (só um ângulo/iluminação).")
        if p > 0 and r > 0 and max(p, r) / min(p, r) > 3:
            print("  Atenção: precision e recall muito diferentes — o modelo erra")
            print("  de um lado só. Revise qual tipo de erro é mais grave aqui.")
    return metrics


def run_val(model_path: Path, data_yaml: Path, imgsz: int) -> None:
    """Roda a validação do Ultralytics (métricas por split)."""
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics indisponível: {exc}")
        return
    print("\nExecutando validação do Ultralytics (pode demorar)...")
    try:
        model = YOLO(str(model_path))
        model.val(data=str(data_yaml), imgsz=imgsz, device=settings.resolve_device(), verbose=True)
    except Exception as exc:
        print(f"Falha na validação: {exc}")


def show_samples(model_path: Path, source: str | None, conf: float) -> None:
    """Salva algumas detecções para inspeção visual."""
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics indisponível: {exc}")
        return

    out_dir = settings.results_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if source:
        src = Path(source)
        if src.is_dir():
            files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:6]
        else:
            files = [src] if src.is_file() else []
    else:
        files = sorted(
            (p for p in (settings.dataset_dir / "images" / "test").glob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        )[:6]
    if not files:
        print("\nSem imagens de exemplo para validar visualmente.")
        print("  Copie fotos para training/datasets/images/test/ ou passe --source.")
        return

    model = YOLO(str(model_path))
    print(f"\nGerando detecções de exemplo em {out_dir}")
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        res = model.predict(img, conf=conf, verbose=False)[0]
        n = len(res.boxes) if res.boxes is not None else 0
        name = f"{f.stem}_pred.jpg"
        cv2.imwrite(str(out_dir / name), res.plot())
        print(f"  {f.name}: {n} detecção(ões) -> {out_dir / name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Valida um modelo YOLO de cadeiras")
    p.add_argument("--weights", default="", help="Caminho do best.pt (ou do run)")
    p.add_argument("--data", default=str(ROOT / "training" / "configs" / "chairs.yaml"))
    p.add_argument("--imgsz", type=int, default=settings.training.image_size)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--source", default="", help="Imagem ou pasta para exemplos")
    p.add_argument("--no-val", action="store_true", help="Não roda o val() do Ultralytics")
    args = p.parse_args(argv)

    run_dir = find_run_dir(args.weights or None)
    if run_dir is None:
        print("Nenhum modelo treinado encontrado.")
        print("  Treine primeiro:  python -m training.scripts.train")
        print(f"  Ou aponte --weights para o best.pt (procurado em {ROOT / settings.training.project})")
        return 1

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"best.pt não encontrado em {run_dir}")
        return 1

    show_metrics(run_dir)
    if not args.no_val:
        run_val(best, Path(args.data), args.imgsz)
    show_samples(best, args.source or None, args.conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
