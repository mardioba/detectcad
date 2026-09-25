"""Prepara o dataset YOLO a partir de anotações de ferramentas externas.

Entrada aceita:

* ``--from-cvat export.xml``       - exportação do CVAT em XML
* ``--from-labelstudio export.json`` - exportação do Label Studio
* ``--copy-only pasta``            - só copia imagens (sem rótulo)

Depois, use::

    python -m app.main            # e a aba Dataset → Dividir

O script:
1. converte as anotações para o formato YOLO (um .txt por imagem);
2. copia/associas as imagens em ``training/datasets/raw``;
3. mostra um relatório do que foi convertido e do que ficou de fora.

Sobre classes: o mapeamento é por NOME. As classes configuradas em
``training/configs/chairs.yaml`` são detectadas automaticamente.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.dataset_service import DatasetService, read_label  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def detect_class_names(labels: set[str]) -> list[str]:
    """Escolhe os nomes de classe, priorizando 'chair' e 'pile'."""
    from app.services.dataset_service import DEFAULT_CLASSES

    preferred = [c for c in ("chair", "pile") if c in labels]
    if preferred:
        rest = sorted(labels - set(preferred))
        return preferred + rest
    if labels:
        return sorted(labels)
    return list(DEFAULT_CLASSES)


def convert_cvat(xml_path: Path, raw_dir: Path, names: list[str] | None) -> dict[str, Any]:
    """Converte a exportação XML do CVAT ('CVAT for images 1.1')."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    raw_dir.mkdir(parents=True, exist_ok=True)

    # mapa label_id -> nome
    label_map: dict[str, str] = {}
    for meta in root.findall(".//meta"):
        labels = meta.find("labels")
        if labels is not None:
            for lab in labels.findall("label"):
                if lab.get("name"):
                    label_map[lab.get("name", "")] = lab.get("name", "")
    for lab in root.findall(".//label"):
        if lab.get("name"):
            label_map[lab.text or lab.get("name", "")] = lab.get("name", "")

    # imagens do dump do CVAT
    image_dir = root.find(".//images")
    copied = 0
    for img_meta in (image_dir.findall("image") if image_dir is not None else []):
        src = Path(img_meta.get("name", ""))
        if not src.is_absolute():
            src = xml_path.parent / src
        if not src.is_file():
            continue
        shutil.copy2(src, raw_dir / src.name)
        copied += 1

    # caixas
    boxes_by_image: dict[str, list[list[float]]] = {}
    used_labels: set[str] = set()
    for box in root.findall(".//box"):
        img = box.get("image")
        label = label_map.get(box.get("label", ""), box.get("label", ""))
        if not img or not label:
            continue
        pts = box.get("pts")
        if not pts:
            continue
        try:
            xs, ys = [], []
            for pair in pts.split(";"):
                px, py = pair.split(",")
                xs.append(float(px))
                ys.append(float(py))
        except ValueError:
            continue
        if not xs or not ys:
            continue
        boxes_by_image.setdefault(img, []).append(
            [min(xs), min(ys), max(xs), max(ys), label]
        )
        used_labels.add(label)

    return _write_labels(boxes_by_image, raw_dir, names, used_labels, copied, "CVAT")


def convert_labelstudio(json_path: Path, raw_dir: Path, names: list[str] | None) -> dict[str, Any]:
    """Converte a exportação JSON do Label Studio."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]
    raw_dir.mkdir(parents=True, exist_ok=True)

    used_labels: set[str] = set()
    boxes_by_image: dict[str, list[list[float]]] = {}
    copied = 0
    for task in data:
        name = None
        for k in ("file_upload", "image", "imagePath"):
            v = (task.get("data") or {}).get(k)
            if isinstance(v, str) and v:
                name = Path(v).name
                break
        if not name:
            continue
        # copia a imagem se existir ao lado do json
        for base in (json_path.parent, json_path.parent / "upload"):
            cand = base / name
            if cand.is_file():
                shutil.copy2(cand, raw_dir / name)
                copied += 1
                break
        else:
            cand = raw_dir / name
            if not cand.is_file():
                continue

        for ann in task.get("annotations") or []:
            for res in ann.get("result") or []:
                if res.get("type") != "rectanglelabels" and "rectanglelabels" not in res.get("result", {}):
                    pass
                labels = res.get("value", {}).get("rectanglelabels") or []
                label = labels[0] if labels else "chair"
                used_labels.add(label)
                x = float(res["value"].get("x", 0))
                y = float(res["value"].get("y", 0))
                w = float(res["value"].get("width", 0))
                h = float(res["value"].get("height", 0))
                if w <= 0 or h <= 0:
                    continue
                boxes_by_image.setdefault(name, []).append([x, y, x + w, y + h, label])

    return _write_labels(boxes_by_image, raw_dir, names, used_labels, copied, "Label Studio")


def _write_labels(
    boxes_by_image: dict[str, list[list[float]]],
    raw_dir: Path,
    names: list[str] | None,
    used_labels: set[str],
    copied: int,
    tool: str,
) -> dict[str, Any]:
    """Grava os .txt YOLO ao lado das imagens em raw/."""
    final_names = names or detect_class_names(used_labels)
    name_to_id = {n: i for i, n in enumerate(final_names)}
    labels_dir = raw_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    boxes = 0
    unknown: set[str] = set()
    for image_name, items in boxes_by_image.items():
        path = raw_dir / image_name
        if not path.is_file():
            continue
        img = _read_size(path)
        if not img:
            continue
        w, h = img
        lines: list[str] = []
        for x1, y1, x2, y2, label in items:
            if label not in name_to_id:
                unknown.add(label)
                continue
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            lines.append(f"{name_to_id[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            boxes += 1
        (labels_dir / (path.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    return {
        "tool": tool,
        "images_copied": copied,
        "label_files": written,
        "boxes": boxes,
        "classes": final_names,
        "unknown_labels": sorted(unknown),
    }


def _read_size(path: Path) -> tuple[int, int] | None:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img.shape[1], img.shape[0]
    except Exception:
        pass
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def copy_only(folder: Path, raw_dir: Path) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(f, raw_dir / f.name)
            n += 1
    return {"tool": "copy-only", "images_copied": n, "label_files": 0, "boxes": 0,
            "classes": [], "unknown_labels": []}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepara o dataset YOLO")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-cvat", help="XML exportado pelo CVAT")
    src.add_argument("--from-labelstudio", help="JSON exportado pelo Label Studio")
    src.add_argument("--copy-only", help="Pasta de imagens para copiar sem rótulo")
    p.add_argument("--out", default="", help="Pasta de destino (padrão: training/datasets/raw)")
    p.add_argument("--names", default="", help="Classes separadas por vírgula (ex.: chair,pile)")
    args = p.parse_args(argv)

    raw_dir = Path(args.out) if args.out else settings.source_images_dir
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None

    if args.from_cvat:
        result = convert_cvat(Path(args.from_cvat), raw_dir, names)
    elif args.from_labelstudio:
        result = convert_labelstudio(Path(args.from_labelstudio), raw_dir, names)
    else:
        result = copy_only(Path(args.copy_only), raw_dir)

    print("=" * 60)
    print("CONVERSÃO CONCLUÍDA")
    print("=" * 60)
    print(f"  Ferramenta:     {result['tool']}")
    print(f"  Imagens:        {result['images_copied']}")
    print(f"  Arquivos .txt:  {result['label_files']}")
    print(f"  Caixas:         {result['boxes']}")
    print(f"  Classes:        {', '.join(result['classes']) or '(nenhuma)'}")
    if result["unknown_labels"]:
        print(f"  Rótulos ignorados (não estão nas classes): {', '.join(result['unknown_labels'])}")
    print(f"  Destino:        {raw_dir}")
    print("=" * 60)
    print()
    print("PRÓXIMOS PASSOS:")
    print("  1. Confira o YAML de classes: training/configs/chairs.yaml")
    print("  2. Abra o dashboard → aba Dataset → Dividir dataset (com seed)")
    print("  3. Treine: aba Treinamento → INICIAR TREINAMENTO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
