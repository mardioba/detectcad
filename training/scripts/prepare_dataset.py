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

# =============================================================================
# ARQUIVO / MAPA  -  training/scripts/prepare_dataset.py
#
# O que faz: traduz anotações feitas em ferramentas externas (CVAT, Label
#   Studio) para o formato YOLO, que é apenas um .txt por imagem. Nada de
#   treinar aqui: este script só prepara o terreno.
#
# Ordem de leitura:
#   1. detect_class_names .. escolhe a ordem das classes (a ordem define o id)
#   2. convert_cvat ........ XML do CVAT -> caixas em coordenadas de pixel
#   3. convert_labelstudio . JSON do Label Studio -> idem
#   4. _write_labels ....... pixel -> normalizado 0..1 e grava o .txt
#   5. copy_only ........... só copia imagens (quando ainda não há anotação)
#   6. main() .............. CLI + relatório do que entrou e do que ficou de fora
#
# Saída: training/datasets/raw/ (configurável com --out), que é a entrada da
#   divisão treino/val/test feita na aba Dataset do dashboard.
# =============================================================================

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
    """Escolhe os nomes de classe, priorizando 'chair' e 'pile'.

    A POSIÇÃO na lista é o id da classe dentro do .txt YOLO, então ela precisa
    ser estável: mudar a ordem troca o significado das anotações antigas.
    Por isso 'chair' e 'pile' vêm sempre primeiro, nessa ordem, e o resto vai
    ordenado alfabeticamente.
    """
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

    # mapa label_id -> nome. O CVAT pode declarar as labels em dois lugares
    # (<meta><labels> e a lista de <label> solta), então os dois são lidos.
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

    # caixas: uma entrada por imagem, com as caixas em pixel (x1,y1,x2,y2,label)
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
            # "pts" do CVAT: "x1,y1;x2,y2;x3,y3;x4,y4" em pixel da imagem.
            # Qualquer ponto que não seja "px,py" é descartado aqui.
            xs, ys = [], []
            for pair in pts.split(";"):
                px, py = pair.split(",")
                xs.append(float(px))
                ys.append(float(py))
        except ValueError:
            continue
        if not xs or not ys:
            continue
        # Polígono vira retângulo envolvente: o YOLO de detecção só aceita caixa.
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
        # Cada tarefa aponta para a imagem por chaves diferentes conforme a
        # versão do Label Studio. Path(v).name tira o caminho completo.
        for k in ("file_upload", "image", "imagePath"):
            v = (task.get("data") or {}).get(k)
            if isinstance(v, str) and v:
                name = Path(v).name
                break
        if not name:
            continue
        # copia a imagem se existir ao lado do json
        # (o export do Label Studio costuma guardar em upload/ com prefixo).
        for base in (json_path.parent, json_path.parent / "upload"):
            cand = base / name
            if cand.is_file():
                shutil.copy2(cand, raw_dir / name)
                copied += 1
                break
        else:
            # O for-else só entra aqui se NENHUM caminho tinha a imagem; aí ela
            # já está em raw/ (envio anterior) e seguimos só com os rótulos.
            cand = raw_dir / name
            if not cand.is_file():
                continue

        # Vários anotadores podem marcar a mesma imagem: por isso o for aninhado
        # sobre todas as anotações e todos os resultados de cada uma.
        for ann in task.get("annotations") or []:
            for res in ann.get("result") or []:
                if res.get("type") != "rectanglelabels" and "rectanglelabels" not in res.get("result", {}):
                    pass
                # rectanglelabels é uma LISTA; este sistema tem uma classe por
                # caixa, então só a primeira etiqueta é usada.
                labels = res.get("value", {}).get("rectanglelabels") or []
                label = labels[0] if labels else "chair"
                used_labels.add(label)
                # O Label Studio dá canto superior + largura/altura em % da
                # imagem; aqui vira x1,y1,x2,y2 em pixel para unificar com o CVAT.
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
    # Se o usuário não passou --names, a ordem sai das próprias anotações.
    final_names = names or detect_class_names(used_labels)
    # nome -> id: é este dicionário que amarra a caixa ao número da 1ª coluna.
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
        # Sem as dimensões reais da imagem não dá para normalizar: pular é
        # melhor do que gravar um .txt com coordenadas inválidas.
        img = _read_size(path)
        if not img:
            continue
        w, h = img
        lines: list[str] = []
        for x1, y1, x2, y2, label in items:
            if label not in name_to_id:
                # Rótulo fora da lista de classes: conta e reporta em vez de
                # inventar um id (o treino quebraria silenciosamente depois).
                unknown.add(label)
                continue
            # Conversão de pixel para o formato YOLO: centro em fração da imagem
            # e TAMANHO também em fração (0..1), não em pixels.
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            lines.append(f"{name_to_id[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            boxes += 1
        # Nome do .txt = nome da imagem sem extensão (regra do YOLO).
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
    """Largura e altura da imagem; None se não der para abrir."""
    # OpenCV primeiro (já é dependência do projeto), PIL como reserva.
    # IMREAD_COLOR é fixado para não varies com o conteúdo do arquivo.
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
    """Só copia as imagens, sem rótulo (para anotar depois no dashboard)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(folder.iterdir()):
        # sorted() = ordem estável, o que faz a divisão com seed ser reproduzível
        # mesmo quando o --copy-only for repetido.
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(f, raw_dir / f.name)
            n += 1
    return {"tool": "copy-only", "images_copied": n, "label_files": 0, "boxes": 0,
            "classes": [], "unknown_labels": []}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepara o dataset YOLO")
    # Mutuamente exclusivo e obrigatório: as três entradas produzem formatos
    # diferentes, misturá-las não faria sentido.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-cvat", help="XML exportado pelo CVAT")
    src.add_argument("--from-labelstudio", help="JSON exportado pelo Label Studio")
    src.add_argument("--copy-only", help="Pasta de imagens para copiar sem rótulo")
    p.add_argument("--out", default="", help="Pasta de destino (padrão: training/datasets/raw)")
    p.add_argument("--names", default="", help="Classes separadas por vírgula (ex.: chair,pile)")
    args = p.parse_args(argv)

    # Sem --out usa settings.source_images_dir = training/datasets/raw, que é
    # de onde a divisão do DatasetService lê.
    raw_dir = Path(args.out) if args.out else settings.source_images_dir
    # --names fixo na mão; vazio = deduzir das anotações (detect_class_names).
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
    # Rótulo ignorado é o aviso mais importante do relatório: significa que
    # alguém anotou um nome de classe que o modelo não vai ter.
    if result["unknown_labels"]:
        print(f"  Rótulos ignorados (não estão nas classes): {', '.join(result['unknown_labels'])}")
    print(f"  Destino:        {raw_dir}")
    print("=" * 60)
    print()
    print("PRÓXIMOS PASSOS:")
    print("  1. Confira o YAML de classes: training/configs/chairs.yaml")
    # A divisão é feita aqui, e não neste script, porque ela usa a MESMA seed
    # do treino (settings.training.seed): sem isso, comparar dois modelos
    # seria comparar também conjuntos de validação diferentes.
    print("  2. Abra o dashboard → aba Dataset → Dividir dataset (com seed)")
    print("  3. Treine: aba Treinamento → INICIAR TREINAMENTO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
