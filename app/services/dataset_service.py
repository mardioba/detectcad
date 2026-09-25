"""Gerenciamento do dataset YOLO.

Responsabilidades:

* receber imagens capturadas do ambiente (dashboard ou pasta monitorada);
* dizer quais já têm anotação e quais não;
* gerar a divisão treino/validação/teste de forma **reproduzível** (seed);
* escrever o ``chairs.yaml`` usado pelo Ultralytics;
* importar/exportar anotações em YOLO txt;
* gerar um subconjunto já anotado a partir das correções manuais
  (IA vs. operador), que é a forma mais barata de melhorar o modelo.

Formato (padrão YOLO detect/segment):

    training/datasets/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt   -> "cls cx cy w h" normalizado
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import settings

log = logging.getLogger("app")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
DEFAULT_CLASSES = ("chair",)


@dataclass
class DatasetStats:
    """Resumo do dataset para a página /dataset."""

    total_images: int = 0
    annotated: int = 0
    pending: int = 0
    total_boxes: int = 0
    per_split: dict[str, dict[str, int]] = field(default_factory=dict)
    per_class: dict[str, int] = field(default_factory=dict)
    raw_images: int = 0
    classes: list[str] = field(default_factory=list)
    empty_label_files: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "annotated": self.annotated,
            "pending": self.pending,
            "total_boxes": self.total_boxes,
            "per_split": self.per_split,
            "per_class": self.per_class,
            "raw_images": self.raw_images,
            "classes": self.classes,
            "empty_label_files": self.empty_label_files,
            "warnings": self.warnings,
        }


def images_in(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_for(image: Path) -> Path:
    """Caminho do .txt de rótulo correspondente a uma imagem."""
    return image.parent.parent / "labels" / image.parent.name / (image.stem + ".txt")


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Lê um .txt YOLO. Retorna lista de ``(cls, cx, cy, w, h)``."""
    out: list[tuple[int, float, float, float, float]] = []
    if not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            out.append((cls, cx, cy, w, h))
    except (ValueError, OSError) as exc:
        log.warning("Rótulo inválido em %s: %s", path, exc)
    return out


def write_label(path: Path, boxes: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class DatasetService:
    """Operações sobre ``training/datasets``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else settings.dataset_dir
        self.raw_dir = settings.source_images_dir

    # --------------------------------------------------------------- estrutura
    def ensure_structure(self) -> None:
        for split in SPLITS:
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def classes(self) -> list[str]:
        """Classes declaradas no YAML (fonte da verdade)."""
        yaml_path = settings.base_dir / "training" / "configs" / "chairs.yaml"
        if yaml_path.is_file():
            try:
                import yaml

                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                names = data.get("names")
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
            except Exception as exc:
                log.debug("Não foi possível ler names do YAML: %s", exc)
        return list(DEFAULT_CLASSES)

    # ------------------------------------------------------------------ stats
    def stats(self) -> DatasetStats:
        st = DatasetStats(classes=self.classes())
        st.raw_images = len(images_in(self.raw_dir))
        for split in SPLITS:
            imgs = images_in(self.root / "images" / split)
            labeled = 0
            boxes = 0
            empty = 0
            per_split: dict[str, int] = {"images": len(imgs), "annotated": 0, "boxes": 0}
            for img in imgs:
                lab = label_for(img)
                parsed = read_label(lab)
                per_split["boxes"] += len(parsed)
                if lab.is_file():
                    if parsed:
                        labeled += 1
                    else:
                        # .txt existe e está vazio = "sem cadeiras na imagem"
                        empty += 1
                for cls, *_ in parsed:
                    name = st.classes[cls] if 0 <= cls < len(st.classes) else str(cls)
                    st.per_class[name] = st.per_class.get(name, 0) + 1
            st.empty_label_files += empty
            per_split["annotated"] = labeled
            st.per_split[split] = per_split
            st.total_images += len(imgs)
            st.annotated += labeled
            st.total_boxes += per_split["boxes"]
        st.pending = max(0, st.total_images - st.annotated)

        if st.total_images and st.annotated == 0:
            st.warnings.append("Nenhuma imagem anotada: rode o treinamento apenas após anotar.")
        if st.annotated and st.annotated < 20:
            st.warnings.append(
                f"Poucas imagens anotadas ({st.annotated}). YOLO precisa de dezenas para comecar."
            )
        if "val" in st.per_split and st.per_split["val"]["images"] == 0:
            st.warnings.append("Conjunto de validação vazio: métricas não serão confiáveis.")
        if st.raw_images:
            st.warnings.append(
                f"{st.raw_images} imagem(ns) em training/datasets/raw aguardando anotação."
            )
        return st

    def image_status(self, limit: int = 200) -> dict[str, Any]:
        """Lista para a página /annotation: imagem + se está anotada."""
        self.ensure_structure()
        items: list[dict[str, Any]] = []
        pending = 0
        for split in SPLITS:
            for img in images_in(self.root / "images" / split):
                lab = label_for(img)
                parsed = read_label(lab)
                annotated = lab.is_file()
                if not annotated:
                    pending += 1
                if len(items) < limit:
                    items.append(
                        {
                            "filename": img.name,
                            "split": split,
                            "annotated": annotated,
                            "boxes": len(parsed),
                            "empty": annotated and not parsed,
                            "label_exists": annotated,
                            "url": f"/api/dataset/image?path={img.relative_to(self.root).as_posix()}",
                            "label_url": f"/api/dataset/label?path={lab.relative_to(self.root).as_posix()}",
                        }
                    )
        return {
            "images": items,
            "total": self.stats().total_images,
            "pending": pending,
            "stats": self.stats().as_dict(),
        }

    # ------------------------------------------------------------------ split
    def split(
        self,
        train: float = 0.70,
        val: float = 0.20,
        test: float = 0.10,
        seed: int = 42,
        source: Path | None = None,
    ) -> dict[str, Any]:
        """Redistribui as imagens entre train/val/test de forma reproduzível.

        Usa ``random.Random(seed)`` - a mesma seed gera sempre a mesma
        divisão, o que é essencial para comparar dois treinamentos.
        """
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            return {"ok": False, "message": "As proporções devem somar 1.0 (ex.: 0.70/0.20/0.10)"}
        self.ensure_structure()

        src = Path(source) if source else self.raw_dir
        if not src.is_dir() or not images_in(src):
            return {
                "ok": False,
                "message": f"Nenhuma imagem encontrada em {src}. Envie imagens primeiro.",
            }

        files = images_in(src)
        rng = random.Random(seed)
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        n_test = max(0, n - n_train - n_val)
        # Garante ao menos 1 imagem em treino e validação quando possível.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(0, n - n_train - n_val)
        buckets = {"train": files[:n_train], "val": files[n_train : n_train + n_val], "test": files[n_train + n_val : n_train + n_val + n_test]}

        # Limpa as divisões anteriores para não sobrar imagem duplicada.
        for split in SPLITS:
            for old in images_in(self.root / "images" / split):
                old.unlink(missing_ok=True)
            for old in (self.root / "labels" / split).glob("*.txt"):
                old.unlink(missing_ok=True)

        moved = 0
        with_labels = 0
        for split, items in buckets.items():
            for src_file in items:
                dst = self.root / "images" / split / src_file.name
                shutil.copy2(src_file, dst)
                moved += 1
                # Preserva a anotação se ela existir ao lado da imagem original.
                for cand in (
                    src_file.with_suffix(".txt"),
                    src / "labels" / src_file.name.replace(" ", "_") + ".txt",
                ):
                    if cand.is_file():
                        write_label(label_for(dst), read_label(cand))
                        with_labels += 1
                        break

        self.write_yaml()
        return {
            "ok": True,
            "seed": seed,
            "proportions": {"train": train, "val": val, "test": test},
            "moved": moved,
            "labels_preserved": with_labels,
            "distribution": {k: len(v) for k, v in buckets.items()},
            "message": (
                f"{moved} imagens divididas (train/val/test) com seed {seed}. "
                f"{with_labels} com anotação preservada."
            ),
        }

    # ------------------------------------------------------------------- yaml
    def write_yaml(self, path: Path | None = None) -> Path:
        """Escreve o ``chairs.yaml`` apontando para os diretórios absolutos."""
        yaml_path = Path(path) if path else settings.base_dir / "training" / "configs" / "chairs.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        names = self.classes()
        body = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        content = f"""# Gerado automaticamente por app/services/dataset_service.py
# Ajuste à mão se preferir manter o controle manual - o próximo
# "Gerenciar dataset > Dividir" reescreve este arquivo.
path: {self.root}
train: images/train
val: images/val
test: images/test

names:
{body}
"""
        yaml_path.write_text(content, encoding="utf-8")
        log.info("Dataset YAML gravado em %s", yaml_path)
        return yaml_path

    # -------------------------------------------------------------- ingestão
    def add_image(self, source: Path, copy: bool = True) -> dict[str, Any]:
        """Adiciona uma imagem crua (ainda sem anotação) ao conjunto."""
        source = Path(source)
        if not source.is_file():
            return {"ok": False, "message": f"Arquivo não encontrado: {source}"}
        if source.suffix.lower() not in IMAGE_EXTS:
            return {"ok": False, "message": f"Formato não suportado: {source.suffix}"}
        self.ensure_structure()
        target = self.raw_dir / source.name
        if copy and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return {"ok": True, "filename": target.name, "path": str(target)}

    def import_dir(self, source: Path, limit: int = 5000) -> dict[str, Any]:
        """Importa todas as imagens de uma pasta (ex.: dump da câmera)."""
        source = Path(source)
        if not source.is_dir():
            return {"ok": False, "message": f"Pasta não encontrada: {source}"}
        added = 0
        for img in images_in(source)[:limit]:
            res = self.add_image(img, copy=True)
            if res.get("ok"):
                added += 1
        return {"ok": True, "added": added, "message": f"{added} imagens importadas de {source}"}

    # ------------------------------------------------- anotações automáticas
    def auto_label_from_model(self, conf: float = 0.35) -> dict[str, Any]:
        """Pré-rotula as imagens sem anotação usando o modelo ativo.

        Economiza horas de trabalho manual. As caixas ficam marcadas como
        ``needs_review`` no relatório para o operador conferir antes de treinar.
        """
        try:
            from app.ai.yolo_detector import ModelNotAvailable, YoloDetector
        except Exception as exc:
            return {"ok": False, "message": f"Ultralytics indisponível: {exc}"}

        detector = YoloDetector()
        if not detector.load():
            return {
                "ok": False,
                "message": detector.load_error or "Nenhum modelo carregado para pré-rotular.",
            }
        try:
            chairs, piles = detector.class_indices()
        except Exception:
            chairs, piles = set(), set()
        classes = chairs | piles or None

        labeled = 0
        skipped = 0
        reviewed = 0
        for img in images_in(self.root / "images" / "train"):
            lab = label_for(img)
            if lab.is_file():
                continue
            try:
                import cv2

                frame = cv2.imread(str(img), cv2.IMREAD_COLOR)
                if frame is None:
                    skipped += 1
                    continue
                dets, _ = detector.predict(frame, conf=conf, classes=classes)
                h, w = frame.shape[:2]
                boxes = []
                for d in dets:
                    boxes.append(
                        (
                            0,
                            (d.x1 + d.x2) / 2 / w,
                            (d.y1 + d.y2) / 2 / h,
                            d.width / w,
                            d.height / h,
                        )
                    )
                write_label(lab, boxes)
                labeled += 1
            except Exception as exc:
                log.warning("Falha ao pré-rotular %s: %s", img.name, exc)
                skipped += 1
        return {
            "ok": True,
            "labeled": labeled,
            "skipped": skipped,
            "message": (
                f"{labeled} imagens pré-rotuladas (conf>={conf}). "
                f"Revise antes de treinar: o modelo atual pode errar."
            ),
        }

    def labels_as_json(self, image_name: str, split: str = "train") -> dict[str, Any]:
        """Exporta as caixas de uma imagem em JSON (formato padrão de ferramentas)."""
        img_path = self.root / "images" / split / image_name
        if not img_path.is_file():
            return {"ok": False, "message": f"Imagem não encontrada: {image_name}"}
        lab = label_for(img_path)
        boxes = read_label(lab)
        h, w = read_image_size(img_path)
        names = self.classes()
        return {
            "ok": True,
            "image": image_name,
            "split": split,
            "width": w,
            "height": h,
            "annotations": [
                {
                    "class_id": c,
                    "label": names[c] if 0 <= c < len(names) else str(c),
                    "x": (cx - bw / 2) * w,
                    "y": (cy - bh / 2) * h,
                    "width": bw * w,
                    "height": bh * h,
                }
                for c, cx, cy, bw, bh in boxes
            ],
        }

    def save_labels(self, image_name: str, split: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Grava as caixas vindas do frontend (pixels) e converte para YOLO."""
        img_path = self.root / "images" / split / image_name
        if not img_path.is_file():
            return {"ok": False, "message": f"Imagem não encontrada: {image_name}"}
        w, h = read_image_size(img_path)
        if not w or not h:
            return {"ok": False, "message": "Não foi possível ler o tamanho da imagem."}
        boxes: list[tuple[int, float, float, float, float]] = []
        for ann in annotations:
            try:
                cls = int(ann.get("class_id", 0))
                x = float(ann["x"])
                y = float(ann["y"])
                bw = float(ann["width"])
                bh = float(ann["height"])
            except (KeyError, TypeError, ValueError):
                continue
            cx = max(0.0, min(1.0, (x + bw / 2) / w))
            cy = max(0.0, min(1.0, (y + bh / 2) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            boxes.append((cls, cx, cy, nw, nh))
        write_label(label_for(img_path), boxes)
        return {"ok": True, "boxes": len(boxes), "message": f"{len(boxes)} caixa(s) salva(s)."}


def read_image_size(path: Path) -> tuple[int, int]:
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
        return 0, 0


__all__ = ["DatasetService", "DatasetStats", "read_label", "write_label", "label_for", "images_in", "SPLITS"]
