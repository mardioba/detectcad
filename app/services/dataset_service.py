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

# ARQUIVO / MAPA
# Dataset YOLO. O mais "chato" dos services: quase tudo aqui é manipulação
# de arquivo e formato, e o valor de um bug é silencioso (o treino sai ruim e
# ninguém sabe por quê).
#
# Ordem de leitura:
#   1. DatasetStats         -> o resumo que a página /dataset mostra
#   2. images_in / label_for / read_label / write_label -> o contrato YOLO
#   3. ensure_structure / classes / stats -> inspecionar
#   4. image_status         -> a fila de anotação
#   5. split                 -> treinar/validação/teste reproduzível
#   6. write_yaml            -> o chairs.yaml que o Ultralytics lê
#   7. add_image / import_dir -> entrada de imagens
#   8. auto_label_from_model -> pré-rotulagem com o modelo atual
#   9. labels_as_json / save_labels -> ponte com o frontend de anotação
#
# Duas regras de ouro:
#   * NÃO renomeie/remova imagens manualmente. split() apaga as divisões
#     antigas antes de recopiar.
#   * "labels" é irmão de "images", com o MESMO nome do split e do arquivo.
#     Essa simetria é o que label_for() explora.

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
# Ordem de SPLITS importa para leitura, não para o treino. A ordem de leitura
# das estatísticas é train, val, test.
SPLITS = ("train", "val", "test")
# Fallback quando o chairs.yaml não existe ou não tem `names`. Fica alinhado
# com o único classe que o projeto realmente treina hoje.
DEFAULT_CLASSES = ("chair",)


@dataclass
class DatasetStats:
    """Resumo do dataset para a página /dataset.

    Distinção que costuma confundir:
    * ``annotated``  = tem .txt COM pelo menos uma caixa;
    * ``pending``    = não tem .txt nenhum (falta anotar);
    * ``empty_label_files`` = tem .txt VAZIO, que é informação valiosa: é o
      operador dizendo "nesta imagem não há cadeira". Some de annotated e
      também não entra em pending, senão ele re-anotaria a mesma imagem para
      sempre.
    """

    total_images: int = 0
    annotated: int = 0
    pending: int = 0
    # Total de caixas (não de imagens): é o número que diz se o modelo tem
    # material suficiente.
    total_boxes: int = 0
    per_split: dict[str, dict[str, int]] = field(default_factory=dict)
    per_class: dict[str, int] = field(default_factory=dict)
    # Imagens em training/datasets/raw, fora de qualquer split. Só aqui o
    # operador pode renomear/anotar antes de dividir.
    raw_images: int = 0
    classes: list[str] = field(default_factory=list)
    empty_label_files: int = 0
    # Avisos legíveis já prontos para a tela: a camada de decisão (o que é
    # bloqueio vs. o que é só sugestão) fica no stats(), não no frontend.
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
    """Lista as imagens de uma pasta, ordenadas, ignorando subpastas."""
    if not path.is_dir():
        return []
    # is_file() e não só sufixo: sem isso, uma pasta "backup.jpg/" entraria
    # na contagem. sorted() garante ordem estável entre execuções.
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_for(image: Path) -> Path:
    """Caminho do .txt de rótulo correspondente a uma imagem.

    É a convenção do Ultralytics:
        images/train/foto.jpg  ->  labels/train/foto.txt
    por isso o dois níveis para cima (sai de train/ para images/) e a troca de
    "images" por "labels".
    """
    return image.parent.parent / "labels" / image.parent.name / (image.stem + ".txt")


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Lê um .txt YOLO. Retorna lista de ``(cls, cx, cy, w, h)``.

    Tolerante a arquivo sujo: linha vazia, comentário com ``#`` e linha
    truncada são puladas em vez de estourar. Um .txt com erro não pode
    derrubar a página inteira de estatísticas.
    """
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
                # Linha pela metade: em vez de falhar, a caixa se perde. O
                # chamador percebe depois por annotated/boxes divergindo.
                continue
            # int(float(...)) aceita "0.0" além de "0": Ultralytics às vezes
            # grava assim, e não vale perder a anotação por isso.
            cls = int(float(parts[0]))
            # cx, cy, w, h NORMALIZADOS (0..1), não pixels. Conversão para
            # pixel é responsabilidade de quem chama.
            cx, cy, w, h = (float(v) for v in parts[1:5])
            out.append((cls, cx, cy, w, h))
    except (ValueError, OSError) as exc:
        # devolve o que conseguiu ler antes do erro
        log.warning("Rótulo inválido em %s: %s", path, exc)
    return out


def write_label(path: Path, boxes: Iterable[tuple[int, float, float, float, float]]) -> None:
    """Grava caixas YOLO normalizadas, criando as pastas se preciso."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 6 casas decimais: bem acima do necessário para resolução de imagem, e
    # legível quando alguém for abrir o .txt num editor de texto.
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    # O "\n" só quando há linhas: um arquivo sem nenhuma byte é lido como
    # "vazio" por read_label, que é o que queremos dizer com "sem cadeiras".
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class DatasetService:
    """Operações sobre ``training/datasets``."""

    def __init__(self, root: Path | None = None) -> None:
        # root injetável para os testes poderem usar um tmp_path.
        self.root = Path(root) if root else settings.dataset_dir
        # raw_dir NÃO fica sob root de propósito: é o inbox de imagens
        # soltas, e o split() move (copia) daqui para os splits.
        self.raw_dir = settings.source_images_dir

    # --------------------------------------------------------------- estrutura
    def ensure_structure(self) -> None:
        """Cria a árvore train/val/test de images e labels, se faltar.

        Idempotente: chamar várias vezes não faz mal. Quase todo método
        público chama antes de mexer em arquivo, para não depender de o
        operador ter rodado o setup.
        """
        for split in SPLITS:
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def classes(self) -> list[str]:
        """Classes declaradas no YAML (fonte da verdade).

        O YAML manda e não o contrário: `names` define o índice de classe que
        o modelo aprende, então mudar a ordem aqui quebraria todos os .txt já
        anotados.
        """
        yaml_path = settings.base_dir / "training" / "configs" / "chairs.yaml"
        if yaml_path.is_file():
            try:
                import yaml

                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                names = data.get("names")
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
                if isinstance(names, dict):
                    # Formato "0: chair\n1: pile": as chaves são texto, então
                    # o sort numérico evita 10 antes de 2.
                    return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
            except Exception as exc:
                # YAML ilegível não é motivo para o dataset sumir da tela:
                # loga em debug e cai no default.
                log.debug("Não foi possível ler names do YAML: %s", exc)
        return list(DEFAULT_CLASSES)

    # ------------------------------------------------------------------ stats
    def stats(self) -> DatasetStats:
        """Varre as três pastas e monta o resumo + os avisos da página."""
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
                    # Nome pelo índice do YAML. Índice fora do range vira o
                    # próprio número como string: melhor mostrar "7" do que
                    # crashar a página por um .txt com classe inexistente.
                    name = st.classes[cls] if 0 <= cls < len(st.classes) else str(cls)
                    st.per_class[name] = st.per_class.get(name, 0) + 1
            st.empty_label_files += empty
            per_split["annotated"] = labeled
            st.per_split[split] = per_split
            st.total_images += len(imgs)
            st.annotated += labeled
            st.total_boxes += per_split["boxes"]
        # pending = imagens sem .txt. As de .txt vazio já estão "respondidas".
        st.pending = max(0, st.total_images - st.annotated)

        # Avisos: os três primeiros são BLOQUEIOS (não treina assim), o último
        # é só informativo. A distinção fica implícita na ordem e no texto.
        if st.total_images and st.annotated == 0:
            st.warnings.append("Nenhuma imagem anotada: rode o treinamento apenas após anotar.")
        if st.annotated and st.annotated < 20:
            # 20 é um piso de ordem de grandeza: abaixo disso o YOLO aprende
            # a cena, não o objeto, e a validação não diz nada.
            st.warnings.append(
                f"Poucas imagens anotadas ({st.annotated}). YOLO precisa de dezenas para comecar."
            )
        if "val" in st.per_split and st.per_split["val"]["images"] == 0:
            # Sem validação, o "melhor modelo" é o último modelo: o early
            # stopping não tem em que se basear.
            st.warnings.append("Conjunto de validação vazio: métricas não serão confiáveis.")
        if st.raw_images:
            st.warnings.append(
                f"{st.raw_images} imagem(ns) em training/datasets/raw aguardando anotação."
            )
        return st

    def image_status(self, limit: int = 200) -> dict[str, Any]:
        """Lista para a página /annotation: imagem + se está anotada.

        ``limit`` corta só a lista enviada, não o ``total``: assim a tela sabe
        que há mais 800 imagens para anotar mesmo recebendo só 200.
        """
        self.ensure_structure()
        items: list[dict[str, Any]] = []
        pending = 0
        for split in SPLITS:
            for img in images_in(self.root / "images" / split):
                lab = label_for(img)
                parsed = read_label(lab)
                # "anotada" = o arquivo existe, mesmo vazio. É a pergunta que
                # a fila de anotação faz: "falta arquivo ou falta caixa?".
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
                            # Caminho RELATIVO ao root, em POSIX: a API
                            # revalida antes de abrir o arquivo, então não é
                            # um caminho confiável, é só um identificador.
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

        DESTRUTIVO por natureza: apaga as divisões anteriores. Rodar de novo com
        a mesma seed devolve exatamente o mesmo resultado; rodar com seed
        diferente reembaralha as imagens e mantém as anotações junto.
        """
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            # Tolerância de 1e-6 por causa de float: 0.7+0.2+0.1 não dá
            # exatamente 1.0 em ponto flutuante.
            return {"ok": False, "message": "As proporções devem somar 1.0 (ex.: 0.70/0.20/0.10)"}
        self.ensure_structure()

        src = Path(source) if source else self.raw_dir
        if not src.is_dir() or not images_in(src):
            return {
                "ok": False,
                "message": f"Nenhuma imagem encontrada em {src}. Envie imagens primeiro.",
            }

        files = images_in(src)
        # Instância local de Random, NÃO random.shuffle: mexer no global
        # embaralharia outras partes do sistema (nomes de arquivo temporário,
        # amostragem em testes) de forma imprevisível.
        rng = random.Random(seed)
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # test é o resto: é o único que absorve o erro de arredondamento, e
        # por isso nunca fica negativo.
        n_test = max(0, n - n_train - n_val)
        # Garante ao menos 1 imagem em treino e validação quando possível.
        # Um val vazio faz o early stopping não ter com o que comparar.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(0, n - n_train - n_val)
        buckets = {"train": files[:n_train], "val": files[n_train : n_train + n_val], "test": files[n_train + n_val : n_train + n_val + n_test]}

        # Limpa as divisões anteriores para não sobrar imagem duplicada.
        # Sem isso, um segundo split com seed diferente deixaria a mesma foto
        # em dois splits: vazamento entre treino e validação, e métricas
        # infladas sem nenhum sintoma visível.
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
                # copy2 e não move: o raw/ é o inbox do operador e precisa
                # continuar intacto para ele reprocessar/redividir.
                # copy2 preserva mtime, útil para depurar a ordem de captura.
                shutil.copy2(src_file, dst)
                moved += 1
                # Preserva a anotação se ela existir ao lado da imagem original.
                for cand in (
                    # Layout padrão: labels/ do lado de images/ no raw.
                    src_file.with_suffix(".txt"),
                    # Fallback: subpasta labels/ com o nome spaces->_ (o que
                    # o exportador antigo do projeto gerava).
                    src / "labels" / src_file.name.replace(" ", "_") + ".txt",
                ):
                    if cand.is_file():
                        # Reescreve em vez de copiar byte a byte: normaliza as
                        # coordenadas e descarta linhas malformadas.
                        write_label(label_for(dst), read_label(cand))
                        with_labels += 1
                        # break: o primeiro .txt achado ganha, não os dois.
                        break

        # O YAML é regerado no fim, para que `path` aponte para as pastas que
        # acabaram de ser populadas.
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
        """Escreve o ``chairs.yaml`` apontando para os diretórios absolutos.

        `path` absoluto (e não relativo) porque o Ultralytics roda com o cwd
        dele, não com o nosso - um caminho relativo seria resolvido no lugar
        errado.
        """
        yaml_path = Path(path) if path else settings.base_dir / "training" / "configs" / "chairs.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        names = self.classes()
        # "0: chair" é YAML de mapa com chave numérica, formato aceito pelo
        # Ultralytics. A ordem de `names` DEFINE o índice de classe, então
        # mudar esta lista invalida todos os .txt já anotados.
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
            # Rejeitar por extensão, e não tentar abrir: o YOLO não aceitaria
            # mesmo, e o erro chegaria mais tarde e mais confuso.
            return {"ok": False, "message": f"Formato não suportado: {source.suffix}"}
        self.ensure_structure()
        target = self.raw_dir / source.name
        # copy=False serve para quando o chamador JÁ movveu o arquivo para lá;
        # o resolve() evita o shutil de um arquivo sobre ele mesmo.
        if copy and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return {"ok": True, "filename": target.name, "path": str(target)}

    def import_dir(self, source: Path, limit: int = 5000) -> dict[str, Any]:
        """Importa todas as imagens de uma pasta (ex.: dump da câmera).

        O teto de 5000 evita travar o request HTTP por horas quando o
        operador aponta para um diretório com dezenas de milhares de frames.
        """
        source = Path(source)
        if not source.is_dir():
            return {"ok": False, "message": f"Pasta não encontrada: {source}"}
        added = 0
        # images_in() já vem ordenada e filtrada por extensão, então o [:limit]
        # é determinístico entre execuções.
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

        Cuidado conhecido: usar um modelo ruim aqui amplifica o erro, porque
        o operador tendeu a aceitar as caixas. O `conf` default é 0.35
        (abaixo do 0.50 de produção) para gerar mais candidatos e dar escolha
        ao revisor.
        """
        try:
            # Import tardio: o DatasetService é usado mesmo sem a
            # Ultralytics instalada (só para listar e dividir imagens).
            from app.ai.yolo_detector import ModelNotAvailable, YoloDetector
        except Exception as exc:
            return {"ok": False, "message": f"Ultralytics indisponível: {exc}"}

        detector = YoloDetector()
        if not detector.load():
            # Sem modelo carregado, nem o genérico: não há o que pré-rotular.
            return {
                "ok": False,
                "message": detector.load_error or "Nenhum modelo carregado para pré-rotular.",
            }
        if not detector.trained_model:
            # Trava deliberada. Com o modelo genérico pré-treinado, o detector
            # enxerga a PILHA INTEIRA como uma cadeira, e a pré-rotulagem
            # escreveria "1 caixa = 1 cadeira" para uma pilha de 8. Isso não é
            # uma anotação ruim: é uma anotação que contradiz a verdade, e o
            # modelo treinado nela aprende a contar pilhas como cadeiras.
            # Como o revisor confia no que a IA sugeriu, o erro passa.
            #
            # A pré-rotulagem só faz sentido com um modelo já treinado NAS
            # SUAS imagens. Até lá, o caminho é anotar à mão.
            return {
                "ok": False,
                "message": (
                    "Pré-rotulagem indisponível: o modelo ativo é genérico "
                    "pré-treinado, que vê a pilha inteira como uma cadeira. "
                    "Use-o só depois de treinar o seu primeiro modelo — treine "
                    "primeiro, volte aqui para só revisar as caixas."
                ),
            }
        try:
            chairs, piles = detector.class_indices()
        except Exception:
            # classes() justo: seguimos com todas as classes do modelo.
            chairs, piles = set(), set()
        classes = chairs | piles or None

        labeled = 0
        skipped = 0
        reviewed = 0
        # Só o split train. Anotar val/test com o próprio modelo vaza o
        # modelo para a validação, e as métricas deixam de significar nada.
        for img in images_in(self.root / "images" / "train"):
            lab = label_for(img)
            if lab.is_file():
                # NÃO sobrescreve anotação existente (humana ou não).
                continue
            try:
                import cv2

                frame = cv2.imread(str(img), cv2.IMREAD_COLOR)
                if frame is None:
                    # Arquivo corrompido ou não é imagem de verdade.
                    skipped += 1
                    continue
                dets, _ = detector.predict(frame, conf=conf, classes=classes)
                h, w = frame.shape[:2]
                boxes = []
                for d in dets:
                    # Conversão de canto (x1,y1,x2,y2) para centro+tamanho
                    # normalizado do YOLO, tudo dividido pela imagem.
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
                # Uma imagem problemática não cancela as outras 500.
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
        """Exporta as caixas de uma imagem em JSON (formato padrão de ferramentas).

        É o inverso de save_labels: o YOLO guarda normalizado, o frontend de
        anotação trabalha em pixels. A conversão acontece aqui, num lugar só.
        """
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
            # w e h vão junto: o frontend precisa das dimensões reais para
            # posicionar as caixas na tela.
            "width": w,
            "height": h,
            "annotations": [
                {
                    "class_id": c,
                    "label": names[c] if 0 <= c < len(names) else str(c),
                    # Centro -> canto: (cx - w/2) * largura_da_imagem.
                    "x": (cx - bw / 2) * w,
                    "y": (cy - bh / 2) * h,
                    "width": bw * w,
                    "height": bh * h,
                }
                for c, cx, cy, bw, bh in boxes
            ],
        }

    def save_labels(self, image_name: str, split: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Grava as caixas vindas do frontend (pixels) e converte para YOLO.

        Todo clamp aqui (max/min em 0..1) existe porque a caixa arrastada no
        navegador pode sair da imagem. Um valor normalizado > 1 faria o
        Ultralytics recusar o dataset inteiro com um erro pouco claro.
        """
        img_path = self.root / "images" / split / image_name
        if not img_path.is_file():
            return {"ok": False, "message": f"Imagem não encontrada: {image_name}"}
        w, h = read_image_size(img_path)
        if not w or not h:
            # Sem dimensões não dá para normalizar: melhor recusar do que
            # gravar um .txt com coordenadas inválidas.
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
                # Anotação malformada: pula só ela, salvando o resto.
                continue
            # Canto+tamanho (pixels) -> centro+tamanho normalizado, com clamp.
            cx = max(0.0, min(1.0, (x + bw / 2) / w))
            cy = max(0.0, min(1.0, (y + bh / 2) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            boxes.append((cls, cx, cy, nw, nh))
        write_label(label_for(img_path), boxes)
        return {"ok": True, "boxes": len(boxes), "message": f"{len(boxes)} caixa(s) salva(s)."}


def read_image_size(path: Path) -> tuple[int, int]:
    """Dimensões ``(largura, altura)`` sem carregar a imagem inteira.

    Try OpenCV primeiro (rápido e já é dependência do projeto) e Pillow como
    reserva. Devolve (0, 0) em vez de levantar: quem chama decide o que fazer
    com a falha.
    """
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            # shape é (altura, largura, canais): a ordem é invertida em
            # relação ao que a função promete.
            return img.shape[1], img.shape[0]
    except Exception:
        pass
    try:
        from PIL import Image

        # im.size já é (largura, altura), sem inversão.
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


__all__ = ["DatasetService", "DatasetStats", "read_label", "write_label", "label_for", "images_in", "SPLITS"]
