"""Treinamento de um modelo YOLO com as imagens das cadeiras.

Executado pelo serviço de treinamento do dashboard
(``POST /api/training/start``) ou direto no terminal::

    python -m training.scripts.train --epochs 100 --batch 16

O treinamento roda em processo separado, então o dashboard continua
respondendo durante todo o tempo.

Antes de treinar, confira:

1. ``training/datasets/images/train`` tem imagens;
2. cada imagem tem um ``.txt`` em ``training/datasets/labels/train``;
3. o ``.txt`` tem as classes certainas (ver ``training/configs/chairs.yaml``).
"""

# =============================================================================
# ARQUIVO / MAPA  -  training/scripts/train.py
#
# O que faz: treina o modelo YOLO PRÓPRIO. O modelo genérico COCO (yolo11n.pt)
#   reconhece "cadeira" solta, mas não a PILHA de cadeiras empilhadas, que é
#   justamente o objeto que o sistema precisa contar.
#   Roda em processo separado (o dashboard continua respondendo).
#
# Ordem de leitura:
#   1. ROOT / sys.path ......... roda com `python -m` de dentro de qualquer pasta
#   2. report_environment ..... GPU/VRAM reais, mostradas ANTES de treinar
#   3. preflight .............. confere dataset, .txt, classes e nº de caixas
#   4. main() .................. CLI (padrões vêm do .env) -> device -> treino
#   5. rodapé ................. onde estão best.pt, métricas e próximos passos
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path (execução via -m)
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def report_environment() -> dict[str, str]:
    """Mostra GPU/VRAM reais antes de começar (requisito do projeto)."""
    info: dict[str, str] = {}
    print("=" * 64)
    print("AMBIENTE DE TREINAMENTO")
    print("=" * 64)
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            print("GPU detectada:")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                # mem_get_info devolve bytes (livre, total). Serve para prever
                # se o batch cabe na VRAM antes de o treino estourar memória.
                free, total = torch.cuda.mem_get_info(i)
                print(
                    f"  {p.name}\n"
                    f"  VRAM total: {total / 1024 ** 3:.1f} GB\n"
                    f"  VRAM livre:  {free / 1024 ** 3:.1f} GB\n"
                    f"  Compute capability: {p.major}.{p.minor}"
                )
            info["device"] = "cuda:0"
        else:
            print("GPU: nenhuma (CUDA indisponível)")
            print("O treino vai rodar na CPU. Funciona, mas bem mais devagar.")
            print("Para usar GPU NVIDIA:")
            print("  pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu124")
            info["device"] = "cpu"
    except Exception as exc:
        # Sem torch (ou driver quebrado) o treino não roda; degradamos para CPU
        # só para a mensagem de erro do Ultralytics ser mais clara.
        print(f"Não foi possível ler a GPU: {exc}")
        info["device"] = "cpu"
    print("=" * 64)
    return info


def preflight(data_yaml: Path, epochs: int) -> list[str]:
    """Valida o dataset e devolve uma lista de problemas encontrados.

    Rodar isso ANTES do treino economiza horas: um dataset sem rótulo ou com
    classe errada só dá erro lá no fim, depois de horas de GPU.
    """
    problems: list[str] = []
    print(f"\nDataset: {data_yaml}")
    if not data_yaml.is_file():
        return [f"Arquivo de dataset não encontrado: {data_yaml}"]

    import yaml

    # O YAML é escrito pelo DatasetService com `path:` ABSOLUTO, senão o
    # Ultralytics tentaria resolver relativo ao cwd e acharia nada.
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = Path(cfg.get("path", data_yaml.parent.parent))
    names = cfg.get("names", ["chair"])
    if isinstance(names, dict):
        n_classes = len(names)
    else:
        n_classes = len(names)
    print(f"Classes ({n_classes}): {names}")

    counts: dict[str, int] = {}
    boxes_total = 0
    for split in ("train", "val", "test"):
        # ATENÇÃO à assimetria: as IMAGENS vêm do YAML (chave `train`/`val`/
        # `test`), mas os RÓTULOS são sempre em `labels/<split>`, pelo nome da
        # imagem (mesmo stem, extensão .txt). É o padrão do YOLO.
        img_dir = base / cfg.get("train" if split == "train" else split, f"images/{split}")
        lab_dir = base / "labels" / split
        if not img_dir.is_dir():
            # test é opcional; train e val são obrigatórios.
            if split in ("train", "val"):
                problems.append(f"Pasta de imagens ausente: {img_dir}")
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        labeled = 0
        for img in imgs:
            lab = lab_dir / (img.stem + ".txt")
            if lab.is_file():
                labeled += 1
                try:
                    # Formato YOLO de detecção: uma linha por caixa,
                    # "id_classe cx cy w h", com cx/cy/w/h NORMALIZADOS 0..1
                    # (fração da largura/altura da imagem). Linha vazia é ignorada.
                    lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
                except OSError:
                    lines = []
                boxes_total += len(lines)
                for line in lines:
                    try:
                        # int(float(...)) tolera "0.0" vindo de alguma ferramenta.
                        cid = int(float(line.split()[0]))
                    except (ValueError, IndexError):
                        problems.append(f"Rótulo malformado em {lab.name}: {line!r}")
                        continue
                    # Índice de classe começa em 0: um id igual a n_classes é erro.
                    if cid >= n_classes:
                        problems.append(
                            f"{lab.name}: classe {cid} não existe no YAML (0..{n_classes - 1})"
                        )
        counts[split] = len(imgs)
        print(f"  {split:<5} imagens={len(imgs):<5} anotadas={labeled:<5}")
        if split == "train" and labeled == 0:
            problems.append("Nenhuma imagem de treino anotada! Treinar não vai funcionar.")
        # ~30 é o piso prático: abaixo disso o YOLO praticamente não generaliza
        # e o resultado é um modelo que só acerta as fotos que viu.
        if split == "train" and labeled < 30:
            problems.append(
                f"Só {labeled} imagens de treino anotadas. Com menos de ~30 o YOLO "
                f"vai ter dificuldade grave. Ver README (quantas imagens usar)."
            )
        if split == "val" and len(imgs) == 0:
            problems.append("Sem imagens de validação: as métricas não serão confiáveis.")

    print(f"Total de caixas anotadas: {boxes_total}")
    # Contar o objeto exige ver o objeto várias vezes. Poucas caixas = pouco
    # sinal, mesmo que muitas imagens existam (imagens vazias não ensinam nada).
    if boxes_total < 100:
        problems.append(f"Apenas {boxes_total} caixas no dataset. Treine com mais imagens.")
    if epochs < 10:
        problems.append(f"Apenas {epochs} épocas: o modelo provavelmente não vai convergir.")
    return problems


def main(argv: list[str] | None = None) -> int:
    # Os defaults vêm do .env (seção TRAINING_*), não de números soltos aqui:
    # assim o mesmo script serve ao terminal e à aba Treinamento do dashboard.
    p = argparse.ArgumentParser(description="Treina um modelo YOLO para contar cadeiras")
    p.add_argument("--data", default=str(ROOT / "training" / "configs" / "chairs.yaml"))
    p.add_argument("--model", default=settings.training.model, help="Modelo base (ex.: yolo11n.pt)")
    p.add_argument("--epochs", type=int, default=settings.training.epochs)
    p.add_argument("--imgsz", type=int, default=settings.training.image_size)
    p.add_argument("--batch", type=int, default=settings.training.batch, help="-1 = automático")
    p.add_argument("--device", default="", help="auto | cpu | cuda | 0")
    p.add_argument("--workers", type=int, default=settings.training.workers)
    p.add_argument("--patience", type=int, default=settings.training.patience)
    p.add_argument("--seed", type=int, default=settings.training.seed)
    p.add_argument("--project", default=settings.training.project)
    p.add_argument("--name", default="", help="Nome do run (padrão: chair_counter)")
    p.add_argument("--resume", action="store_true", help="Retoma o último treino interrompido")
    p.add_argument("--dry-run", action="store_true", help="Só valida o dataset e sai")
    args = p.parse_args(argv)

    setup = report_environment()
    # Precedência do device: --device (linha de comando) > TRAINING_DEVICE do
    # .env > detecção real da máquina. "auto" também é aceito e cai no
    # resolve_device(), que pergunta ao torch.
    device = args.device or settings.training.device or setup.get("device", "cpu")
    if device.lower() == "auto":
        device = settings.resolve_device()

    data_yaml = Path(args.data)
    problems = preflight(data_yaml, args.epochs)
    if problems:
        print("\nPROBLEMAS ENCONTRADOS:")
        for prob in problems:
            print(f"  - {prob}")
        if args.dry_run:
            return 1
        print("\nDeseja continuar mesmo assim? Use --dry-run para apenas validar.")
        # Guarda de segurança para quando a chamada vem do dashboard (sem
        # terminal): `input()` levantaria erro e derrubaria o serviço.
        if not problems or all("aqui está" in x for x in problems):
            return 1
        answer = input("Continuar mesmo assim? [s/N] ").strip().lower()
        if answer not in ("s", "sim", "y", "yes"):
            return 1

    if args.dry_run:
        print("\nDataset OK para treino (--dry-run: nada foi treinado).")
        return 0

    # Import tarde: `ultralytics` é pesado e só é necessário no treino de fato.
    from ultralytics import YOLO

    name = args.name or settings.training.name
    project = ROOT / args.project
    print(f"\nIniciando treino: {args.model} | {args.epochs} épocas | device={device}")
    print(f"Saída em: {project / name}")
    print("-" * 64)

    model = YOLO(args.model)
    try:
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            # batch -1 = o Ultralytics mede a VRAM livre e escolhe o maior batch
            # que couber. Em CPU, um batch grande estouraria a RAM.
            batch=args.batch,
            device=device,
            # workers = processos que leem e pré-processam as imagens. Ajuda até
            # saturar o disco; acima disso só gasta RAM. 0 desliga o paralelismo.
            workers=args.workers,
            project=str(project),
            name=name,
            # Sobrescreve um run anterior com o MESMO nome. Cada modelo novo
            # precisa de um --name diferente para não perder o anterior.
            exist_ok=True,
            # Mesma seed + mesmo dataset = mesmo resultado. Sem isso, comparar
            # dois ajustes de hiperparâmetro não significa nada.
            seed=args.seed,
            # Early stopping: para o treino se a métrica não melhorar há N épocas.
            patience=args.patience,
            # Aumento de dados: mudos para imagens de depósito.
            # Iluminação varia muito ao longo do dia e as pilhas são vistas de
            # ângulos levemente diferentes, mas a geometria da cadeira é
            # rígida - por isso os valores são moderados.
            hsv_h=0.015,        # matiz
            hsv_s=0.5,          # saturação
            hsv_v=0.4,          # brilho/exposição
            degrees=3.0,        # rotação (pouco: a câmera é fixa)
            translate=0.08,     # deslocamento
            scale=0.25,         # escala
            shear=1.0,          # leve cisalhamento
            perspective=0.0005,
            flipud=0.0,         # NÃO vira vertical (cadeira para cima é outra coisa)
            fliplr=0.5,         # espelhar horizontal: pilhas são simétricas
            mosaic=0.5,         # mosaico: ajuda com variação de contexto
            mixup=0.0,          # mixup embaralha cenas: não usar em pilhas
            erasing=0.0,
            close_mosaic=10,    # desliga o mosaico nas últimas 10 épocas
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
        return 130  # 130 = código de saída convencional de "interrompido por SIGINT"
    except Exception as exc:
        print(f"\nFalha no treinamento: {exc}")
        return 1

    run_dir = project / name
    print("\n" + "=" * 64)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 64)
    # best.pt = época de melhor fitness (maior mAP) - é o que vai para produção.
    # last.pt = última época; serve para retomar com --resume.
    print(f"best.pt  : {run_dir / 'weights' / 'best.pt'}")
    print(f"last.pt  : {run_dir / 'weights' / 'last.pt'}")
    # results.csv é a fonte VERDADEIRA das métricas (o validate.py só lê ele).
    print(f"métricas : {run_dir / 'results.csv'}")
    print(f"gráficos : {run_dir / 'results.png'}")
    print(f"matriz de confusão: {run_dir / 'confusion_matrix.png'}")
    print()
    print("PRÓXIMOS PASSOS:")
    print(f"  1. Validar:  python -m training.scripts.validate --weights {run_dir / 'weights' / 'best.pt'}")
    print(f"  2. Registrar no banco: POST /api/models/register-run  {{\"run_dir\": \"{run_dir}\"}}")
    print(f"  3. Ativar:  aba Modelo no dashboard → ATIVAR")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
