"""Coleta de dataset com controle de variedade — o anti-"43 fotos iguais".

    python -m tools.collect_dataset --status
    python -m tools.collect_dataset --scene "pilha 5 - esquerda" --seconds 20
    python -m tools.collect_dataset --from-dir fotos/ --scene "minhas fotos"

Por que esta ferramenta existe
==============================

``extract_frames`` extrai frames, e faz isso bem. O problema é o que acontece
depois: com a câmera parada, 43 frames da mesma cena saem quase idênticos
(diferença média de 2,7 em 255 - ruído de sensor) e o operador fica com a
impressão de que tem dado de treino. Não tem: são 1 exemplo, 43 vezes.

Para treinar um detector é preciso **variedade de cena**, não volume. Um
frame de cada arranjo diferente vale mais que trinta frames do mesmo arranjo,
porque é a mudança de arranjo que ensina "onde está a cadeira" em vez de
"como é esta parede".

O que ela faz
=============

1. Cada captura recebe um **nome de cena** (``--scene``) e é registrada num
   manifesto (``training/datasets/raw/_scenes.json``).
2. Cada cena guarda uma **assinatura visual** (32x32 em tons de cinza).
   Quando você captura uma cena nova, a ferramenta compara a assinatura com
   as existentes e **avisa se ela é parecida demais** - quase sempre o
   sintoma de "esqueci de rearrumar as cadeiras".
3. ``--status`` mostra o progresso em direção a uma meta legível: quantas
   cenas distintas existem, quantas imagens, e o que ainda falta.

Medida de "cena diferente"
=========================

Distância = diferença média absoluta entre duas assinaturas 32x32 (0..255).
Os limiares vieram de medição real neste projeto, não de chute:

    mesma cena, câmera parada ................ 0,85 a 1,27
    mesma cena, outra hora (exposição) ........ 4,26
    cena diferente (cadeiras rearrumadas) ..... 22,62 a 23,92

O limiar padrão é **8**: ficam ~3x acima do pior caso de "mesma cena" e
~2,7x abaixo do melhor caso de "cena diferente". Um número no meio do
buraco entre os dois grupos, que é onde um limiar deve ficar.

Meta
====

    15 a 30 cenas distintas, 100 a 300 imagens no total

Abaixo de 15 o YOLO costuma não convergir. Acima de 30 o ganho é pequeno e o
custo de anotar cresce. O número de imagens importa menos que o de cenas.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402

RAW = ROOT / "training" / "datasets" / "raw"
MANIFEST = RAW / "_scenes.json"

# Distância acima da qual duas imagens são de cenas diferentes.
# Ver a tabela no docstring do módulo.
SAME_SCENE_MAX = 8.0

# Tamanho da assinatura. 32x32 em cinza: pequeno o bastante para o manifesto
# ficar leve, grande o bastante para distinguir rearranjo de variação de luz.
SIG_SIZE = 32

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Metas.-as duas importam: número de Cenas é o que faz o modelo generalizar,
# número de imagens é o que dá volumetria para não decorar.
TARGET_SCENES_MIN = 15
TARGET_SCENES_GOOD = 30
TARGET_IMAGES = (100, 300)


# ------------------------------------------------------------------ assinatura
def signature(path: Path) -> np.ndarray:
    """Reduz a imagem a 32x32 em tons de cinza (a 'impressão digital' da cena).

    Deliberadamente burra: sem rede neural, sem dependência extra, e com
    resultado estável entre execuções. O objetivo é responder "isto é a mesma
    cena?", não "estas imagens são bonitas".
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"não consegui ler a imagem: {path}")
    small = cv2.resize(img, (SIG_SIZE, SIG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância entre duas assinaturas (0 = idênticas, ~100 = muito diferente)."""
    return float(np.mean(np.abs(a - b)))


def encode_sig(sig: np.ndarray) -> str:
    """Serializa a assinatura para o manifesto (base64 de uint8)."""
    return base64.b64encode(np.clip(sig, 0, 255).astype(np.uint8).tobytes()).decode()


def decode_sig(text: str) -> np.ndarray:
    raw = base64.b64decode(text.encode())
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(SIG_SIZE, SIG_SIZE)


# ------------------------------------------------------------------- manifesto
def load_manifest() -> dict:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Manifesto corrompido não pode derrubar a coleta: as imagens
            # estão no disco e valem mais que o índice. Recomeça vazio.
            return {"scenes": []}
    return {"scenes": []}


def save_manifest(data: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def images_on_disk() -> list[Path]:
    if not RAW.is_dir():
        return []
    return sorted(
        p for p in RAW.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# --------------------------------------------------------------------- coleta
def run_extract(seconds: float, fps: float, prefix: str, min_diff: float, rtsp_env: str) -> None:
    """Chama o extract_frames reaproveitando toda a lógica já existente.

    Delegar em vez de reimplementar a captura (RTSP, recorte de moldura do
    player, ROI, dedup) evita dois lugares com a mesma regra. --min-diff 0
    porque quem decide o que é duplicado aqui é a comparação de CENA, feita
    depois: guardar os frames e descartar o conjunto inteiro é mais barato do
    que recapturar a câmera.
    """
    cmd = [
        sys.executable, "-m", "tools.extract_frames",
        "--rtsp-env", rtsp_env,
        "--seconds", str(seconds),
        "--fps", str(fps),
        "--min-diff", str(min_diff),
        "--prefix", prefix,
    ]
    print(f"capturando {seconds:.0f}s a {fps} fps da câmera (variável {rtsp_env})...")
    subprocess.run(cmd, cwd=ROOT, check=False)


def import_dir(src: Path) -> list[Path]:
    """Copia imagens de outro lugar para o dataset."""
    if not src.is_dir():
        raise SystemExit(f"pasta não encontrada: {src}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"nenhuma imagem em {src}")
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    copied = []
    for i, f in enumerate(files):
        dest = RAW / f"import_{stamp}_{i:03d}{f.suffix.lower()}"
        dest.write_bytes(f.read_bytes())
        copied.append(dest)
    print(f"{len(copied)} imagem(ns) importada(s) de {src}")
    return copied


# -------------------------------------------------------------------- relatório
def analyse(manifest: dict) -> dict:
    """Junta manifesto e disco num retrato usável do dataset."""
    scenes = manifest.get("scenes", [])
    n_images = len(images_on_disk())

    # Imagens que entraram no disco sem passar pelo manifesto (upload pelo
    # dashboard, cópia manual). Elas contam para o total, mas não viraram
    # cena nomeada - e é por isso que a lista de cenas pode ser menor que o
    # número de imagens sem que nada esteja errado.
    accounted = sum(s.get("count", 0) for s in scenes)
    orphan = max(0, n_images - accounted)

    return {
        "scenes": scenes,
        "n_images": n_images,
        "n_scenes": len(scenes),
        "orphan": orphan,
    }


def check_new_scene(sig: np.ndarray, manifest: dict, exclude: str | None) -> str | None:
    """Devolve o nome da cena mais parecida, se estiver perto demais.

    É o alerta que salva tempo: "esqueci de rearrumar" e "a câmera não moveu"
    produzem o mesmo sintoma (imagens novas idênticas às antigas), e sem
    esse aviso a pessoa só descobre depois de horas de anotação inútil.
    """
    best_name, best_d = None, float("inf")
    for scene in manifest.get("scenes", []):
        if scene.get("name") == exclude:
            continue
        d = distance(sig, decode_sig(scene["signature"]))
        if d < best_d:
            best_name, best_d = scene.get("name", "?"), d
    if best_name is not None and best_d < SAME_SCENE_MAX:
        return f"{best_name} (distância {best_d:.1f})"
    return None


def bar(value: float, total: float, width: int = 24) -> str:
    filled = int(round(width * min(1.0, value / max(1e-9, total))))
    return "█" * filled + "░" * (width - filled)


def report(manifest: dict) -> int:
    info = analyse(manifest)
    scenes = info["scenes"]
    n_img, n_scen = info["n_images"], info["n_scenes"]

    print()
    print("=" * 66)
    print("  DATASET — o que você já tem")
    print("=" * 66)
    print(f"  imagens no disco ....... {n_img}")
    print(f"  cenas nomeadas ......... {n_scen}")
    if info["orphan"]:
        print(f"  sem cena associada ..... {info['orphan']}  "
              f"(entrou por upload/cópia — rode --status após nomeá-las)")
    print()

    if scenes:
        print("  Cenas registradas:")
        for s in scenes:
            print(f"    {s.get('count', 0):>3} img  {s.get('name', '?')}"
                  f"   ({s.get('at', '')[:19]})")
        print()

    # --- progresso ---
    print("  Progresso:")
    good = TARGET_SCENES_GOOD
    print(f"    cenas distintas  {bar(n_scen, good)}  {n_scen}/{good}"
          f"   (mínimo viável: {TARGET_SCENES_MIN})")
    lo, hi = TARGET_IMAGES
    print(f"    imagens          {bar(n_img, hi)}  {n_img}/{hi}"
          f"   (mínimo: {lo})")
    print()

    # --- diagnóstico e o que fazer ---
    if n_scen == 0:
        print("  ⚠ Nenhuma cena nomeada ainda.")
        print("    Rode com --scene para registrar cada arranjo:")
        print('      python -m tools.collect_dataset --scene "pilha 3 - centro" --seconds 20')
    elif n_scen < TARGET_SCENES_MIN:
        faltam = TARGET_SCENES_MIN - n_scen
        print(f"  ⚠ Faltam ~{faltam} cenas distintas. Reposicionar as cadeiras e")
        print("    capturar de novo. Contagem de imagens não substitui isso.")
    elif n_img < lo:
        faltam = lo - n_img
        print(f"  ⚗ Cenas suficientes, mas faltam ~{faltam} imagens para volumetria.")
        print("    Pode capturar mais frames por cena (a câmera parada serve).")
    else:
        print("  ✓ Dataset com aparência saudável. Próximo passo: anotar e treinar.")
    print()

    print("  O que mais faz o modelo generalizar (checklist):")
    for item, dica in [
        ("variedade de quantidade", "1, 2, 3, 5, 8, 12, 20 cadeiras — o que mais importa"),
        ("variedade de posição", "pilha à esquerda, ao centro e à direita"),
        ("cena vazia", "uma captura sem nenhuma cadeira: ensina que o vazio é vazio"),
        ("iluminação", "dia e noite, ou com/sem luz acesa"),
        ("oclusão", "com e sem pessoa passeando na frente"),
        ("ângulo", "câmera um pouco mais alta e um pouco mais baixa"),
    ]:
        print(f"    - {item:22s} {dica}")
    print()
    print("=" * 66)
    return 0


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.collect_dataset",
        description="Coleta dataset de treinamento controlando a variedade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Meta\n====\n")[0],
    )
    p.add_argument("--scene", help="Nome do arranjo capturado (ex.: 'pilha 5 - esquerda')")
    p.add_argument("--seconds", type=float, default=20.0, help="Duração da captura (padrão 20s)")
    p.add_argument("--fps", type=float, default=1.0, help="Frames por segundo (padrão 1)")
    p.add_argument("--min-diff", type=float, default=0.0,
                   help="Filtro de duplicata do extrator (padrão 0: a cena decide)")
    p.add_argument("--from-dir", help="Importa imagens de uma pasta em vez de capturar")
    p.add_argument("--rtsp-env", default="CAMERA_RTSP_URL",
                   help="Variável de ambiente com a URL RTSP (padrão CAMERA_RTSP_URL)")
    p.add_argument("--status", action="store_true", help="Só mostra o relatório e sai")
    p.add_argument("--allow-similar", action="store_true",
                   help="Aceita uma cena parecida com outra já existente")
    args = p.parse_args(argv)

    manifest = load_manifest()

    if args.status or not (args.scene or args.from_dir):
        return report(manifest)

    if not args.scene:
        p.error("--scene é obrigatório ao capturar ou importar (dá nome ao arranjo)")

    # --- captura ou importação ---
    if args.from_dir:
        new_files = import_dir(Path(args.from_dir))
    else:
        before = set(images_on_disk())
        # Prefixo COM horário, e não um nome fixo: o extract_frames numera do
        # zero, então um prefixo repetido faria a cena nova sobrescrever os
        # arquivos da cena anterior no mesmo caminho. Cada captura precisa
        # ocupar o namespace dela.
        prefix = f"col_{time.strftime('%Y%m%d_%H%M%S')}_"
        run_extract(args.seconds, args.fps, prefix, args.min_diff, args.rtsp_env)
        new_files = [f for f in images_on_disk() if f not in before]

    if not new_files:
        print("\nNenhuma imagem nova foi salva. A câmera entregou a mesma cena?")
        return 1

    # --- a cena é nova mesmo? ---
    sig = signature(new_files[0])
    similar = check_new_scene(sig, manifest, exclude=None)
    if similar and not args.allow_similar:
        # As imagens JÁ estão no disco (o extrator salva antes de decidirmos).
        # Deixá-las seria pior que inútil: elas inflam a contagem e fariam o
        # relatório dizer que há mais dado do que há. Um aviso que deixa lixo
        # atrás ensina a ignorar o aviso.
        for f in new_files:
            f.unlink(missing_ok=True)
        print()
        print(f"⚠ AVISO: esta cena é praticamente igual a '{similar}'.")
        print(f"  As {len(new_files)} imagens foram descartadas.")
        print("  Reposicione as cadeiras antes de capturar de novo.")
        print("  Se for intencional (mesmo arranjo, outro momento), use --allow-similar.")
        return 2

    # --- registra ---
    manifest["scenes"].append({
        "name": args.scene,
        "count": len(new_files),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "signature": encode_sig(sig),
    })
    save_manifest(manifest)
    print(f"\ncena registrada: '{args.scene}' com {len(new_files)} imagem(ns)")
    return report(manifest)


if __name__ == "__main__":
    sys.exit(main())
