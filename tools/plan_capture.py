"""Planeja as próximas capturas de dataset a partir do que já foi registrado.

    python -m tools.plan_capture
    python -m tools.plan_capture --json

Lê o mesmo manifesto que o ``collect_dataset`` escreve
(``training/datasets/raw/_scenes.json``) e responde à única pergunta que
importa quando se está montando um dataset: **o que eu capturo agora?**

O foco é a dimensão que mais pesa
=================================

Não é o número de imagens, é o **espalhamento da quantidade de cadeiras**.
Um detector que só viu pilhas de 3 e 4 nunca aprendeu a generalizar: ele
nunca viu o conceito de "mais cadeiras" mudar. Um que viu 1 e 20 aprendeu.

Por isso o planejador mede **spread**, não contagem: duas cenas com 3 e 4
cadeiras têm spread 1,3x — praticamente o mesmo exemplo repetido com nuance.
Com 1 e 20, o spread é 20x, e é aí que o modelo começa a generalizar.

O que este utilitário NÃO faz
==============================

- **Não inferi nada dos pixels.** Seria tentador detectar "cena vazia" por
  textura, e foi testado: não funciona neste ambiente. O piso cerâmico de um
  galpão tem tanta borda quanto uma pilha de cadeiras (29,1 contra 30,6 de
  energia de gradiente), porque ladrilho tem linhas. Um detector assim
  classificaria o chão como cadeira.

- **Não adivinha arranjo pelo nome com segurança.** As dimensões abaixo são
  lidas do que você escreveu em ``--scene``. Nomeie de forma consistente
  ("pilha 8 - esquerda - noite") e o relatório fica preciso. Nomeie tudo
  "foto1", "foto2" e ele só consegue dizer quantas imagens existem.

Ele sugere, você decide. Ele não enxerga o seu galpão.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "training" / "datasets" / "raw"
MANIFEST = RAW / "_scenes.json"

# Quantidades que valem a pena cobrir. Não é sorteio: a escada vai do
# "cabe na mão" ao "corredor inteiro", e o modelo precisa deExamples nos dois
# extremos para saber extrapolar.
QUANTIDADES = [1, 2, 3, 5, 8, 12, 20]

# Palavras que aparecem no nome da cena e a dimensão que cobrem.
PALAVRAS: dict[str, tuple[str, ...]] = {
    "vazio": ("vazio", "vazia", "sem cadeira", "nenhuma", "sem nada", "nada"),
    "esquerda": ("esquerda", "esq", "left"),
    "centro": ("centro", "central", "meio"),
    "direita": ("direita", "dir", "right"),
    "dia": ("dia", "luz", "claro", "manha", "tarde"),
    "noite": ("noite", "escuro", "luz apagada", "sem luz"),
    "pessoa": ("pessoa", "gente", "frente", "oclu", "parado"),
}

# Números escritos por extenso que aparecem em nome de cena.
NUMEROS = {
    "uma": 1, "um": 1, "duas": 2, "dois": 2, "tres": 3, "três": 3,
    "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
    "dez": 10, "doze": 12, "quinze": 15, "vinte": 20, "trinta": 30,
}


def load_scenes() -> list[dict]:
    """Cenas registradas. Arquivos soltos viram uma cena sem nome."""
    scenes: list[dict] = []
    if MANIFEST.is_file():
        try:
            scenes = json.loads(MANIFEST.read_text(encoding="utf-8")).get("scenes", [])
        except json.JSONDecodeError:
            scenes = []
    if not scenes:
        # Sem manifesto: ainda assim dá para dizer o essencial.
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        loose = [p for p in RAW.iterdir() if p.is_file() and p.suffix.lower() in exts] if RAW.is_dir() else []
        if loose:
            scenes = [{"name": "(sem nome)", "count": len(loose), "at": ""}]
    return scenes


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_quantities(names: list[str]) -> list[int]:
    """Puxa a contagem de cadeiras escrita no nome da cena.

    Aceita "pilha 8", "pilha de 12", "8 cadeiras", "20 cadeiras empilhadas".

    O número precisa estar **colado** na palavra (antes ou depois). Isso
    importa: com uma janela ampla, "camera 2 - pilha 5" seria lido como 2 e
    5, porque o "pilha" do fim da frase contaria como contexto do 2. Ignorar
    a vizinhança larga é o que separa "quantidade de cadeiras" de "nome de
    arquivo" ou "data".
    """
    # "pilha 8", "pilha de 8", "8 cadeiras", "8 unidades de cadeira".
    depois = r"(?:pilha|cadeira|unidade|pe[çc]a)s?\s*(?:de\s*)?(\d{1,3})\b"
    antes = r"\b(\d{1,3})\s*(?:unidades\s+de\s+)?(?:cadeiras?|pilhas?|unidades?)\b"
    padroes = (re.compile(depois), re.compile(antes))

    found: set[int] = set()
    for name in names:
        t = norm(name)
        for word, value in NUMEROS.items():
            if re.search(rf"\b{word}\b", t):
                found.add(value)
                break
        for rx in padroes:
            for m in rx.finditer(t):
                value = int(m.group(1))
                if 1 <= value <= 500:
                    found.add(value)
    return sorted(found)


def cover(names: list[str]) -> dict[str, bool]:
    """Dimensão coberta? Lê o nome de todas as cenas."""
    blob = " | ".join(norm(n) for n in names)
    out: dict[str, bool] = {}
    for key, words in PALAVRAS.items():
        out[key] = any(w in blob for w in words)
    return out


def suggest(quantities: list[int], covered: dict[str, bool]) -> list[tuple[int, str, str]]:
    """Sugere as próximas cenas. (prioridade, rótulo, comando pronto)."""
    out: list[tuple[int, str, str]] = []

    # --- 1. quantidade: a dimensão que mais pesa ---
    have = set(quantities)
    missing = [q for q in QUANTIDADES if q not in have]
    if not quantities:
        out.append((0, "comece por uma quantidade bem definida",
                    f'pilha {QUANTIDADES[2]} - centro'))
    for q in missing:
        # Extremos primeiro: eles ensinam extrapolação, que é o difícil.
        prio = 1 if q in (1, 20) else 2
        out.append((prio, f"quantidade {q} cadeiras",
                    f"pilha {q} - centro"))
    if not covered["vazio"]:
        out.append((0, "cena VAZIA, sem nenhuma cadeira",
                    "vazio - sem cadeira"))
    if not covered["esquerda"]:
        out.append((2, "pilha encostada à esquerda",
                    "pilha 5 - esquerda"))
    if not covered["direita"]:
        out.append((2, "pilha encostada à direita",
                    "pilha 5 - direita"))
    if not covered["noite"]:
        out.append((3, "contraluz / pouca luz",
                    "pilha 8 - centro - noite"))
    if not covered["pessoa"]:
        out.append((4, "pessoa passando na frente (oclusão)",
                    "pilha 8 - centro - pessoa na frente"))

    out.sort(key=lambda x: x[0])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.plan_capture",
        description="Planeja as próximas capturas de dataset",
    )
    p.add_argument("--json", action="store_true", help="Saída em JSON")
    args = p.parse_args(argv)

    scenes = load_scenes()
    names = [s.get("name", "") for s in scenes]
    quantities = extract_quantities(names)
    covered = cover(names)
    tips = suggest(quantities, covered)

    if args.json:
        print(json.dumps({
            "scenes": len(scenes),
            "images": sum(int(s.get("count", 0)) for s in scenes),
            "quantities": quantities,
            "spread": (round(max(quantities) / min(quantities), 1)
                       if len(quantities) > 1 else 0),
            "coverage": covered,
            "suggestions": [{"prio": pr, "label": lb, "scene": sc} for pr, lb, sc in tips],
        }, ensure_ascii=False, indent=2))
        return 0

    n_img = sum(int(s.get("count", 0)) for s in scenes)
    print()
    print("=" * 68)
    print("  PLANO DE CAPTURA")
    print("=" * 68)
    print(f"  cenas registradas ... {len(scenes)}")
    print(f"  imagens ............. {n_img}")
    print()

    # --- espalhamento ---
    print("  Quantidades de cadeiras já vistas:")
    if quantities:
        print(f"    {quantities}")
        if len(quantities) > 1:
            spread = max(quantities) / min(quantities)
            print(f"    espalhamento: {spread:.1f}x  (de {min(quantities)} a {max(quantities)})")
            if spread < 4:
                print("    ⚠ poco. O modelo nunca viu a quantidade mudar de verdade.")
                print("      Alvo: pelo menos 4x, com os extremos (1 e ~20).")
            elif spread < 10:
                print("    ⚗ razoável. Mais amplitude no extremo alto ajuda.")
            else:
                print("    ✓ bom: o modelo vai ver 'mais cadeiras' como conceito.")
        else:
            print("    ⚠ só um valor. Isso não é variedade de quantidade, é uma cena só.")
    else:
        print("    (nenhuma quantidade encontrada nos nomes das cenas)")
        print("    Dica: nomeie como 'pilha 8 - esquerda' para o plano ficar preciso.")
    print()

    # --- cobertura ---
    print("  Cobertura por dimensão:")
    labels = {
        "vazio": "cena vazia (sem cadeira)",
        "esquerda": "pilha à esquerda",
        "centro": "pilha ao centro",
        "direita": "pilha à direita",
        "dia": "com luz",
        "noite": "pouca luz / noite",
        "pessoa": "oclusão (pessoa na frente)",
    }
    for key, label in labels.items():
        mark = "✓" if covered.get(key) else "·"
        print(f"    {mark} {label}")
    print()
    print("  (leitura feita a partir do NOME das cenas — renomeie se precisar de precisão)")
    print()

    # --- próximos passos ---
    print("  Capture a seguir, nesta ordem:")
    for prio, label, scene in tips:
        print(f"    [{prio}] {label}")
        print(f"         python -m tools.collect_dataset --scene \"{scene}\" --seconds 20")
    print()
    print("  Reposicione as cadeiras entre uma captura e outra: a ferramenta")
    print("  descarta imagens que repitam a cena anterior.")
    print()
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
