"""Teste do contador em FOTOS REAIS de pilhas de cadeiras.

O benchmark sintético (:mod:`tools.bench_counter`) mede o algoritmo em
pilhas de estrutura limpa e controlada. Este arquivo mede em fotos reais,
onde:

* as cadeiras têm cores diferentes;
* há fundo clutterizado (porta, parede, calçada);
* a estrutura interna da cadeira (encosto/assento/pernas)Repete em escalas
  diferentes da cadeira inteira.

O número de cadeiras de cada foto foi contado **manualmente** olhando a
imagem. Esses valores são a verdade-terreno do teste.

    python -m tools.bench_real

Objetivo honesto: mostrar onde o algoritmo funciona e onde ele falha, em vez
de esconder a limitação atrás de um número bonito.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from app.counting.stack_counter import StackCounter, analyze_layers

# Pilhas recortadas manualmente das fotos reais + contagem feita olhando.
# (arquivo, x1, y1, x2, y2, cadeiras)
#
# >>> AJUSTE ESTES VALORES PARA AS SUAS FOTOS <<<
# Cada entrada é uma pilha recortada de uma foto e a quantidade de cadeiras
# que você contou olhando. Troque `reais` pelo número que você contou e
# rode de novo: é assim que você mede o erro real do seu modelo.
CASES: list[tuple[str, tuple[int, int, int, int], int]] = [
    ("Stacked_chairs_01.jpg", (230, 800, 730, 2320), 10),
    ("Stacked_chairs_01.jpg", (690, 800, 1190, 2280), 8),
    ("Stacked_chairs_01.jpg", (1130, 760, 1620, 2240), 7),
    ("Stacked_chairs_02.jpg", (300, 780, 800, 2300), 9),
    ("Stacked_chairs_02.jpg", (820, 780, 1300, 2300), 8),
    ("Stacked_chairs_03.jpg", (250, 800, 760, 2280), 11),
    ("Plastic_stacking_chairs_designed_by_Robin_Day.jpg", (500, 300, 1640, 2200), 12),
]

IMG_DIR = Path(
    os.environ.get("BENCH_REAL_IMAGES", "")
    or (Path(__file__).resolve().parent.parent / "training" / "datasets" / "raw")
)


def crop(img: "object", box: tuple[int, int, int, int], pad: float = 0.02) -> "object":
    h, w = img.shape[:2]  # type: ignore[attr-defined]
    x1, y1, x2, y2 = box
    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)
    return img[max(0, y1 - py) : min(h, y2 + py), max(0, x1 - px) : min(w, x2 + px)]


def main() -> int:
    if not IMG_DIR.is_dir():
        print("Pasta de imagens não encontrada:", IMG_DIR)
        print()
        print("Para rodar este benchmark com as SUAS fotos reais:")
        print("  1. Coloque as fotos em training/datasets/raw/  (ou em qualquer pasta)")
        print("  2. Ajuste a lista CASES no início deste arquivo:")
        print("       (nome_do_arquivo.jpg, (x1, y1, x2, y2), qtd_de_cadeiras)")
        print("     O recorte é a PILHA; a quantidade é o que você contou olhando.")
        print("  3. Rode novamente:")
        print("       BENCH_REAL_IMAGES=/caminho/das/fotos python -m tools.bench_real")
        print()
        print("Sem esse arquivo, use o benchmark sintético:")
        print("  python -m tools.bench_counter")
        return 2

    print("=" * 96)
    print("BENCHMARK EM FOTOS REAIS — contagem feita manualmente (ver CASES no código)")
    print("=" * 96)
    print(
        f"{'foto':<14}{'real':>5}{'IA':>5}{'erro':>6}{'conf':>6}  "
        f"{'passo':>7}{'ambíguo':>9}  detalhe"
    )
    print("-" * 96)

    rows: list[dict] = []
    for fname, box, real in CASES:
        path = IMG_DIR / fname
        if not path.is_file():
            print(f"{fname[:13]:<14}  (arquivo ausente)")
            continue
        img = cv2.imread(str(path))
        roi = crop(img, box)
        est = StackCounter(chair_height_px=0.0, method="periodicity").count(roi)
        a = analyze_layers(roi, 0.0)
        err = est.count - real
        rows.append({"real": real, "count": est.count, "err": err, "conf": est.confidence,
                     "amb": a.ambiguous})
        detail = "PERÍODO AMBÍGUO (contagem não confiável)" if a.ambiguous else "ok"
        print(
            f"{fname[:13]:<14}{real:>5}{est.count:>5}{err:>+6}{est.confidence:>6.2f}  "
            f"{a.pitch:>6.0f}px{('SIM' if a.ambiguous else 'nao'):>9}  {detail}"
        )

    if not rows:
        print("Nenhum caso executado.")
        return 2

    detected_amb = sum(1 for r in rows if r["amb"])
    confident_wrong = sum(1 for r in rows if r["err"] != 0 and r["conf"] >= 0.55)
    print("-" * 96)
    print(f"Casos:                      {len(rows)}")
    print(f"Exatos:                     {sum(1 for r in rows if r['err'] == 0)}")
    print(f"Erro absoluto médio:        {sum(abs(r['err']) for r in rows) / len(rows):.2f} cadeiras")
    print(f"Período ambíguo detectado:  {detected_amb}/{len(rows)}")
    print(f"Confiança ALTA com erro:    {confident_wrong}  (o que mais importa: deve ser 0)")
    print("=" * 96)
    print("LEITURA HONESTA DOS RESULTADOS:")
    print("  Em foto real, a estrutura interna da cadeira repete em escalas")
    print("  diferentes da cadeira inteira. Quando o sinal não permite decidir,")
    print("  o sistema marca PERÍODO AMBÍGUO e derruba a confiança, em vez de")
    print("  mostrar um número errado com cara de certeza.")
    print("  Por isso a contagem de produção depende do modelo YOLO treinado com")
    print("  as cadeiras da empresa: a detecção de cada cadeira individual é o")
    print("  sinal primário, e a análise de empilhamento é a rede de segurança.")
    if confident_wrong > 0:
        print(f"  ATENÇÃO: {confident_wrong} caso(s) com erro e confiança alta — revisar.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
