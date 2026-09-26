"""Contagem de cadeiras empilhadas - o coração do sistema.

Este módulo recebe **somente a região (ROI) de uma pilha** e estima quantas
cadeiras existem nela. Ele não confia exclusivamente na detecção YOLO,
porque em uma pilha a maioria das cadeiras está parcialmente escondida.

Estratégia híbrida
==================

Quatro estimadores independentes são calculados e depois fundidos:

``detections``
    Soma das detecções YOLO de cadeira dentro da pilha, ponderada por
    ``chair_weight``. É exato quando o modelo enxerga todas as cadeiras e
    subestima quando há oclusão.

``peaks``
    Contagem direta das camadas visíveis. Um perfil de bordas horizontais é
    calculado para a ROI; os picos locais de energia correspondem às
   -separações entre cadeiras (encosto/assento/pernas).

``period``
    Extrapolação pelo **período vertical** do empilhamento: a pilha é um
    padrão repetido, então ``quantidade ≈ extensão / passo``. O passo é
    estimado por correlação automática + ajuste de pente (*comb score*),
    o que é robusto a camadas que o YOLO não viu.

``size``
    ``altura da pilha / altura de uma cadeira``. Só fica disponível quando a
    altura da cadeira em pixels é conhecida (calibração ou derivada das
    detecções do próprio frame).

Os quatro geram um valor e uma **confiança própria**. A fusão ponderada
produz ``(contagem, confiança)``, e a confiança final também incorpora o
acordo entre os estimadores.

Calibração
==========

A altura de uma cadeira em pixels é um dado **medido**, nunca inventado:

* informada pelo operador na página de Calibração
  (``chair_height_px``), ou
* derivada das detecções de cadeiras do próprio frame (mediana das alturas),
  ou
* ``chair_height_px = 0`` significa "não calibrado" - nesse caso os
  estimadores ``size`` e a faixa de busca do ``period`` ficam restritos.

Além disso existe um ``count_offset`` inteiro (default 0) que corrige
diferenças sistemáticas de "uma cadeira". O sistema **sugere** esse valor a
partir das correções manuais registradas (IA vs. operador), em vez de
chutar: :func:`suggest_offset`.

Escolha do método
=================

``COUNTING_METHOD`` no .env:

* ``auto``     - usa todos os estimadores disponíveis e funde (padrão)
* ``fused``    - igual ao auto, mas exige concordância mínima
* ``detections`` / ``periodicity`` / ``size`` - força um estimador
"""

# ===========================================================================
# ARQUIVO / MAPA  -  stack_counter.py
# ===========================================================================
# Leitura na ordem das seções numeradas abaixo.
#
# 1. Processamento de imagem do ROI   (linhas ~93-215)
#    - preprocess_roi: CLAHE para matar variação de luz do galpão.
#    - row_edge_profile: perfil 1D de gradiente vertical -> "impressão
#      digital" da pilha (uma linha por y).
#    - detrend: tira a tendência lenta (sombra/iluminação).
#    - find_peaks / _find_peaks_scipy: picos por PROEMINÊNCIA (não por
#      altura), que é o que separa camada real de ruído.
#    1ª coisa: sem sinal 1D limpo não existe contagem por padrão.
#
# 2. Estimativa do passo (período vertical)   (~218-504)
#    - estimate_period: autocorrelação (palpite) + comb score (refino).
#    - _comb_score: encaixa um pente no sinal. Sabe que o período escolhido
#      pode ser múltiplo do verdadeiro - por isso é só palpite.
#    - estimate_period_from_peaks: passo pelos espaçamentos entre picos.
#      Escolhe o MAIOR candidato com suporte, não o mais pontuado.
#    - periodic_extent: mede só o trecho que realmente se repete e conta
#      as camadas. ESTE é o estimador que decide a contagem real.
#    - period_ambiguity: metade/dobro/terço do passo explicam o sinal?
#      Se sim, declara "indeterminado" em vez de chutar.
#
# 3. Análise de camadas   (~507-809)
#    - LayerAnalysis: dataclass com 3 estimadores independentes
#      (n_lattice = amplitude, n_extent = geometria, n_peaks_raw = só
#      diagnóstico) + métricas de qualidade.
#    - otsu_threshold: separa pilha de fundo sem constante mágica.
#    - active_extent: faixa vertical com estrutura, sem usar picos.
#    - count_lattice_peaks: conta SLOTS do pente (1 por cadeira), não
#      todos os picos.
#    - analyze_layers: orquestra tudo acima e escolhe a faixa de passo a
#      partir da altura da cadeira (calibrada ou medida no frame).
#
# 4. Contador   (~812-1164)
#    - StackCounter.count: roda os 4 estimadores, funde por pesos e devolve
#      (contagem, confiança, candidatos, notas legíveis).
#    - _fuse: média ponderada + MEDIDA DE ACORDO entre estimadores.
#    - _final_confidence: 0.45*acordo + 0.40*conf.média + 0.15*cobertura,
#      com penalidades (padrão não confirmado, período ambíguo, salto vs
#      frame anterior) e teto quando existe detecção YOLO.
#    - diagnose: despeja a análise inteira (dashboard / testes).
#
# 5. Sugestão de offset   (~1167-1203)
#    - suggest_offset: offset = mediana de (humano - IA) nas correções
#      manuais. Só devolve se houver amostras suficientes.
#
# Ordem de leitura sugerida: seção 3 (o que cada estimador mede) ->
# seção 4 (como funde) -> seção 2 (por que o passo é o número-chave) ->
# seção 1 (detalhes de imagem).
#
# Convenção do arquivo: posições de pico e passo são SEMPRE em pixels (y do
# ROI), nunca em pixels absolutos da imagem. A ROI já foi recortada antes.
# ===========================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from app.config import settings
from app.schemas import CountEstimate, Detection, PileCandidate

log = logging.getLogger("ai")

# --------------------------------------------------------------------- limites
MIN_ROI_H = 12          # abaixo disso não há como contar
MIN_ROI_W = 8
PERIOD_MIN_PX = 3.0     # menor passo plausível entre duas cadeiras
# Proeminência mínima do pico, em 0..1 do perfil já normalizado. Abaixo
# disso o "pico" é granulação de sensor ou ruído de compressão.
PEAK_MIN_PROMINENCE = 0.12
# Constante legada: sobrou da versão que derivava a distância mínima entre
# picos a partir do passo. Fica como documentação da ordem de grandeza
# (0,3 x passo) e como aviso: não reintroduza um min_distance grande aqui.
PEAK_MIN_DIST_FRAC = 0.30   # distância mínima entre picos = 30% do passo

# Quanto da altura de UMA cadeira ela acrescenta a uma pilha.
# Cadeiras plásticas encaixadas: ~0,03 (medido nas fotos do galpão).
# Cadeiras empilhadas frouxamente: ~0,6-0.7.
# A faixa é larga de propósito: o mesmo sistema roda nos dois tipos de
# pilha e não sabe qual é o caso antes de contar.
NEST_MIN_RATIO = 0.02
NEST_MAX_RATIO = 1.10


# =========================================================================== #
# 1. Processamento de imagem do ROI
# =========================================================================== #
def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Prepara a ROI para análise de padrão vertical.

    Equalização adaptativa (CLAHE) paraczyto a variação de iluminação do
    ambiente, que é a maior fonte de erro em contagem por padrão.
    """
    if roi_bgr.ndim == 2:
        gray = roi_bgr
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # clipLimit=2.0 é Conservative de propósito: com clip mais alto o CLAHE
    # realça também o granulado da câmera e cria picos falsos no perfil.
    # tileGridSize 8x8 dá ~1 bloco por 1/8 da pilha, compensando o topo
    # iluminado e a base na sombra.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def row_edge_profile(gray: np.ndarray) -> np.ndarray:
    """Perfil 1D de energia de borda horizontal, por linha (y).

    Cadeiras empilhadas produzem linhas horizontais repetidas (borda do
    encosto, do assento, das pernas). Somar o gradiente vertical ao longo
    de X produz um sinal 1D cujo padrão periódico é a "impressão digital"
    da pilha.
    """
    if gray.ndim == 3:
        # Aceita BGR/BGRA também: quem chama pode não ter convertido antes.
        code = cv2.COLOR_BGRA2GRAY if gray.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(gray, code)
    # Sobel só na direção Y (dy=1, dx=0): as cadeiras empilhadas criam bordas
    # horizontais, e o gradiente horizontal (pernas, Tubos) só poluiria o sinal.
    # ksize=3 = operador de Sobel clássico, mais estável que a derivada simples.
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Média ao longo de X: cada y vira UM número = energia horizontal da linha.
    # Isso é o que faz o perfil ser 1D e ignorável a posição lateral.
    profile = np.abs(grad_y).mean(axis=1)
    if profile.size == 0:
        return profile
    # Normaliza para 0..1 (escala independe do contraste da cena).
    # Usa o percentil 99 e não o máximo: um único ponto de reflexo (lâmpada,
    # plástico brilhoso) no máximo estouraria a escala e achataria todo o resto.
    peak = float(np.percentile(profile, 99))
    if peak > 1e-6:
        profile = profile / peak
    return profile.astype(np.float32)


def detrend(profile: np.ndarray, window: int | None = None) -> np.ndarray:
    """Remove a tendência lenta (iluminação / sombra) do perfil.

    Sem isso, um gradiente de luz forte faz o perfil parecer "mais forte em
    cima" e a contagem de picos sai errada.
    """
    n = profile.size
    if n < 5:
        return profile.astype(np.float32)
    if window is None:
        # n/6: a janela precisa ser bem mais larga que o passo (poucas dezenas
        # de px) e bem mais estreita que a pilha (centenas), senão a média
        # moparia os próprios picos. 6 blocos na altura é o meio-termo.
        # O "| 1" força janela ímpar: kernel par faz a convolução "same"
        # deslocar o perfil em meio pixel e enviesaria a fase do pente.
        window = max(7, (n // 6) | 1)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return profile.astype(np.float32)
    # Média móvel = a própria suavização. Subtrair é o detrend mais simples
    # que existe: não assume forma de tendência (reta, polinômio), só que
    # ela é lenta.
    kernel = np.ones(window, dtype=np.float32) / float(window)
    trend = np.convolve(profile, kernel, mode="same")
    return (profile - trend).astype(np.float32)


def find_peaks(
    signal: np.ndarray, min_prominence: float = PEAK_MIN_PROMINENCE, min_distance: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Picos de um sinal 1D, com proeminência mínima.

    Usa o mesmo conceito de proeminência do SciPy (topo de morfo), que é
    exatamente o que distingue uma camada real de ruído.
    """
    if signal.size < 3:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    idx, prom, _ = _find_peaks_scipy(signal, min_prominence=min_prominence)
    if idx.size and min_distance > 1:
        # Filtro guloso de distância: fica com o pico mais proeminente de cada
        # janela. Não é o SciPy "peak_widths", mas dá o mesmo resultado neste
        # sinal (os picos de cadeira estão bem separados).
        keep = [0]
        for i in range(1, idx.size):
            if idx[i] - idx[keep[-1]] >= min_distance:
                keep.append(i)
        idx = idx[keep]
        prom = prom[keep]
    heights = signal[idx] if idx.size else np.array([], dtype=float)
    return idx, prom, heights


def _find_peaks_scipy(
    signal: np.ndarray, min_prominence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Picos com proeminência, implementação própria (evita SciPy).

    Proeminência de um pico = ``altura_do_pico - max(base_esquerda, base_direita)``,
    onde cada base é o menor valor percorrido até encontrar um pico mais alto
    (ou o fim do sinal).

    A implementação anterior comparava o pico com os máximos laterais, o que
    dava proeminência zero justamente para o pico mais alto da pilha - ou
    seja, some com a camada mais visível.
    """
    n = signal.size
    indices: list[int] = []
    prominences: list[float] = []
    # Varredura O(n) por pico, mas as buscas de base são O(n) cada: no pior
    # caso (sinal todo liso) isto é O(n²). n é a altura da ROI (dezenas/centenas
    # de px), então cabe folgado no orçamento por frame. Se a ROI algum dia
    # passar de milhares de px, trocar por stack de mínimos.
    for i in range(1, n - 1):
        # Definição de pico local: sobe em relação à esquerda e não desce em
        # relação à direita. As bordas (i=0 e i=n-1) ficam de fora de propósito.
        if not (signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]):
            continue
        height = float(signal[i])
        # base à esquerda: caminha até achar um pico mais alto; o menor valor
        # do caminho é a "base" (o vale mais fundo antes de subir de novo).
        left_base = height
        for j in range(i - 1, -1, -1):
            if signal[j] > height:
                break
            left_base = min(left_base, float(signal[j]))
        # base à direita: mesma coisa pelo outro lado
        right_base = height
        for j in range(i + 1, n):
            if signal[j] > height:
                break
            right_base = min(right_base, float(signal[j]))
        # Proeminência = quanto o pico "sobresai" do vale mais próximo dos
        # dois lados. Usa max() das bases: o vale MENOR entre os dois lados é
        # o que faz este pico ser proeminente.
        prom = height - max(left_base, right_base)
        if prom >= min_prominence:
            indices.append(i)
            prominences.append(max(0.0, prom))
    return (
        np.array(indices, dtype=int),
        np.array(prominences, dtype=float),
        np.array([signal[i] for i in indices], dtype=float),
    )


# =========================================================================== #
# 2. Estimativa do passo (período vertical)
# =========================================================================== #
def estimate_period(
    signal: np.ndarray, min_period: float, max_period: float
) -> tuple[float, float, float]:
    """Estima o passo do padrão repetido.

    Retorna ``(passo, qualidade 0..1, fase)``. A fase é a posição (em px) do
    primeiro pico do pente que melhor se encaixa no sinal - usada depois
    para contar as camadas alinhadas a esse pente.

    Duas etapas complementares:

    1. **Autocorrelação** - o pico da autocorrelação na faixa ``[min,max]``
       dá uma estimativa grosseira, robusta.
    2. **Comb score** - para o passo candidato, procura a fase que maximiza
       ``média(sinal[picos]) - média(sinal[vales entre picos])``. Isso mede
       quão bem um pente se encaixa no sinal, e vale para padrão com
       período que não é perfeitamente senoidal.
    """
    n = signal.size
    if n < 8:
        return 0.0, 0.0, 0.0
    # Reduz a faixa de busca a lags INTEIROS: abaixo de 2 px não dá para
    # distinguir cadeira, e acima de n-2 o lag não tem base para comparar.
    lo = int(max(2, math.floor(min_period)))
    hi = int(min(n - 2, math.ceil(max_period)))
    if hi <= lo:
        return 0.0, 0.0, 0.0

    # Autocorrelação só faz sentido em sinal de média zero: senão a linha de
    # base_some e o lag 0 (trivialmente perfeito) ganha de todo mundo.
    x = signal - float(np.mean(signal))
    if float(np.max(np.abs(x))) < 1e-6:
        return 0.0, 0.0, 0.0

    # --- 1. autocorrelação (FFT, O(n log n)) --------------------------------
    # nfft >= 2n-1: sem zero-padding, o rfft "enrolaria" a convolução e a
    # autocorrelação sairia circular (o lag 0 contaminaria todos os outros).
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, nfft)
    # Teorema de Parseval: |X(f)|² é a autocorrelação via rfft → irfft. Em vez
    # de um np.correlate O(n²) do sinal inteiro.
    acf = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:n]
    # Normaliza: acf[0] é a energia total do sinal, e a correlação tem de ser
    # medida como fração dela, senão o valor absoluto depende do contraste.
    acf = acf / max(1e-9, acf[0])
    window = acf[lo : hi + 1]
    if window.size == 0:
        return 0.0, 0.0, 0.0
    # A autocorrelação é ruidosa (periódica E harmônica, 2x, 3x...), então ela
    # só dá um palpite. Nunca é o passo final.
    best_lag = int(np.argmax(window)) + lo
    acf_score = float(window.max())

    # --- 2. refino local do passo com comb score ----------------------------
    best_p, best_phase, best_score = float(best_lag), 0.0, 0.0
    # Só refina ±20% em torno do palpite, em passos de 0,5 px. Refinar a faixa
    # toda seria caro e, como as harmônicas já estão na faixa, poderia cair
    # num múltiplo pior. 0,5 px é ~1% do passo típico: abaixo disso é ruído.
    search = np.arange(max(lo, best_lag * 0.80), min(hi, best_lag * 1.20) + 0.5, 0.5)
    if search.size == 0:
        search = np.array([float(best_lag)])
    for period in search:
        phase, score = _comb_score(signal, float(period))
        if score > best_score:
            best_score, best_p, best_phase = score, float(period), float(phase)
    if best_score <= 0.0:
        # Nenhum pente encaixou (pior que zero = não há periodicidade real).
        # Cai no palpite da autocorrelação, com a qualidade dela como score.
        best_p, best_phase = float(best_lag), 0.0
        best_score = max(0.0, acf_score)

    # Qualidade: combinação da força da autocorrelação com o encaixe do pente.
    # 0.45/0.55: o pente pesa mais porque é ele que é refinado; a
    # autocorrelação entra mais fraca porque só delimitou a região de busca.
    quality = float(np.clip(0.45 * max(0.0, acf_score) + 0.55 * best_score, 0.0, 1.0))
    return best_p, quality, best_phase


def _comb_score(signal: np.ndarray, period: float, phase_step: float = 1.0) -> tuple[float, float]:
    """Encaixe de um pente de período ``period`` no sinal.

    ``score = média(sinal nos picos) - média(sinal no meio dos vãos)``

    NOTA sobre por que isso não basta sozinho: um múltiplo do período
    verdadeiro (27 px → 54, 81, 400...) também "encaixa", porque um pente
    largo amostra menos pontos e erra menos. Por isso o período escolhido
    aqui é apenas um *palpite inicial*; quem decide a contagem é
    :func:`periodic_extent`, que mede o quanto o padrão realmente se repete.
    """
    n = signal.size
    if period < 2 or n < 2 * period:
        # n >= 2*period: com menos disso não há nem um pico e um vale, e o
        # score vira ruído. É o que impede "achar" passo maior que a pilha.
        return 0.0, 0.0
    half = period / 2.0
    # phase_step=1 px: suficiente porque o sinal já é amostrado 1x por linha
    # (não dá para refinar fase mais que isso sem interpolar).
    phases = np.arange(0.0, period, phase_step)
    if phases.size == 0:
        return 0.0, 0.0
    k = np.arange(0, int(np.ceil(n / period)) + 1)
    pos = phases[:, None] + k[None, :] * period          # (fases, k)
    valid = pos < n
    counts = valid.sum(axis=1)
    # Uma fase só é confiável com >= 3 amostras. Com 1 ou 2, o "encaixe" é
    # puro acaso: um pico alto escolhido na fase certa daria score máximo.
    usable = counts >= 3
    if not usable.any():
        return 0.0, 0.0

    # Índices com clip: nos últimos pixels a fase pode "vazar" para fora do
    # sinal, e quem valida é `valid`, não o clip.
    idx = np.clip(np.round(pos).astype(int), 0, n - 1)
    peak_means = np.where(valid, signal[idx], 0.0).sum(axis=1) / np.maximum(counts, 1)

    # Vales = meio do caminho entre dois picos. Comparar pico vs vale (e não
    # contra 0) é o que torna o score imune ao offset do sinal.
    vpos = pos + half
    vvalid = vpos < n
    vcounts = vvalid.sum(axis=1)
    vidx = np.clip(np.round(vpos).astype(int), 0, n - 1)
    valley_means = np.where(vvalid, signal[vidx], 0.0).sum(axis=1) / np.maximum(vcounts, 1)

    # -1e9 descarta as fases inutilizáveis sem precisar de máscara booleana.
    scores = np.where(usable, peak_means - valley_means, -1e9)
    i = int(np.argmax(scores))
    return float(phases[i]), float(scores[i])


def estimate_period_from_peaks(
    peaks: Sequence[float],
    min_spacing: float = 3.0,
    tol: float = 0.30,
    min_period: float = 0.0,
    max_period: float = 0.0,
) -> tuple[float, float, int]:
    """Escolhe o período de empilhamento a partir dos espaçamentos entre picos.

    Dois casos reais e opostos precisam ser tratados:

    * **Cadeira encaixada** (galpão, medido): 1 cadeira = 1 borda dominante.
      O passo de 27 px tinha 20 espaçamentos compatíveis, contra 1–2 dos
      múltiplos. O encaixe de pente, ao contrário, *preferia* os múltiplos
      (400 px pontuava mais que 27 px), porque um pente largo amostra menos
      pontos e erra menos.

    * **Cadeira com varias bordas** (encosto + assento + pernas): 1
      cadeira = 3-4 bordas a ~1/3 do passo. Aí o espaçamento mais frequente é
      a **sub**-harmônica, e escolher o mais repetido contaria 3–4 vezes.

    Por isso a escolha é: entre os candidatos com suporte relevante,
    **o maior**. O padrão de fundo de uma cadeira é sempre o *menor*
    intervalo (~1/3 do passo); os múltiplos do passo, ao contrário, quase
    nunca se repetem de verdade. Escolher o maior candidato com bom suporte
    acerta os dois casos.

    Devolve ``(período, qualidade 0..1, nº de espaçamentos que apoiam)``.
    """
    if len(peaks) < 3:
        return 0.0, 0.0, 0
    # Ordena: os espaçamentos só fazem sentido entre picos vizinhos.
    pts = np.sort(np.asarray(peaks, dtype=float))
    diffs = np.diff(pts)
    # Descarta espaçamentos pequenos demais: entre duas bordas da MESMA
    # cadeira (encosto/assento/pernas) há poucos px e isso não é um passo.
    diffs = diffs[diffs >= min_spacing]
    # Filtro físico: só interessam passos dentro da faixa plausível para uma
    # cadeira. É aqui que a calibração (chair_height_px) ganha poder.
    if min_period > 0:
        diffs = diffs[diffs >= min_period]
    if max_period > 0:
        diffs = diffs[diffs <= max_period]
    if diffs.size == 0:
        return 0.0, 0.0, 0

    # Candidatos = valores de espaçamento arredondados a 0,1 px. O arredondamento
    # agrupa 27,0 e 27,4 no mesmo candidato, senão cada par seria um grupo só.
    cands = np.unique(np.round(diffs, 1))
    # Support = quantos espaçamentos estão dentro de ±tol do candidato.
    # tol=0.30: 30% é folga para pixelização, perspectiva e sombra entre
    # camadas, mas ainda menor que 2x (senão 27 e 54 seriam o mesmo candidato).
    supports = np.array([int(np.sum(np.abs(diffs - c) <= c * tol)) for c in cands], dtype=float)
    if supports.max() <= 0:
        return 0.0, 0.0, 0

    # Relevance: pelo menos 40% do melhor suporte
    relevant = supports >= 0.40 * supports.max()
    if relevant.any():
        best_p = float(cands[relevant].max())   # o MAIOR relevante
        best_support = int(supports[relevant][int(np.argmax(cands[relevant]))])
    else:
        best_p = float(cands[int(np.argmax(supports))])
        best_support = int(supports.max())
    if best_support < 2:
        # Um único par repetido não é padrão, é coincidência.
        return 0.0, 0.0, best_support

    # Confiança: quantos dos espaçamentos observados seguem o passo escolhido,
    # normalizado por max(3, metade dos espaçamentos). O 3 evita qualidade alta
    # com pouquíssimas amostras; a metade evita que 1 outlier derrube a nota.
    own = float(np.sum(np.abs(diffs - best_p) <= best_p * tol))
    quality = float(min(1.0, own / max(3.0, 0.5 * diffs.size)))
    return best_p, quality, int(own)


def periodic_extent(peaks: Sequence[float], pitch: float, tol: float = 0.35) -> dict[str, Any]:
    """Mede o trecho em que o padrão realmente se repete, e conta as camadas.

    Este é o estimador que decide a contagem em pilhas reais.

    O erro clássico da análise de empilhamento é dividir a **altura total**
    da pilha pelo passo. Numa pilha de cadeiras encaixadas isso superestima,
    porque a região das pernas no fundo tem estrutura diferente e não se
    repete com o mesmo passo: ali estão menos assentos do que o quociente
    da altura sugere.

    Aqui medimos só o que é periódico de fato: os picos ligados por
    espaçamentos compatíveis com o passo, do primeiro ao último. Picos
    isolados (ruído) e trechos com outro espaçamento (as pernas) ficam de
    fora, mas uma lacuna de um ou dois picos **no meio** da sequência não
    interrompe a contagem - Happens o bastante em imagem real com sombra.
    """
    out: dict[str, Any] = {"count": 0, "span": 0.0, "top": 0.0, "bottom": 0.0,
                           "coverage": 0.0, "regularity": 0.0}
    if pitch <= 0 or len(peaks) < 2:
        return out
    pts = np.asarray(sorted(peaks), dtype=float)
    diffs = np.diff(pts)
    if diffs.size == 0:
        return out
    med = float(np.median(diffs))
    if med <= 0:
        return out

    # Espaçamentos compatíveis com o passo. tol=0.35 é folgado de propósito:
    # numa pilha real o espaçamento varia com a perspectiva (mais embaixo a
    # cadeira aparece "maior" que a de cima). O piso de 1.0 px evita que a
    # tolerância suma quando o passo é minúsculo.
    good = np.abs(diffs - pitch) <= max(1.0, pitch * tol)
    if not good.any():
        return out

    # Tolerância a 1-2 picos faltando no meio: dilata a máscara uma vez
    # (dilatação morfológica booleana). Shift de 1 e de 2 = aceita 1 ou 2
    # lacunas; mais que isso já é outra estrutura, não uma sombra.
    keep = good.copy()
    for shift in (1, 2):
        shifted = np.zeros_like(good)
        if shift < good.size:
            shifted[shift:] = good[:-shift]
        keep |= shifted
        keep |= np.roll(shifted, -shift) if shift < good.size else False

    used = np.where(keep)[0]
    if used.size == 0:
        return out
    # Primeiro e último pico conectados por espaçamentos válidos
    # +1 no índice de baixo porque diffs[i] liga pts[i] -> pts[i+1].
    top_idx, bottom_idx = int(used[0]), int(used[-1] + 1)
    top, bottom = float(pts[top_idx]), float(pts[bottom_idx])
    span = bottom - top
    if span <= 0:
        return out

    out.update(
        {
            # +1 porque N camadas dão N-1 intervalos entre elas. É o mesmo
            # "+1" que gera o erro clássico de contar N+1 linhas como N+1
            # cadeiras quando se usa a contagem de picos crua.
            "count": int(round(span / pitch)) + 1,
            "span": span,
            "top": top,
            "bottom": bottom,
            # Regularidade = fração de espaçamentos válidos. Serve de
            # diagnóstico: 1.0 = pilha bem uniforme, 0.3 = pilha "suja".
            "regularity": float(good.sum()) / float(good.size),
        }
    )
    return out


def period_ambiguity(
    signal: np.ndarray, period: float, min_period: float, max_period: float, tol: float = 0.78
) -> dict[str, Any]:
    """Detecta se o passo encontrado é uma harmônica (metade ou o dobro).

    Uma cadeira empilhada tem bordas horizontais tanto na escala da cadeira
    inteira quanto na sua estrutura interna (encosto, assento, pernas). Se o
    encaixe de ``period/2`` ou ``2*period`` for quase tão bom quanto o de
    ``period``, o sinal **não consegue dizer** qual é a cadeira - e a
    contagem seria uma loteria.

    Nesse caso devolvemos a ambiguidade explicitamente, para que o sistema
    exiba "indeterminado / baixa confiança" em vez de um número errado com
    aparência de certeza. Essa é a diferença entre um sistema confiável e um
    que erra em silêncio.
    """
    info: dict[str, Any] = {"ambiguous": False, "chosen": period, "options": []}
    if period <= 0:
        return info
    ref = _comb_score(signal, period)[1]
    if ref <= 0:
        return info

    options: list[tuple[float, float]] = []
    # Fatores testados: 1/2, 2, 1/3 e 3. Cobre (a) metade do passo = erro de
    # 2x na contagem, (b) passo duplo = metade da contagem, (c)/(d) a
    # sub-divisão em 3 ou a super-divisão, que é o que acontece quando a
    # "cadeira" real tem 3 bordas horizontais.
    for factor in (0.5, 2.0, 1.0 / 3.0, 3.0):
        cand = period * factor
        # Candidatos fora da faixa física (×0.9/×1.1 = folga de arredondamento)
        # são descartados: não há como a cadeira medir 0,4 px.
        if cand < max(2.0, min_period * 0.9) or cand > max_period * 1.1:
            continue
        # Descarta o próprio período e valores a menos de 1 px (mesma coisa).
        if abs(cand - period) < 1.0:
            continue
        score = _comb_score(signal, cand)[1]
        # tol=0.78: se a harmônica chega a 78% do encaixe do passo escolhido,
        # o sinal não decide. Limiar alto de propósito - preferimos dizer
        # "indeterminado" a exibir um número 2x errado com cara de certeza.
        if score >= tol * ref:
            options.append((cand, score))
    if not options:
        return info

    options.sort()
    # Escolha do passo ambíguo:
    #  - se a calibração já exclui os candidatos menores (eles estão abaixo do
    #    mínimo fisicamente possível), o maior válido é a cadeira;
    #  - caso contrário não há critério válido, então ficamos com o de maior
    #    encaixe e marcamos o resultado como indeterminado.
    valid = [c for c, _ in options if min_period * 0.95 <= c <= max_period * 1.05]
    # PERIOD_MIN_PX*1.5 = 4,5 px: só acima disso a faixa de busca é estreita
    # o bastante para dizer "isto não cabe na física" de verdade.
    if valid and min_period > PERIOD_MIN_PX * 1.5:
        chosen = max(valid)
    else:
        chosen = max(options, key=lambda t: t[1])[0]
    info.update(
        {
            "ambiguous": True,
            "chosen": chosen,
            "options": [{"period": round(c, 1), "score": round(s, 4)} for c, s in options],
            "reason": "várias escalas de repetição explicam o sinal igualmente bem",
            "calibrated": min_period > PERIOD_MIN_PX * 1.5,
        }
    )
    return info


# =========================================================================== #
# 3. Análise de camadas
# =========================================================================== #
@dataclass
class LayerAnalysis:
    """Resultado da análise de padrão vertical de uma pilha.

    Traz **três estimadores independentes**, escolhidos para não derivarem
    um do outro (é a concordância entre eles que vira confiança):

    ``n_lattice``
        Quantas camadas estão alinhadas ao pente detectado. É uma medida de
        **amplitude**: conta os picos de borda que caem na fase do pente.
        Uma cadeira real produz 2 ou 3 bordas horizontais (encosto, assento,
        pernas), por isso não serve contar todos os picos - conta-se apenas
        os que estão na fase do padrão, um por cadeira.

    ``n_extent``
        ``extensão_ativa / passo``. É uma medida de **geometria**: usa a
        faixa vertical onde há estrutura (via envelope de energia, sem
        depender dos picos) dividida pelo passo.

    ``n_peaks_raw``
        Total de picos brutos. Serve só de diagnóstico (mostrado na página
        de calibração) e nunca entra na contagem.
    """

    valid: bool = False
    # Os 4 números de contagem candidatos, explicados no docstring da classe.
    n_lattice: int = 0
    n_extent: int = 0
    n_periodic: int = 0
    pitch_support: int = 0
    periodic: dict[str, Any] = field(default_factory=dict)
    n_peaks_raw: int = 0
    # Geometria do padrão, em px do ROI.
    pitch: float = 0.0
    phase: float = 0.0
    extent_top: float = 0.0
    extent_bottom: float = 0.0
    extent: float = 0.0
    lattice_positions: list[float] = field(default_factory=list)
    # Métricas de qualidade, todas 0..1 (exceto gap_cv, que é um CV cru).
    periodicity_quality: float = 0.0
    peak_uniformity: float = 0.0
    # gap_cv = desvio-padrão dos espaçamentos / média. Mede o quanto os
    # espaçamentos são irregulares; 0 = pilha perfeita, 0.5 = pilha torta.
    gap_cv: float = 0.0
    ambiguous: bool = False
    ambiguity: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Dump para JSON (API/dashboard), com arredondamento de floats.

        Arredonda só na serialização: internamente vale a precisão cheia.
        """
        return {
            "valid": self.valid,
            "n_lattice": self.n_lattice,
            "n_extent": self.n_extent,
            "n_periodic": self.n_periodic,
            "pitch_support": self.pitch_support,
            "periodic": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.periodic.items()},
            "n_peaks_raw": self.n_peaks_raw,
            "pitch": round(self.pitch, 1),
            "phase": round(self.phase, 1),
            "extent": round(self.extent, 1),
            "extent_top": round(self.extent_top, 1),
            "extent_bottom": round(self.extent_bottom, 1),
            "n_periodic_layers": len(self.lattice_positions),
            "periodicity_quality": round(self.periodicity_quality, 3),
            "peak_uniformity": round(self.peak_uniformity, 3),
            "gap_cv": round(self.gap_cv, 3),
            "ambiguous": self.ambiguous,
            "ambiguity": self.ambiguity,
            "reasons": self.reasons,
        }


def otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    """Limiar de Otsu: separa o sinal em duas classes (pilha x fundo).

    Em uma pilha de cadeiras o perfil de bordas é claramente bimodal: as
    linhas da pilha têm energia alta e o fundo (parede, chão, espaço vazio)
    tem energia baixa. Otsu encontra a divisão ótima sem depender de
    constantes mágicas - importante porque a iluminação muda ao longo do dia.
    """
    v = values.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return lo
    # 128 bins num sinal de algumas centenas de amostras: resolução grossa
    # o bastante para não casar com o ruído, fina o bastante para separar
    # as duas modas da pilha.
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return lo
    p = hist / total
    centers = (edges[:-1] + edges[1:]) / 2.0
    # omega = fração acumulada; mu = média acumulada (soma parcial).
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    # sigma_b = (μ_total·ω - μ)² / (ω(1-ω))  -> variância ENTRE as classes.
    # Maximizar sigma_b é o critério do Otsu: o limiar que deixa as duas
    # classes o mais separadas possível.
    denom = omega * (1.0 - omega)
    # denom -> 0 nas pontas do histograma (uma classe vazia): ali sigma_b
    # é indefinido, e forçar 0 impede inf/nan de contaminar o argmax.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def active_extent(profile: np.ndarray, min_fill: float = 0.10) -> tuple[float, float]:
    """Faixa vertical onde há estrutura, sem depender dos picos.

    Suaviza o perfil, separa pilha de fundo com Otsu e devolve o primeiro e
    o último índice acima do limiar. É independente da contagem de picos, por
    isso serve como estimador geométrico separado.

    A janela de suavização é propositalmente curta (``H/20``): uma janela
    grande espalha a energia das linhas das extremidades da pilha para o
    fundo e encurta a extensão medida em meia cadeira, exigindo uma correção
    artificial. Com a janela curta o viés medido fica abaixo de 2 px.

    ``min_fill`` evita aceitar um limiar tão alto que sobre quase nada: se a
    parte "forte" ocupar menos de 10% da ROI, usa o percentil 80 do perfil.
    """
    n = profile.size
    if n < 5:
        return 0.0, float(n)
    window = max(5, (n // 20) | 1)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.convolve(profile, kernel, mode="same").astype(np.float64)

    thr = otsu_threshold(smooth)
    strong = smooth > thr
    # min_fill=0.10: se o limiar ficou tão alto que "forte" cobre quase nada,
    # ele não representa a pilha. Aí desce para o percentil 80 - um percentil
    # alto o bastante para ignorar fundo, mas que sempre cobre 20% da ROI.
    if strong.sum() < n * min_fill:
        thr = float(np.percentile(smooth, 80))
        strong = smooth > thr
    idx = np.where(strong)[0]
    if idx.size == 0:
        return 0.0, float(n)
    # Devolve ÍNDICES (não valores de energia) e exclusive no fim: quem chama
    # usa como faixa [top, bottom) para dividir por um comprimento.
    return float(idx[0]), float(idx[-1] + 1)


def count_lattice_peaks(
    peaks: np.ndarray, phase: float, pitch: float, region: tuple[float, float], tolerance: float = 0.45
) -> list[int]:
    """Conta as camadas visíveis alinhadas ao pente ``phase + k*pitch``.

    Uma cadeira real produz **várias** bordas horizontais (topo do encosto,
    frente do assento, pernas), separadas por ~1/3 ou ~1/4 do passo. Somar
    todos os picos contaria 3-4 vezes mais que o número de cadeiras.

    A solução é contar **slots**, não picos: para cada posição do pente, usa
    o pico mais forte dentro da janela de tolerância. Assim há no máximo uma
    camada por cadeira, e a contagem responde "quantas cadeiras têm a borda
    dominante visível" - que é a informação útil de amplitude.

    Só para diagnóstico: a diferença entre ``n_lattice`` e o valor
    geométrico ``extent / pitch`` indica pilha cortada pela imagem, ou uma
    harmônica errada do passo.
    """
    if peaks.size == 0 or pitch <= 0:
        return []
    top, bottom = region
    # tolerance=0.45 -> janela de ±45% do passo. Precisa ser larga (o pico
    # real não cai nunca exatamente na fase) e menor que 0.5 para não invadir
    # o slot vizinho. O piso de 1,0 px cobre passos muito pequenos.
    tol = max(1.0, pitch * tolerance)
    peaks_sorted = peaks[np.argsort(peaks)]
    positions: list[int] = []
    # k0/k1 = índices dos slots que podem tocar a região, com 1 slot de
    # folga dos dois lados para não perder camada na borda.
    k0 = int(math.floor((top - phase) / pitch)) - 1
    k1 = int(math.ceil((bottom - phase) / pitch)) + 1
    for k in range(k0, k1 + 1):
        center = phase + k * pitch
        if center < top - tol or center > bottom + tol:
            continue
        lo = int(center - tol)
        hi = int(center + tol)
        window = peaks_sorted[(peaks_sorted >= lo) & (peaks_sorted <= hi)]
        if window.size:
            # Slot ocupado: usa o pico mais próximo do centro do pente.
            positions.append(int(window[np.argmin(np.abs(window - center))]))
    return positions


def analyze_layers(
    roi_bgr: np.ndarray, chair_height_px: float = 0.0, detect_height_px: float = 0.0
) -> LayerAnalysis:
    """Analisa o padrão vertical da pilha e estima as camadas.

    ``chair_height_px`` (calibração) e ``detect_height_px`` (altura média das
    cadeiras detectadas pelo YOLO) servem para delimitar a faixa de busca do
    passo. Se nenhum dos dois existir, usa uma faixa genérica.
    """
    result = LayerAnalysis()
    h, w = roi_bgr.shape[:2]
    if h < MIN_ROI_H or w < MIN_ROI_W:
        result.reasons.append(f"ROI pequena demais ({w}x{h})")
        return result

    # Pipeline do sinal: CLAHE -> perfil de bordas -> sem tendência.
    # detrend() é aplicado ao PERFIL, não à imagem: é mais barato e o que
    # queremos nivelar é o sinal, não a foto.
    gray = preprocess_roi(roi_bgr)
    profile = row_edge_profile(gray)
    sig = detrend(profile)
    if float(np.max(np.abs(sig))) < 1e-4:
        # Depois do detrend o sinal tem média ~0, então 1e-4 quer dizer
        # "praticamente plano": ROI lisa, parede vazia ou recorte errado.
        result.reasons.append("Sem textura vertical suficiente na ROI")
        return result

    # --- faixa de busca do passo -------------------------------------------
    # Precedência: calibração do operador > altura medida no frame > genérica.
    # A medida do frame é menos confiável (pode ser mistura de classes) e por
    # isso só entra como pista, nunca como calibração persistente.
    height_hint = chair_height_px if chair_height_px > 0 else detect_height_px
    if height_hint > 0:
        # Medido em pilhas reais: uma cadeira plástica *encaixada* acrescenta
        # só ~3% da própria altura à pilha (a cadeira de baixo recebe quase
        # toda a de cima). Cadeiras empilhadas frouxamente chegam a 60-70%.
        #
        # A faixa antiga (62%-105% da altura) assumia que as cadeiras quase não
        # se sobrepõem e **excluía** o passo verdadeiro em qualquer pilha real
        # de plástico — foi exatamente o que aconteceu nos vídeos do galpão.
        lo = max(PERIOD_MIN_PX, height_hint * NEST_MIN_RATIO)
        # Teto de 80% da altura da ROI: um passo maior que isso exigiria mais
        # de uma cadeira dentro da pilha, o que é fisicamente impossível.
        hi = min(h * 0.80, height_hint * NEST_MAX_RATIO)
        if hi <= lo * 1.5:
            # Faixa degenerou (altura de cadeira minúscula ou inconsistente com
            # a ROI). Cai numa faixa genérica ancorada na própria altura da
            # pilha: de 2% a 80% dela.
            lo = max(PERIOD_MIN_PX, height_hint * 0.02)
            hi = max(lo * 3.0, min(h * 0.80, height_hint * 0.80))
        result.reasons.append(
            f"faixa de passo pela altura da cadeira ({height_hint:.0f}px): "
            f"{lo:.0f}-{hi:.0f}px"
        )
    else:
        # Sem calibração: só sabemos que a pilha tem h pixels. 60% do teto
        # porque o passo de uma cadeira encaixada é minúsculo em relação à
        # pilha inteira - a faixa generosa é o preço da ausência de medição.
        lo, hi = PERIOD_MIN_PX, max(PERIOD_MIN_PX + 1.0, h * 0.6)
        result.reasons.append(
            "altura da cadeira desconhecida: faixa de passo genérica "
            "(calibre CHAIR_HEIGHT_PX para uma contagem confiável)"
        )

    period, quality, phase = estimate_period(sig, lo, hi)
    result.periodicity_quality = quality
    if period <= 0 or quality < 0.08:
        # 0.08 = piso propositalmente baixo. Abaixo disso não há periodicidade
        # nenhuma; entre 0.08 e ~0.3 a análise segue, mas a confiança final
        # já reflete a qualidade baixa (ver _final_confidence).
        result.reasons.append(f"Sem período confiável (qualidade {quality:.2f})")
        return result

    # --- ambiguidade de harmônica -----------------------------------------
    # Se metade ou o dobro do passo explicam o sinal tão bem quanto o passo,
    # não é possível saber quantas cadeiras são. Dizemos isso em vez de
    # inventar um número.
    amb = period_ambiguity(sig, period, lo, hi)
    if amb.get("ambiguous"):
        result.ambiguous = True
        result.ambiguity = amb
        period = float(amb["chosen"])
        result.reasons.append(
            f"PERÍODO AMBÍGUO: {[o['period'] for o in amb['options']]} px explicam a pilha "
            f"igualmente bem; usando {period:.0f}px. Contagem marcada como indeterminada."
        )

    result.pitch = period
    result.phase = phase

    # --- extensão ativa (geometria, independente dos picos) ----------------
    # Usa o perfil CRU (não o detrended): aqui queremos "onde há pixels", e a
    # tendência lenta é informação de que a pilha existe, não ruído.
    top, bottom = active_extent(profile)
    result.extent_top, result.extent_bottom = top, bottom
    result.extent = max(0.0, bottom - top)
    if result.extent > period * 0.8:
        # Só conta se a extensão passa de ~1 passo. Abaixo disso é ruído de
        # borda (uma cadeira só encostando na ROI) e arredondaria para 1 ou 2.
        result.n_extent = max(1, int(round(result.extent / period)))

    # --- passo pelos próprios picos (fonte mais confiável) -----------------
    # Os picos são detectados com distância mínima pequena para não perder
    # camadas; o passo vem de quantos espaçamentos entre eles se repetem.
    # 0.15 x passo: bem menor que o passo, para não "colar" dois picos da
    # mesma cadeira; mas grande o bastante para não contar granulado.
    loose_dist = max(2, int(round(period * 0.15)))
    idx, prom, _ = find_peaks(sig, min_prominence=PEAK_MIN_PROMINENCE, min_distance=loose_dist)
    result.n_peaks_raw = int(idx.size)

    # Restringe os candidatos de passo à faixa fisicamente possível
    # (mesma fórmula da faixa de busca acima, reaplicada por segurança: se o
    # passo foi trocado pela desambiguação, os limites também mudaram).
    p_lo, p_hi = max(PERIOD_MIN_PX, height_hint * NEST_MIN_RATIO) if height_hint > 0 else PERIOD_MIN_PX, \
                 min(h * 0.80, height_hint * NEST_MAX_RATIO) if height_hint > 0 else h * 0.6
    p_peaks, q_peaks, support = estimate_period_from_peaks(
        [float(i) for i in idx], min_period=p_lo, max_period=p_hi
    )
    result.pitch_support = support
    if p_peaks > 0 and q_peaks >= 0.15:
        # O passo medido pelos picos vale mais que o do pente: ele não é
        # atraído por múltiplos do período.
        if abs(p_peaks - period) > period * 0.25:
            # 0.25 = acima disso a discordância já não é arredondamento,
            # é uma das duas leituras errada. Fica registrado nas notas.
            result.reasons.append(
                f"passo corrigido: pente sugeriu {period:.0f}px, mas os espaçamentos "
                f"entre {support} pares de picos indicam {p_peaks:.0f}px"
            )
        period = p_peaks
        result.pitch = period
        # A qualidade final é o MELHOR dos dois métodos, não a média: eles
        # olham o mesmo sinal por ângulos diferentes e um bom em um ângulo
        # já é evidência real de periodicidade.
        result.periodicity_quality = max(result.periodicity_quality, q_peaks)

    # --- extensão PERIÓDICA: o trecho em que o padrão realmente se repete --
    # Este é o estimador que decide a contagem em pilhas reais. Usar a altura
    # total superestima, porque a parte das pernas no fundo tem estrutura
    # diferente e não se repete com o mesmo passo das cadeiras encaixadas.
    pe = periodic_extent([float(i) for i in idx], period)
    result.periodic = pe
    result.n_periodic = int(pe.get("count", 0))
    if result.n_periodic:
        # Só os picos DENTRO do trecho periódico contam como camadas. Os de
        # fora (topo da pilha, fundo nas pernas) são estrutura, não cadeira.
        # A folga de 1,0 px absorve o arredondamento de índice do topo/base.
        lattice_pos = [float(p) for p in idx if pe["top"] - 1.0 <= p <= pe["bottom"] + 1.0]
    else:
        lattice_pos = []
    result.lattice_positions = lattice_pos
    result.n_lattice = len(lattice_pos)

    if len(lattice_pos) >= 3:
        # Só com 3+ pontos há espaçamento para medir variabilidade. Com menos
        # isso, qualquer "coeficiente de variação" é ruído amplificado.
        diffs = np.diff(np.array(lattice_pos, dtype=float))
        med = float(np.median(diffs))
        if med > 0:
            result.gap_cv = float(np.std(diffs) / med)
            # peak_uniformity = 1 - gap_cv, saturado em 0. Serve como
            # "as camadas estão igualmente espaçadas?" em 0..1.
            result.peak_uniformity = float(np.clip(1.0 - result.gap_cv, 0.0, 1.0))
    elif len(lattice_pos) == 2:
        # Com 2 pontos existe espaçamento, mas não existe variabilidade.
        # 0.4 é meio-termo arbitrário: "algo sabemos, pouco".
        result.peak_uniformity = 0.4

    # Válido = ao menos um estimador de contagem produziu número > 0.
    result.valid = result.n_periodic > 0 or result.n_extent > 0
    return result


# =========================================================================== #
# 4. Contador
# =========================================================================== #
class StackCounter:
    """Estima quantas cadeiras há em uma pilha.

    Uma instância por sessão/pipeline: guarda pesos, offset e calibração.
    Não guarda estado por pilha (isso é do :mod:`stability`).

    Uso::

        counter = StackCounter()
        estimate = counter.count(roi_bgr, chair_detections=[...], previous_count=20)
        estimate.count        # -> 23
        estimate.confidence   # -> 0.96
    """

    def __init__(
        self,
        *,
        method: str | None = None,
        chair_height_px: float | None = None,
        chair_weight: float | None = None,
        min_confidence: float | None = None,
        max_chairs: int | None = None,
        count_offset: int = 0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.method = method or settings.counting.counting_method
        self.chair_height_px = (
            settings.counting.chair_height_px if chair_height_px is None else chair_height_px
        )
        self.chair_weight = settings.ai.chair_weight if chair_weight is None else chair_weight
        self.min_confidence = (
            settings.counting.min_confidence if min_confidence is None else min_confidence
        )
        self.max_chairs = settings.counting.max_chairs if max_chairs is None else max_chairs
        self.count_offset = count_offset
        # Pesos de fusão. Nenhum estimador é "a verdade": eles se complementam
        # e a concordância entre eles é o que produz a confiança.
        #   extent     - geometria: extensão ativa / passo (sem ambiguidade)
        #   detections - exata quando o YOLO enxerga todas as cadeiras
        #   size       - geométrico, exige altura de cadeira calibrada
        # A contagem de camadas (lattice) não entra como número: ela valida o
        # passo (ver `_final_confidence(pattern_ok=...)`).
        # Peso 1.0 em periodic e detections: os dois estimadores em que
        # confiamos de verdade. extent e size ficam abaixo porque ambos
        # derivam do mesmo passo - concordar com eles é quase redundante.
        self.weights: dict[str, float] = weights or {
            "extent": 0.35,
            "periodic": 1.00,
            "detections": 1.00,
            "size": 0.45,
        }

    # ------------------------------------------------------------------ API
    def count(
        self,
        roi_bgr: np.ndarray | None,
        *,
        chair_detections: Sequence[Detection] | None = None,
        pile: PileCandidate | None = None,
        previous_count: int | None = None,
        chair_height_px: float | None = None,
    ) -> CountEstimate:
        """Conta as cadeiras de uma ROI de pilha.

        ``previous_count`` é usado apenas como referência de consistência: uma
        grande variação em relação ao frame anterior reduz um pouco a
        confiança (sem nunca "corrigir" a leitura atual - quem decide é a
        estabilidade temporal).
        """
        notes: list[str] = []
        candidates: dict[str, float] = {}
        cweights: dict[str, float] = {}
        confidences: dict[str, float] = {}

        # Os 3 dicionários acima andam juntos de propósito:
        #   candidates[k]  = contagem proposta pelo estimador k
        #   cweights[k]    = peso do estimador k na fusão
        #   confidences[k] = confiança 0..1 do estimador k
        # A confiança NÃO entra como peso: ela é medida à parte, porque um
        # estimador com confiança 0.2 ainda vale como voto (é discordância
        # que informa, não peso baixo).

        chair_height = self.chair_height_px if chair_height_px is None else chair_height_px
        dets = list(chair_detections or [])
        method = self.method
        # Atalho: no método "detections" a imagem é inútil e analyze_layers
        # custaria a FFT + as varreduras de proeminência à toa.
        wants_image = method != "detections"

        # --- altura da cadeira, medida a partir das detecções ---------------
        detect_h = self._chair_height_from_detections(dets)
        if detect_h > 0:
            notes.append(f"altura da cadeira medida nas detecções: {detect_h:.0f}px")
        if chair_height <= 0 and detect_h > 0:
            # Usa a medida do frame como pista de período, mas NÃO como
            # calibração persistente (evita realimentar erro).
            effective_height = detect_h
        else:
            effective_height = chair_height

        # --- análise de padrão vertical da ROI -------------------------------
        analysis = LayerAnalysis()
        if roi_bgr is not None and wants_image:
            # effective_height primeiro (usa a calibração se existir) e
            # detect_h segundo (pista secundária) - analyze_layers escolhe.
            analysis = analyze_layers(roi_bgr, effective_height, detect_h)
        for reason in analysis.reasons:
            # As reasons são escritas para o log/dashboard em português: são
            # o único jeito de um operador entender POR QUE o número saiu.
            notes.append(reason)

        # --- estimador principal: extensão / passo (geometria) -------------
        # Este é o estimador de referência porque não tem ambiguidade de
        # convenção: o passo é a distância de repetição cadeira-a-cadeira, e
        # a extensão é a altura ocupada pela pilha. Logo ``extensão/passo`` é
        # diretamente o número de cadeiras.
        if method in ("auto", "fused", "periodicity") and analysis.n_periodic > 0:
            candidates["periodic"] = float(analysis.n_periodic)
            cweights["periodic"] = self.weights.get("periodic", 1.0)
            # Confiança do estimador periódico: 0.35 de base (ele é
            # estrutural, não observacional) + 0.35 pela regularidade dos
            # espaçamentos + 0.25 pela qualidade do passo. Nunca chega a 1.0
            # porque sempre pode haver uma camada escondida.
            confidences["periodic"] = float(
                np.clip(0.35 + 0.35 * analysis.peak_uniformity + 0.25 * analysis.periodicity_quality, 0.0, 1.0)
            )

        if method in ("auto", "fused", "periodicity") and analysis.n_extent > 0:
            candidates["extent"] = float(analysis.n_extent)
            cweights["extent"] = self.weights["extent"]
            # Base menor (0.25) que a do periódico: a extensão ativa é uma
            # medição mais "gorda" (pega a sombra e a borda da pilha), logo
            # carrega menos informação de que o padrão se repete.
            confidences["extent"] = float(
                np.clip(0.25 + 0.45 * analysis.periodicity_quality, 0.0, 1.0)
            )

        # --- validação: camadas visíveis alinhadas ao pente ----------------
        # NÃO entra como contagem concorrente. O pente marca os *limites*
        # entre cadeiras, então uma pilha de N cadeiras exibe N+1 linhas: usar
        # isso como segunda contagem introduziria um erro de ±1 garantido.
        # Serve para confirmar que o passo encontrado é real (cobertura) e
        # para penalizar quando há discordão forte.
        if analysis.n_lattice > 0 and analysis.n_extent > 0:
            coverage = analysis.n_lattice / max(1.0, float(analysis.n_extent))
            # Faixa 0.80-1.30: as duas contagens devem concordar de perto.
            # Abaixo de 0.80 o padrão vertical não "cobre" a geometria
            # (poucas bordas visíveis = pilha suja ou passo errado); acima de
            # 1.30 há mais bordas que cadeiras, ou seja, estamos contando as
            # sub-bordas (encosto/assento) como camadas.
            lattice_ok = 0.80 <= coverage <= 1.30
            if not lattice_ok:
                notes.append(
                    f"cobertura de camadas ({analysis.n_lattice} para {analysis.n_extent} "
                    f"estimadas) fora do esperado; confiança reduzida"
                )
        else:
            # Sem um dos dois lados não dá para checar cobertura. Falso (não
            # verdadeiro) para não dar "cobertura perfeita" sem evidência.
            lattice_ok = False

        # --- detecção direta (YOLO) ------------------------------------------
        if method in ("auto", "fused", "detections") and dets:
            n_det = self._count_from_detections(dets)
            if n_det > 0:
                candidates["detections"] = float(n_det)
                cweights["detections"] = self.weights["detections"]
                conf = self._detection_confidence(dets, detect_h)
                confidences["detections"] = conf
                if conf < 0.35:
                    # Abaixo disso o YOLO mal "vê" a pilha; ainda entra na
                    # fusão, mas com confiança declarada baixa.
                    notes.append("confiança das detecções baixa (poucas cadeiras visíveis)")

        # --- estimador: altura calibrada da cadeira --------------------------
        # effective_height > 0 é a condição de "existe calibração OU medida no
        # frame". Com chair_height_px=0 e sem detecções, este estimador
        # simplesmente não existe - e é exatamente o que o operador precisa
        # ler para entender que calibrar resolve.
        if method in ("auto", "fused", "size") and effective_height > 0 and roi_bgr is not None:
            n_size = self._count_from_size(roi_bgr, effective_height, analysis)
            if n_size > 0:
                candidates["size"] = float(n_size)
                cweights["size"] = self.weights["size"]
                # Quando size é o método FORÇADO, ele é o único que responde,
                # então a confiança sobe (0.35) mesmo sendo um método frágil.
                confidences["size"] = 0.35 if method == "size" else 0.25

        if not candidates:
            # Nenhum estimador respondeu: ROI vazia, sem padrão e sem YOLO.
            # count=0 vira status UNKNOWN no dashboard, que é a resposta
            # honesta ("não sei"), não "zero cadeiras".
            return CountEstimate(
                count=0,
                confidence=0.0,
                detection_confidence=0.0,
                method=self.method,
                candidates={},
                candidate_weights={},
                chair_height_px=effective_height,
                pitch_px=analysis.pitch,
                notes=notes + ["nenhum estimador produziu uma contagem utilizável"],
            )

        # --- período ambíguo e sem modelo: não chame número de cadeiras ----
        # Se nem o YOLO (que distingue cada cadeira) nem a calibração estão
        # disponíveis, e o padrão vertical é ambíguo, qualquer número seria
        # chute. O certo é devolver 0 -> status UNKNOWN -> o dashboard mostra
        # "?" e o operador sabe que precisa de modelo ou calibração.
        if analysis.ambiguous and not dets and chair_height <= 0:
            # Guardamos os candidatos mas com peso 0: o dashboard mostra o
            # que teria sido estimado, marcado como não usado.
            return CountEstimate(
                count=0,
                confidence=0.0,
                detection_confidence=0.0,
                method=self.method,
                candidates=candidates,
                candidate_weights={k: 0.0 for k in candidates},
                chair_height_px=effective_height,
                pitch_px=analysis.pitch,
                notes=notes
                + [
                    "CONTAGEM INDETERMINADA: o padrão vertical é ambíguo e não há "
                    "modelo treinado nem altura de cadeira calibrada. Treine o modelo "
                    "ou calibre CHAIR_HEIGHT_PX para obter um número confiável."
                ],
            )

        # --- fusão ponderada ------------------------------------------------
        fused, agreement, det_conf = self._fuse(candidates, cweights, confidences)
        # O offset entra DEPOIS da fusão (e não como estimador): ele é uma
        # correção de viés sistemático, não mais uma opinião sobre a pilha.
        fused += self.count_offset
        # Arredonda e satura em [0, max_chairs]. max_chairs é um teto de
        # segurança: uma pilha de 2000 cadeiras é erro, não realidade.
        fused = int(max(0, min(self.max_chairs, round(fused))))

        # --- confiança final -------------------------------------------------
        confidence = self._final_confidence(
            value=fused,
            agreement=agreement,
            det_conf=det_conf,
            candidates=candidates,
            confidences=confidences,
            previous_count=previous_count,
            pattern_ok=lattice_ok,
            ambiguous=analysis.ambiguous,
        )
        if self.count_offset:
            notes.append(f"offset de calibração aplicado: {self.count_offset:+d}")

        return CountEstimate(
            count=fused,
            confidence=confidence,
            detection_confidence=det_conf,
            method=self.method,
            candidates=candidates,
            # Normaliza os pesos para somar 1 (o _fuse usa os valores crus,
            # mas o dashboard quer ver "45% para periodic, 35% para detections").
            candidate_weights={k: round(v / max(1e-6, sum(cweights.values())), 3) for k, v in cweights.items()},
            chair_height_px=effective_height,
            pitch_px=analysis.pitch,
            notes=notes,
        )

    # ------------------------------------------------------------- estimadores
    def _count_from_detections(self, dets: Sequence[Detection]) -> int:
        """Soma ponderada das detecções de cadeira.

        ``chair_weight`` (default 1.0) diz quantas cadeiras cada detecção
        representa: 1.0 para modelo de cadeira individual, >1 se o modelo
        foi treinado para marcar a pilha inteira. O piso em 1.0 garante que
        um peso mal configurado (0.5) não "desconte" cadeiras.
        """
        total = 0.0
        for d in dets:
            total += max(1.0, self.chair_weight)
        return int(round(total))

    def _chair_height_from_detections(self, dets: Sequence[Detection]) -> float:
        """Mediana das alturas das cadeiras detectadas (0 se não houver).

        Mediana e não média: uma detecção errada (a pilha inteira, uma caixa)
        não deve arrastar a medida. Exige >= 2 detecções e descarta as
        minúsculas (<4px), que são rótulos ruins do YOLO.
        """
        heights = [d.height for d in dets if d.height > 4 and d.width > 4]
        if len(heights) < 2:
            return 0.0
        return float(np.median(heights))

    def _detection_confidence(self, dets: Sequence[Detection], detect_h: float) -> float:
        """Confiança do estimador por detecção.

        Média das confidências do YOLO, com dois ajustes:

        * alturas muito discrepantes entre as detecções indicam mistura de
          classes ou detecção ruim;
        * detecções muito altas comparadas com a mediana indicam que o
          modelo está marcando partes grandes (p.ex.: a pilha inteira) como
          cadeira, o que superestima a contagem.
        """
        if not dets:
            return 0.0
        mean_conf = float(np.mean([d.conf for d in dets]))
        heights = np.array([d.height for d in dets], dtype=float)
        heights = heights[heights > 4]
        consistency = 1.0
        if heights.size >= 3:
            med = float(np.median(heights))
            if med > 0:
                # CV das alturas. 0.7 = quanto de dispersão custa toda a
                # confiança de consistência: CV 1.0 (alturas 2x diferentes)
                # zera a consistência; CV 0.3 ainda mantém ~0.8.
                cv_ = float(np.std(heights) / med)
                consistency = float(np.clip(1.0 - cv_ * 0.7, 0.25, 1.0))
        # Piso 0.65: mesmo com detecções inconsistentes, o YOLO ainda viu
        # alguma coisa. O 0.35 no meio é o peso da consistência.
        # detect_h não é usado no cálculo - fica na assinatura porque a
        # comparação altura-detecção vs altura-medida da ROI é o lugar
        # natural para penalizar o modelo que marca a pilha inteira.
        return float(np.clip(mean_conf * (0.65 + 0.35 * consistency), 0.0, 1.0))

    def _count_from_size(
        self, roi_bgr: np.ndarray, chair_height: float, analysis: LayerAnalysis
    ) -> int:
        """``extensão ativa / altura calibrada da cadeira``.

        Só existe com calibração: sem ``chair_height_px`` não há divisor
        confiável. Nota que divide a EXTENSÃO (não a altura da ROI): a
        diferença entre as duas é justamente a parte vazia em volta da
        pilha, que aqui não entra.
        """
        if chair_height <= 0:
            return 0
        if analysis.extent > 0:
            extent = analysis.extent
        else:
            # Fallback: sem extensão ativa, usa a altura inteira da ROI. Pior,
            # mas melhor que devolver 0 (que zeraria o estimador).
            extent = float(roi_bgr.shape[0])
        if extent <= 0:
            return 0
        # max(1, ...): mesmo que o arredondamento dê 0 (extensão < metade da
        # cadeira), há pelo menos 1 cadeira ali.
        return int(max(1, round(extent / chair_height)))

    # ---------------------------------------------------------------- fusão
    def _fuse(
        self,
        candidates: dict[str, float],
        cweights: dict[str, float],
        confidences: dict[str, float],
    ) -> tuple[float, float, float]:
        """Combina os estimadores.

        Retorna ``(valor, concordância 0..1, confiança de detecção)``.

        A concordância mede o quanto os estimadores que sobraram estão perto
        uns dos outros: dois estimadores que dizem 20 e 21 concordam mais que
        dois que dizem 20 e 30.
        """
        total_w = sum(cweights.values())
        if total_w <= 0:
            return 0.0, 0.0, 0.0
        # Média ponderada simples: com 2 estimadores é o ponto médio, com 4
        # é o "centro de gravidade" das opiniões.
        value = sum(candidates[k] * cweights[k] for k in candidates) / total_w

        values = np.array(list(candidates.values()), dtype=float)
        if values.size == 1:
            # Só um estimador: não há com quem discordar, mas também não há
            # com quem concordar. 0.6 base + 0.4 da confiança própria: melhor
            # que 0 (o estimador fez alguma coisa) e pior que 1 (ninguém
            # checou). É por isso que `coverage` pune tanto na confiança final.
            agreement = 0.6 + 0.4 * confidences.get(next(iter(candidates)), 0.0)
        else:
            # Agreement = 1 - spread/escala. Espalhamento (desvio-padrão, não
            # amplitude) porque um estimador completamente fora não deve
            # dominar: 20/21/30 tem spread pequeno perto de 20/20/20.
            spread = float(np.std(values))
            # Escala = 25% da contagem (mínimo 2 px de contagem). Um erro de
            # 25% numa pilha de 20 = 5 cadeiras, que é o limite do aceitável
            # antes de dizer "os estimadores discordam".
            scale = max(2.0, 0.25 * max(1.0, value))  # tolerância ~25% da contagem
            agreement = float(np.clip(1.0 - spread / scale, 0.0, 1.0))
            # Mesma métrica com escala metade (mais rígida: 12% da contagem,
            # piso 3). A média das duas evita que Agreement seja
            # hiper-generoso (escala 25%) ou hiper-rígido (escala 12%) em
            # faixas de contagem diferentes.
            agreement = 0.5 * agreement + 0.5 * float(
                np.clip(1.0 - spread / max(3.0, 0.5 * max(1.0, value)), 0.0, 1.0)
            )
        det_conf = confidences.get("detections", 0.0)
        return value, float(np.clip(agreement, 0.0, 1.0)), det_conf

    def _final_confidence(
        self,
        *,
        value: int,
        agreement: float,
        det_conf: float,
        candidates: dict[str, float],
        confidences: dict[str, float],
        previous_count: int | None,
        pattern_ok: bool = True,
        ambiguous: bool = False,
    ) -> float:
        """Combina concordância, confiança dos estimadores e continuidade."""
        if not candidates:
            return 0.0
        mean_est_conf = float(np.mean(list(confidences.values())))
        # Um estimador só não é penalizado tanto: ele já carrega sua confiança.
        # coverage = min(1, n_estimadores/2): 2 estimadores já dá 1.0. A ideia
        # é "concordância precisa de pelo menos duas opiniões", mas mais que isso
        # não acrescenta nada.
        coverage = min(1.0, len(candidates) / 2.0)
        # Soma dos pesos = 1.0. Agreement domina (0.45) porque é o sinal mais
        # forte de que o número é certo; a média das confianças (0.40) mede
        # a qualidade; coverage (0.15) é só um empurrão de quem tem mais dados.
        conf = 0.45 * agreement + 0.40 * mean_est_conf + 0.15 * coverage

        # O padrão vertical foi confirmado pelas camadas visíveis? Se não, o
        # passo pode ser uma harmônica errada - a contagem geométrica continua
        # sendo a melhor aposta, mas a confiança cai.
        if not pattern_ok and "extent" in candidates:
            conf *= 0.72

        # Período ambíguo: o sinal não distingue a cadeira da sua estrutura
        # interna. A contagem pode estar errada por um fator de 2, então a
        # confiança despenca - o dashboard vai mostrar "indeterminado" e o
        # operador precisa de um modelo treinado (ou calibrar a altura).
        if ambiguous and "detections" not in candidates:
            # Só pune se não há YOLO. Com detecção existe uma segunda fonte
            # independente, então o problema da harmônica fica muito menor.
            conf *= 0.35

        # Continuidade: mudança grande em relação ao frame anterior reduz a
        # confiança deste frame (o filtro de estabilidade resolve a troca).
        if previous_count is not None and previous_count > 0 and value > 0:
            delta = abs(value - previous_count) / max(1.0, previous_count)
            if delta > 0.35:
                # 0.35 = 35% de variação é o "normal" de uma pilha real
                # (cadeira adicionada/removida). Acima disso, penalidade
                # linear de 0.8 por unidade de delta excedente, com piso
                # 0.45: uma troca real de pilha (metade a menos) não zera
                # a confiança, só diz "isso mudou, olha com cuidado".
                conf *= float(np.clip(1.0 - (delta - 0.35) * 0.8, 0.45, 1.0))

        # Teto: quando existe detecção do YOLO, a confiança não pode passar
        # muito acima da qualidade dessa detecção.
        if det_conf > 0:
            # Se o YOLO viu bem (det_conf=1), teto = 1.0. Se viu mal
            # (det_conf=0.2), teto = 0.48: por mais que a geometria
            # concordasse, não dá para mostrar 90% com uma detecção fraca.
            conf = min(conf, max(0.0, 0.35 + 0.65 * det_conf))
        return float(np.clip(conf, 0.0, 1.0))

    # ------------------------------------------------------------ diagnóstico
    def diagnose(self, roi_bgr: np.ndarray, chair_height_px: float = 0.0) -> dict[str, Any]:
        """Análise detalhada - usada no dashboard e nos testes.

        Não passa pela fusão: devolve os estimadores separados, que é o que
        se quer ao depurar ("qual estimador errou?").
        """
        analysis = analyze_layers(roi_bgr, chair_height_px or self.chair_height_px)
        return analysis.as_dict()


# =========================================================================== #
# 5. Sugestão de offset a partir de correções manuais
# =========================================================================== #
def suggest_offset(corrections: Iterable[tuple[int, int]], min_samples: int = 3) -> dict[str, Any]:
    """Sugere um ``count_offset`` a partir de pares ``(ai_count, correct_count)``.

    Só devolve um valor quando há evidência estatística suficiente
    (mediana do erro com pelo menos ``min_samples`` amostras). Nunca inventa:
    sem amostras, retorna ``offset=0`` e ``reliable=False``.
    """
    diffs: list[int] = []
    for ai, human in corrections:
        # Erro com sinal de operador: positivo = faltou cadeira na leitura da IA.
        diffs.append(int(human) - int(ai))
    if len(diffs) < min_samples:
        return {
            "offset": 0,
            "reliable": False,
            "samples": len(diffs),
            "median_error": None,
            "message": (
                f"São necessárias {min_samples} correções manuais para sugerir um offset "
                f"(hoje há {len(diffs)})."
            ),
        }
    # Mediana, não média: um operador que errou a conta uma vez (digamos,
    # +8) não deve puxar o offset de todas as outras pilhas.
    median_err = int(round(float(np.median(diffs))))
    # 0.6 = 60% das correções precisam bater com a mediana para o offset
    # ser considerado confiável. Abaixo disso é coincidência, não viés.
    consistent = sum(1 for d in diffs if d == median_err) / len(diffs)
    return {
        "offset": median_err,
        "reliable": bool(consistent >= 0.6),
        "samples": len(diffs),
        "median_error": median_err,
        "consistency": round(consistent, 3),
        "message": (
            f"Erro mediano (humano - IA) = {median_err:+d} em {len(diffs)} correções "
            f"({consistent * 100:.0f}% consistentes)."
        ),
    }


__all__ = [
    "StackCounter",
    "LayerAnalysis",
    "analyze_layers",
    "active_extent",
    "count_lattice_peaks",
    "estimate_period_from_peaks",
    "periodic_extent",
    "estimate_period",
    "find_peaks",
    "row_edge_profile",
    "preprocess_roi",
    "detrend",
    "suggest_offset",
]
