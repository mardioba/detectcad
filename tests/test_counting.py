"""Testes do algoritmo de contagem (a parte mais crítica do sistema).

Estes testes usam o gerador de pilhas sintéticas, onde o número de cadeiras é
CONHECIDO. Isso valida o algoritmo de forma determinística, sem depender de
um modelo treinado.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.counting.stack_counter import (
    StackCounter,
    active_extent,
    analyze_layers,
    count_lattice_peaks,
    estimate_period,
    find_peaks,
    otsu_threshold,
    row_edge_profile,
    suggest_offset,
)
from app.schemas import Detection
from tools.make_synthetic_stack import draw_stack, make_scene


def build_stack(n: int, chair_h: int = 34, base_y: int = 633, pad: float = 0.10):
    """Pilha sintética com ``n`` cadeiras + ROI com folga (como a bbox do YOLO)."""
    img = make_scene([n], width=420, height=720, chair_h=chair_h, n_piles=1)
    x1, y1, x2, y2 = draw_stack(img, 210, base_y, n, chair_h=chair_h, noise=0.01)
    px = int((x2 - x1) * pad) + 4
    py = int((y2 - y1) * pad) + 4
    roi = img[max(0, y1 - py) : y2 + py, max(0, x1 - px) : x2 + px]
    return roi, chair_h


# ------------------------------------------------------------------ imagens
def test_preprocess_and_profile_shapes():
    img = make_scene([10], n_piles=1)
    assert img.ndim == 3 and img.shape[2] == 3
    prof = row_edge_profile(img)
    assert prof.size == img.shape[0]
    assert np.all(prof >= 0)


def test_otsu_separates_bimodal():
    values = np.concatenate([np.zeros(500), np.ones(500) * 10.0])
    thr = otsu_threshold(values)
    # precisa ficar entre os dois modos (0 e 10) para separar corretamente
    assert 0.0 < thr < 5.0, thr


def test_find_peaks_detects_periodic_peaks():
    sig = np.zeros(200, dtype=np.float32)
    for i in range(20, 200, 20):  # evita o índice 0 (não é pico local)
        sig[i] = 1.0
    idx, prom, _ = find_peaks(sig, min_prominence=0.2, min_distance=3)
    assert len(idx) == 9, len(idx)
    assert prom.min() > 0


def test_highest_peak_keeps_prominence():
    """Regressão: o pico mais alto da pilha não pode ter proeminência 0."""
    sig = np.zeros(300, dtype=np.float32)
    for i in range(50, 300, 50):
        sig[i] = 1.0
    sig[150] = 2.0  # pico dominante
    idx, prom, _ = find_peaks(sig, min_prominence=0.2, min_distance=3)
    assert 150 in idx.tolist()
    # altura 2.0 sobre base 0.0 -> proeminência 2.0 (antes dava 0)
    assert prom[np.where(idx == 150)[0][0]] == 2.0


def test_estimate_period_recovers_known_period():
    sig = np.zeros(400, dtype=np.float32)
    for i in range(0, 400, 25):
        sig[i] = 1.0
    period, quality, phase = estimate_period(sig, 10, 60)
    assert 22 <= period <= 28, f"esperado ~25, veio {period}"
    assert quality > 0.2


def test_active_extent_finds_structure_band():
    prof = np.concatenate([np.zeros(100), np.ones(200), np.zeros(100)]).astype(np.float32)
    top, bottom = active_extent(prof)
    assert 95 <= top <= 105
    assert 295 <= bottom <= 305


def test_count_lattice_peaks_one_per_slot():
    peaks = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    got = count_lattice_peaks(peaks, 0.0, 30.0, (0.0, 90.0))
    # slots em 0/30/60/90 -> 1 pico por slot (nunca 2 do mesmo slot)
    assert len(got) == 4
    assert len(set(got)) == len(got)


# --------------------------------------------------------------- contagem
@pytest.mark.parametrize("n", [5, 10, 15, 18])
def test_count_within_one_chair(n):
    roi, chair_h = build_stack(n)
    est = StackCounter(chair_height_px=float(chair_h), method="auto").count(roi)
    assert abs(est.count - n) <= 1, f"real={n} contado={est.count} {est.candidates}"


@pytest.mark.parametrize("chair_h", [18, 26, 34])
def test_count_respects_chair_scale(chair_h):
    roi, _ = build_stack(12, chair_h=chair_h, base_y=700)
    est = StackCounter(chair_height_px=float(chair_h), method="auto").count(roi)
    assert abs(est.count - 12) <= 1, f"chair_h={chair_h} real=12 contado={est.count}"


def test_detections_estimator():
    roi, chair_h = build_stack(10)
    dets = [
        Detection(x1=10, y1=10, x2=40, y2=44, conf=0.9, class_id=0, class_name="chair")
        for _ in range(10)
    ]
    est = StackCounter(chair_height_px=float(chair_h), method="detections").count(roi, chair_detections=dets)
    assert est.count == 10
    assert est.detection_confidence > 0.5


def test_size_estimator_uses_calibration():
    roi, chair_h = build_stack(14)
    est = StackCounter(chair_height_px=float(chair_h), method="size").count(roi)
    assert abs(est.count - 14) <= 1


def test_too_small_roi_returns_zero():
    roi = np.full((5, 5, 3), 128, dtype=np.uint8)
    est = StackCounter().count(roi)
    assert est.count == 0
    assert est.confidence == 0.0


def test_blank_roi_has_no_confident_count():
    roi = np.full((300, 100, 3), 128, dtype=np.uint8)  # sem textura
    est = StackCounter().count(roi)
    assert est.count == 0 or est.confidence < 0.5


def test_offset_is_applied():
    roi, chair_h = build_stack(10)
    base = StackCounter(chair_height_px=float(chair_h)).count(roi)
    with_off = StackCounter(chair_height_px=float(chair_h), count_offset=3).count(roi)
    assert with_off.count == base.count + 3


def test_count_is_clamped_by_max():
    roi, chair_h = build_stack(10)
    est = StackCounter(chair_height_px=float(chair_h), max_chairs=5).count(roi)
    assert est.count <= 5


def test_analysis_is_deterministic():
    roi, chair_h = build_stack(13)
    a1 = analyze_layers(roi, float(chair_h))
    a2 = analyze_layers(roi, float(chair_h))
    assert a1.pitch == a2.pitch
    assert a1.n_extent == a2.n_extent


# ------------------------------------------------------- sugestão de offset
def test_suggest_offset_requires_samples():
    r = suggest_offset([(10, 11), (20, 21)], min_samples=3)
    assert r["offset"] == 0
    assert r["reliable"] is False


def test_suggest_offset_detects_systematic_plus_one():
    r = suggest_offset([(10, 11), (20, 21), (30, 31), (40, 41)], min_samples=3)
    assert r["offset"] == 1
    assert r["reliable"] is True


def test_suggest_offset_ignores_noise():
    # erros +1, -1, +1, -1 -> sem viés, não deve sugerir offset
    r = suggest_offset([(10, 11), (20, 19), (30, 31), (40, 39)], min_samples=3)
    assert r["reliable"] is False
