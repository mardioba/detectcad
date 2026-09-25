"""Testes do filtro de estabilidade temporal e dos estados de contagem."""

from __future__ import annotations

import pytest

from app.counting.stability import StabilityFilter, decide_status
from app.schemas import CountStatus


# ------------------------------------------------------- exemplo do projeto
def test_single_spike_does_not_change_display():
    """20,20,21,20,20,... -> continua 20 (o frame 21 é ignorado)."""
    f = StabilityFilter(window=10, mode="mode", change_min_frames=3)
    seq = [20, 20, 21, 20, 20, 20, 20, 20, 20, 20]
    for v in seq:
        r = f.update(1, v)
    assert r.stable_count == 20, r.as_dict()


def test_sustained_change_is_accepted():
    """20,20,21,21,21,21 -> vira 21 depois de change_min_frames leituras."""
    f = StabilityFilter(window=10, mode="mode", change_min_frames=3)
    for v in [20, 20, 21, 21, 21, 21]:
        r = f.update(1, v)
    assert r.stable_count == 21, r.as_dict()


def test_outlier_burst_shorter_than_min_frames_is_rejected():
    f = StabilityFilter(window=10, change_min_frames=3)
    for v in [30] * 10:
        f.update(1, v)
    # 2 frames de 50 não devem trocar o valor
    f.update(1, 50)
    r = f.update(1, 50)
    assert r.stable_count == 30


def test_median_mode_resists_outliers():
    f = StabilityFilter(window=10, mode="median", change_min_frames=3)
    for v in [15] * 8:
        f.update(2, v)
    for v in [99, 99]:
        f.update(2, v)
    r = f.update(2, 15)
    assert r.stable_count == 15


def test_mean_mode_averages():
    f = StabilityFilter(window=5, mode="mean", change_min_frames=2)
    for v in [10, 10, 12]:
        f.update(3, v)
    r = f.update(3, 12)
    assert r.stable_count == 11  # média 11.0


def test_stability_confidence_drops_with_disagreement():
    f = StabilityFilter(window=10, change_min_frames=3)
    for v in [20] * 10:
        f.update(4, v)
    high = f.update(4, 20).stability_confidence
    for v in [40, 40, 40, 40, 40]:
        f.update(4, v)
    low = f.update(4, 40).stability_confidence
    assert high > low, (high, low)


def test_independent_piles_are_isolated():
    f = StabilityFilter(window=5, change_min_frames=2)
    for _ in range(5):
        f.update(1, 10)
        f.update(2, 50)
    assert f.get(1) == 10
    assert f.get(2) == 50


def test_force_sets_manual_value():
    f = StabilityFilter(window=5, change_min_frames=2)
    for v in [10] * 5:
        f.update(1, v)
    f.force(1, 25)
    assert f.get(1) == 25
    assert f.is_locked(1)
    # o valor forçado resiste a leituras divergentes da IA
    for v in [99] * 5:
        r = f.update(1, v)
    assert r.stable_count == 25, "correção manual não pode ser sobrescrita pela IA"


def test_unlock_returns_control_to_ai():
    f = StabilityFilter(window=5, change_min_frames=2)
    for v in [10] * 5:
        f.update(1, v)
    f.force(1, 25)
    f.unlock(1)
    assert not f.is_locked(1)
    for v in [30] * 3:
        r = f.update(1, 30)
    assert r.stable_count == 30


def test_remove_clears_state():
    f = StabilityFilter()
    f.update(7, 10)
    f.remove(7)
    assert f.get(7) is None


def test_history_is_recorded():
    f = StabilityFilter(window=5)
    for v in [3, 3, 4]:
        f.update(1, v)
    assert f.history(1) == [3, 3, 4]


# ------------------------------------------------------------------ estados
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(count=20, confidence=0.9, stability_confidence=0.9, partial=False), CountStatus.STABLE),
        (dict(count=20, confidence=0.9, stability_confidence=0.3, partial=False), CountStatus.UNSTABLE),
        (dict(count=20, confidence=0.3, stability_confidence=0.9, partial=False), CountStatus.LOW_CONFIDENCE),
        (dict(count=20, confidence=0.9, stability_confidence=0.9, partial=True), CountStatus.PARTIAL),
        (dict(count=0, confidence=0.9, stability_confidence=0.9, partial=False), CountStatus.UNKNOWN),
    ],
)
def test_decide_status(kwargs, expected):
    assert decide_status(**kwargs) is expected


def test_low_confidence_wins_over_unstable():
    st = decide_status(count=20, confidence=0.2, stability_confidence=0.1, partial=False)
    assert st is CountStatus.LOW_CONFIDENCE


def test_unknown_wins_over_everything():
    st = decide_status(count=0, confidence=0.0, stability_confidence=0.0, partial=True)
    assert st is CountStatus.UNKNOWN
