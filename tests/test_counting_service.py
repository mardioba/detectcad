"""Testes da fachada de contagem (:mod:`app.services.counting_service`)."""

from __future__ import annotations

import numpy as np
import pytest

from app.schemas import CountStatus, PileState
from app.services.counting_service import CountingService
from app.services.state_store import StateStore


@pytest.fixture()
def service() -> CountingService:
    return CountingService(store=StateStore())


def make_pile(pid: int, count: int, status: CountStatus = CountStatus.STABLE, **kw) -> PileState:
    """Cria uma pilha de estado, permitindo sobrescrever campos."""
    fields: dict = dict(
        pile_id=pid, tracking_id=pid, bbox=(0, 0, 100, 400), raw_count=count,
        stable_count=count, confidence=0.9, detection_confidence=0.9,
        stability_confidence=0.9,
    )
    fields.update(kw)
    fields["status"] = status
    return PileState(**fields)


def test_snapshot_shape(service: CountingService):
    service.store.set_piles([make_pile(1, 20), make_pile(2, 15)])
    snap = service.snapshot()
    assert snap["total"] == 35
    assert snap["pile_count"] == 2
    assert len(snap["piles"]) == 2
    for p in snap["piles"]:
        assert "status_label" in p and "color" in p
        assert p["needs_attention"] is False


def test_describe_flags_uncertain_piles(service: CountingService):
    service.store.set_piles([
        make_pile(1, 20),
        make_pile(2, 15, CountStatus.LOW_CONFIDENCE, confidence=0.3),
        make_pile(3, 8, CountStatus.PARTIAL),
    ])
    snap = service.snapshot()
    attention = {p["pile_id"]: p["needs_attention"] for p in snap["piles"]}
    assert attention == {1: False, 2: True, 3: True}
    # o total confiável só pega a pilha estável
    assert snap["total"] == 20 + 8
    assert snap["totals"]["total_tentative"] == 15


def test_status_colors_are_valid_hex(service: CountingService):
    for status in CountStatus:
        d = service.describe(make_pile(1, 5, status))
        assert d["color"].startswith("#") and len(d["color"]) == 7
        assert d["status_label"]


# --------------------------------------------------------------- validação
def test_manual_count_validates_range(service: CountingService):
    assert service.apply_manual_count(None, 1, -5)["ok"] is False
    assert service.apply_manual_count(None, 1, 99999)["ok"] is False
    assert service.apply_manual_count(None, 1, 20)["ok"] is False, "sem worker deve recusar"


def test_release_manual_count_without_worker(service: CountingService):
    assert service.release_manual_count(None, 1)["ok"] is False


# ------------------------------------------------------------- diagnóstico
def test_diagnose_roi_requires_frame(service: CountingService):
    r = service.diagnose_roi(None, {"x": 0, "y": 0, "w": 10, "h": 10}, 0)
    assert r["ok"] is False
    assert "frame" in r["message"].lower()


def test_diagnose_roi_validates_shape(service: CountingService):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert service.diagnose_roi(frame, {"x": 0}, 0)["ok"] is False
    assert service.diagnose_roi(frame, {"x": 0, "y": 0, "w": 4, "h": 4}, 0)["ok"] is False


def test_diagnose_roi_on_synthetic_stack(service: CountingService):
    from tools.make_synthetic_stack import draw_stack, make_scene

    img = make_scene([12], width=420, height=720, chair_h=30, n_piles=1)
    x1, y1, x2, y2 = draw_stack(img, 210, 633, 12, chair_h=30, noise=0.01)
    r = service.diagnose_roi(img, {"x": x1 - 10, "y": y1 - 10, "w": (x2 - x1) + 20, "h": (y2 - y1) + 20}, 30.0)
    assert r["ok"] is True
    assert abs(r["estimate"]["count"] - 12) <= 1
    assert r["analysis"]["pitch"] > 0


def test_measure_chair_height(service: CountingService):
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    r = service.measure_chair_height(frame, {"x": 50, "y": 60, "w": 100, "h": 140})
    assert r["ok"] is True
    assert r["measured_height_px"] == 140
    assert r["measured_width_px"] == 100
    assert "140" in r["message"]


def test_measure_chair_height_out_of_bounds(service: CountingService):
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    r = service.measure_chair_height(frame, {"x": 350, "y": 350, "w": 200, "h": 200})
    assert r["ok"] is False


# ----------------------------------------------------------------- feedback
def test_error_statistics_empty(service: CountingService):
    stats = service.error_statistics([])
    assert stats["samples"] == 0
    assert stats["mean_abs_error"] is None


def test_error_statistics_with_corrections(service: CountingService):
    stats = service.error_statistics([(18, 20), (15, 15), (30, 32), (10, 9)])
    assert stats["samples"] == 4
    assert stats["mean_abs_error"] == pytest.approx(1.25)
    assert stats["exact_match_rate"] == pytest.approx(0.25)
    assert stats["max_abs_error"] == 2


def test_offset_suggestion_needs_evidence(service: CountingService):
    weak = service.offset_suggestion([(20, 21)])
    assert weak["reliable"] is False
    strong = service.offset_suggestion([(20, 21), (30, 31), (40, 41), (50, 51)])
    assert strong["offset"] == 1
    assert strong["reliable"] is True
