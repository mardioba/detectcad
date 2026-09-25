"""Testes da camada de IA: pilhas, tracking e gerenciador de modelos.

O carregamento do YOLO real é opcional: se não houver modelo disponível, os
testes que dependem dele são pulados (``pytest.mark.skipif``) em vez de
inventar um resultado.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.model_manager import parse_version_from_name, read_metrics_from_run
from app.ai.pile_detector import PileDetector, build_pile_detector
from app.ai.tracker import PileTracker, group_by_x, resolve_tracker_config
from app.ai.yolo_detector import YoloDetector
from app.schemas import Detection, PileCandidate


def det(x1, y1, x2, y2, conf=0.9, cls=0, name="chair") -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, class_id=cls, class_name=name)


# ------------------------------------------------------------------ util
def test_iou_and_x_overlap():
    a = det(0, 0, 100, 100)
    b = det(50, 0, 150, 100)
    assert a.iou(b) == pytest.approx(5000 / 15000, rel=1e-3)
    from app.schemas import x_overlap_ratio

    assert x_overlap_ratio(b, a) == pytest.approx(0.5, rel=1e-3)
    far = det(500, 0, 600, 100)
    assert a.iou(far) == 0.0
    assert x_overlap_ratio(far, a) == 0.0


# ---------------------------------------------------------------- pilhas
def test_chair_group_forms_one_pile():
    pd = PileDetector(chair_classes={0}, pile_classes=set(), overlap_min=0.3, gap_px=40)
    chairs = [det(0, i * 30, 120, i * 30 + 28) for i in range(10)]
    piles = pd.detect(chairs)
    assert len(piles) == 1
    assert len(piles[0].chair_detections) == 10
    assert piles[0].source == "chair_group"


def test_separate_columns_form_separate_piles():
    pd = PileDetector(chair_classes={0}, pile_classes=set(), overlap_min=0.3, gap_px=40)
    chairs = []
    for base in (0, 400, 800):  # 3 pilhas lado a lado
        chairs += [det(base, i * 30, base + 120, i * 30 + 28) for i in range(8)]
    piles = pd.detect(chairs)
    assert len(piles) == 3, [p.bbox for p in piles]
    for p in piles:
        assert len(p.chair_detections) == 8


def test_pile_class_takes_precedence():
    pd = PileDetector(chair_classes={0}, pile_classes={1}, overlap_min=0.3, gap_px=40)
    pile_box = det(0, 0, 300, 600, conf=0.95, cls=1, name="pile")
    chairs = [det(20, i * 30, 260, i * 30 + 28) for i in range(12)]
    piles = pd.detect(pile_box and chairs + [pile_box])
    assert len(piles) == 1
    assert piles[0].source == "pile_class"
    assert len(piles[0].chair_detections) == 12


def test_chairs_outside_pile_are_ignored():
    pd = PileDetector(chair_classes={0}, pile_classes={1})
    pile_box = det(0, 0, 300, 600, cls=1, name="pile")
    inside = det(20, 10, 260, 38)
    outside = det(900, 10, 1140, 38)
    piles = pd.detect([pile_box, inside, outside])
    assert len(piles) == 1
    assert len(piles[0].chair_detections) == 1


def test_small_group_penalty_lowers_confidence():
    pd = PileDetector(chair_classes={0}, pile_classes=set())
    one = pd.detect([det(0, 0, 100, 30, conf=0.9)])
    many = pd.detect([det(0, i * 30, 100, i * 30 + 28, conf=0.9) for i in range(10)])
    assert one[0].conf < many[0].conf


def test_build_detector_resolves_names():
    pd = build_pile_detector({0: "chair", 1: "pile"})
    assert 0 in pd.chair_classes
    assert 1 in pd.pile_classes


def test_build_detector_falls_back_when_no_match():
    pd = build_pile_detector({0: "giz", 1: "mesa"})
    assert pd.chair_classes == {0, 1}, "sem classes conhecidas, assume todas como cadeira"


def test_group_by_x_helper():
    dets = [det(0, 0, 100, 20), det(20, 25, 120, 45), det(600, 0, 700, 20)]
    groups = group_by_x(dets, overlap_min=0.3, gap_px=40)
    assert len(groups) == 2


# -------------------------------------------------------------- tracking
def test_tracker_keeps_id_across_frames():
    t = PileTracker(iou_threshold=0.3, max_age=5)
    first = t.update([PileCandidate(bbox=(100, 0, 300, 400))])
    tid = first[0][0]
    second = t.update([PileCandidate(bbox=(105, 0, 305, 405))])
    assert second[0][0] == tid, "o ID da pilha não pode mudar a cada frame"


def test_tracker_survives_missed_frames():
    t = PileTracker(iou_threshold=0.3, max_age=3)
    tid = t.update([PileCandidate(bbox=(100, 0, 300, 400))])[0][0]
    t.update([])  # pilha sumiu por 1 frame
    out = t.update([PileCandidate(bbox=(100, 0, 300, 400))])
    assert out[0][0] == tid, "após um frame perdido, o ID deve ser preservado"


def test_tracker_expires_after_max_age():
    t = PileTracker(iou_threshold=0.3, max_age=2)
    t.update([PileCandidate(bbox=(0, 0, 100, 200))])
    for _ in range(4):
        t.update([])
    assert t.track_count == 0


def test_tracker_gives_new_id_to_new_pile():
    t = PileTracker(iou_threshold=0.3, max_age=5)
    a = t.update([PileCandidate(bbox=(0, 0, 100, 200))])[0][0]
    out = t.update([PileCandidate(bbox=(0, 0, 100, 200)), PileCandidate(bbox=(600, 0, 700, 200))])
    ids = {o[0] for o in out}
    assert a in ids and len(ids) == 2


def test_tracker_ids_never_reused():
    t = PileTracker(iou_threshold=0.3, max_age=1)
    first = t.update([PileCandidate(bbox=(0, 0, 100, 200))])[0][0]
    for _ in range(4):
        t.update([])
    later = t.update([PileCandidate(bbox=(0, 0, 100, 200))])[0][0]
    assert later != first, "ID encerrado não pode ser reutilizado"


def test_resolve_tracker_config():
    assert resolve_tracker_config("internal") == ("internal", None)
    assert resolve_tracker_config("bytetrack") == ("ultralytics", "bytetrack.yaml")
    assert resolve_tracker_config("botsort") == ("ultralytics", "botsort.yaml")
    assert resolve_tracker_config("qualquer") == ("internal", None)


# --------------------------------------------------------------- modelos
def test_parse_version_from_name():
    assert parse_version_from_name("chair_counter_v3.pt") == ("chair_counter", "3")
    assert parse_version_from_name("chair_v1.2.3.pt") == ("chair", "1.2.3")
    assert parse_version_from_name("chair_counter.pt") == ("chair_counter", "1.0")


def test_read_metrics_returns_none_when_absent(tmp_path: Path):
    metrics = read_metrics_from_run(tmp_path)
    assert metrics.get("precision") is None
    assert metrics.get("map50") is None, "não pode inventar métrica sem results.csv"


def test_read_metrics_from_results_csv(tmp_path: Path):
    (tmp_path / "results.csv").write_text(
        "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.5,0.40,0.50,0.45,0.20\n"
        "2,0.4,0.70,0.65,0.72,0.44\n",
        encoding="utf-8",
    )
    m = read_metrics_from_run(tmp_path)
    assert m["epochs"] == 2
    assert m["precision"] == pytest.approx(0.70)
    assert m["map50"] == pytest.approx(0.72)
    assert m["map5095"] == pytest.approx(0.44)


# -------------------------------------------------------------- detector
def test_detector_reports_missing_model_cleanly(tmp_path: Path):
    det = YoloDetector(tmp_path / "nao_existe.pt")
    assert det.is_available() is False
    assert det.load() is False
    assert "não encontrado" in det.load_error
    assert det.info()["loaded"] is False
    assert det.info()["trained_model"] is False


def test_detector_info_has_gpu_section():
    info = YoloDetector().info()
    assert "gpu" in info
    assert "cuda_available" in info["gpu"]
    assert isinstance(info["using_gpu"], bool)


def test_detector_class_indices_empty_model():
    det = YoloDetector()
    chairs, piles = det.class_indices()
    assert chairs == set() and piles == set()
