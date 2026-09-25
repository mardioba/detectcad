"""Testes do estado global e do pipeline de inferência.

O pipeline é testado com um ``FrameSource`` de imagem sintética e um modelo
**inexistente** de propósito: assim o caminho de "modo de preparação" (que é o
estado inicial de qualquer instalação nova) fica coberto sem depender de
download de pesos.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ai.model_manager import ModelManager
from app.ai.yolo_detector import YoloDetector
from app.camera.camera_manager import CameraManager
from app.schemas import CameraState, CountStatus
from app.services.inference_service import InferenceService
from app.services.state_store import StateStore, state_store


# ------------------------------------------------------------- state store
def test_totals_breakdown_has_pile_count():
    """Regressão: o pipeline lia 'pile_count' do breakdown e quebrava."""
    st = StateStore()
    d = st.totals_breakdown()
    assert "pile_count" in d
    assert d["pile_count"] == 0
    assert d["total_stable"] == 0


def test_totals_only_count_stable_piles():
    from app.schemas import PileState

    st = StateStore()
    st.set_piles(
        [
            PileState(pile_id=1, tracking_id=1, bbox=(0, 0, 1, 1), raw_count=20, stable_count=20,
                      confidence=0.9, detection_confidence=0.9, stability_confidence=0.9,
                      status=CountStatus.STABLE),
            PileState(pile_id=2, tracking_id=2, bbox=(0, 0, 1, 1), raw_count=15, stable_count=15,
                      confidence=0.4, detection_confidence=0.4, stability_confidence=0.9,
                      status=CountStatus.LOW_CONFIDENCE),
        ]
    )
    assert st.total(only_stable=True) == 20, "pilha de baixa confiança não entra no total confiável"
    assert st.total(only_stable=False) == 35
    b = st.totals_breakdown()
    assert b["total_stable"] == 20
    assert b["total_tentative"] == 15
    assert b["pile_count"] == 2


def test_manual_pile_enters_total_regardless_of_status():
    from app.schemas import PileState

    st = StateStore()
    st.set_piles([
        PileState(pile_id=1, tracking_id=1, bbox=(0, 0, 1, 1), raw_count=18, stable_count=20,
                  confidence=1.0, detection_confidence=0.0, stability_confidence=1.0,
                  status=CountStatus.STABLE, manual=True),
    ])
    assert st.total(only_stable=True) == 20


def test_overall_confidence_is_count_weighted():
    from app.schemas import PileState

    st = StateStore()
    st.set_piles([
        PileState(pile_id=1, tracking_id=1, bbox=(0, 0, 1, 1), raw_count=90, stable_count=90,
                  confidence=1.0, detection_confidence=1.0, stability_confidence=1.0, status=CountStatus.STABLE),
        PileState(pile_id=2, tracking_id=2, bbox=(0, 0, 1, 1), raw_count=10, stable_count=10,
                  confidence=0.5, detection_confidence=0.5, stability_confidence=0.5, status=CountStatus.STABLE),
    ])
    # ponderada por quantidade: a pilha maior domina
    assert st.overall_confidence() > 0.9


def test_frame_store_returns_new_jpeg_once():
    st = StateStore()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    st.set_frame(b"jpeg-bytes", img, 123.0)
    jpeg, counter = st.get_frame_jpeg(0)
    assert jpeg == b"jpeg-bytes" and counter == 1
    # com min_counter igual, não reentrega o mesmo frame
    assert st.get_frame_jpeg(counter)[0] is None


def test_snapshot_shape_and_events():
    st = StateStore()
    st.push_event("INFO", "teste", "mensagem")
    assert st.get_events()[0]["event"] == "teste"
    snap = st.snapshot()
    for key in ("piles", "total", "confidence", "pile_count", "camera_state"):
        assert key in snap


# --------------------------------------------------------- pipeline real
@pytest.fixture()
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pipeline completo com fonte de imagem e SEM modelo treinado."""
    from app.config import settings

    monkeypatch.setattr(settings.web, "database_url", f"sqlite:///{tmp_path / 'pipe.db'}")
    from app.database import database

    database.reset_engine()
    database.init_db()

    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), img)

    cameras = CameraManager()
    cameras.add(1, url=str(img_path), name="Teste", mode=__import__(
        "app.schemas", fromlist=["SourceMode"]).SourceMode.IMAGE, start=True)
    detector = YoloDetector(tmp_path / "sem_modelo.pt")  # não carrega
    models = ModelManager(detector)
    store = StateStore()
    svc = InferenceService(detector, models, cameras, camera_id=1, store=store)
    svc.start()
    yield svc
    svc.stop()
    cameras.stop_all()
    database.reset_engine()


def test_pipeline_runs_without_model(pipeline: InferenceService):
    """Sem modelo o sistema não quebra: segue rodando em modo preparação."""
    import time

    deadline = time.time() + 12
    while time.time() < deadline and pipeline.frames_processed < 2:
        time.sleep(0.2)
    assert pipeline.frames_processed >= 1, pipeline.status()
    # estado coerente: sem modelo não há contagem inventada
    assert pipeline.store.get_piles() == []
    assert pipeline.store.total() == 0
    assert pipeline.store.get_status()["model_loaded"] is False
    assert pipeline.store.frame_counter >= 1, "o overlay deve ser gerado mesmo sem contagem"


def test_pipeline_writes_overlay_frame(pipeline: InferenceService):
    import time

    deadline = time.time() + 12
    while time.time() < deadline and pipeline.store.frame_counter < 1:
        time.sleep(0.2)
    jpeg, _ = pipeline.store.get_frame_jpeg(0)
    assert jpeg and len(jpeg) > 500
    assert jpeg[:2] == b"\xff\xd8", "deve ser um JPEG válido"


def test_pipeline_manual_count_validation(pipeline: InferenceService):
    # pilha inexistente -> recusa e explica
    res = pipeline.manual_count(999, 20)
    assert res["ok"] is False


def test_pipeline_status_shape(pipeline: InferenceService):
    st = pipeline.status()
    for key in ("camera_id", "frames_processed", "process_fps", "inference_ms", "running"):
        assert key in st


def test_count_persisted_without_camera_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regressão: gravar contagem não pode falhar por FOREIGN KEY.

    Se a tabela ``cameras`` estiver vazia, o worker precisa criar a câmera
    sozinho - senão nenhuma contagem é gravada e o histórico fica perdido
    sem erro visível.
    """
    from app.config import settings
    from app.database import database

    monkeypatch.setattr(settings.web, "database_url", f"sqlite:///{tmp_path / 'fk.db'}")
    database.reset_engine()
    database.init_db()

    with database.session_scope() as s:
        from app.database.models import Camera

        assert s.query(Camera).count() == 0, "pré-condição: sem câmera cadastrada"

    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), img)

    from app.schemas import SourceMode

    cameras = CameraManager()
    cameras.add(1, url=str(img_path), name="FK", mode=SourceMode.IMAGE, start=True)
    detector = YoloDetector(tmp_path / "sem_modelo.pt")
    store = StateStore()
    svc = InferenceService(detector, ModelManager(detector), cameras, camera_id=1, store=store)
    svc.start()
    try:
        import time

        deadline = time.time() + 12
        while time.time() < deadline and svc.frames_processed < 2:
            time.sleep(0.2)
        assert svc.frames_processed >= 1
        assert svc.status()["persist_error"] == "", svc.status()["persist_error"]
        with database.session_scope() as s:
            from app.database.models import Camera

            assert s.query(Camera).count() == 1, "a câmera deveria ter sido criada"
    finally:
        svc.stop()
        cameras.stop_all()
        database.reset_engine()
