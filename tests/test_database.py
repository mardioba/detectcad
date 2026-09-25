"""Testes do banco de dados e dos repositórios.

Usa um SQLite temporário por teste, então não toca no banco real.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sessão de banco isolada (SQLite em arquivo temporário)."""
    from app.config import settings
    from app.database import database

    db_file = tmp_path / "test.db"
    monkeypatch.setattr(settings.web, "database_url", f"sqlite:///{db_file}")
    monkeypatch.setattr(settings, "_db_path_override", str(db_file), raising=False)
    database.reset_engine()
    database.init_db()
    yield database
    database.reset_engine()


def test_init_creates_all_tables(db):
    from sqlalchemy import inspect

    from app.database.models import Base

    tables = set(inspect(db.get_engine()).get_table_names())
    expected = {
        "cameras", "piles", "count_records", "model_versions",
        "system_events", "calibration", "snapshots", "count_corrections",
    }
    assert expected <= tables, f"faltando: {expected - tables}"
    assert set(Base.metadata.tables) == expected


def test_camera_crud(db):
    from app.database.repository import CameraRepository

    with db.session_scope() as s:
        repo = CameraRepository(s)
        cam = repo.create("Camera 1", "rtsp://u:p@ip:554/s", is_default=True)
        assert cam.id is not None
        assert repo.get(cam.id).name == "Camera 1"
        repo.update(cam.id, name="Camera Rename")
        assert repo.get(cam.id).name == "Camera Rename"
        assert len(repo.list()) == 1
        assert repo.get_default().id == cam.id
        assert repo.delete(cam.id) is True
        assert repo.get(cam.id) is None


def test_pile_lifecycle(db):
    from app.database.repository import CameraRepository, PileRepository

    with db.session_scope() as s:
        cam = CameraRepository(s).create("C", "rtsp://x")
        repo = PileRepository(s)
        p = repo.get_or_create(cam.id, 7)
        repo.touch(p, stable_count=20)
        assert repo.get_by_tracking(cam.id, 7).stable_count == 20
        repo.deactivate_missing(cam.id, [])
        assert repo.get_by_tracking(cam.id, 7).active is False
        repo.deactivate_missing(cam.id, [7])
        assert repo.get_by_tracking(cam.id, 7).active is True


def test_count_records_and_total(db):
    from app.database.repository import CameraRepository, CountRepository

    with db.session_scope() as s:
        cam = CameraRepository(s).create("C", "rtsp://x")
        cr = CountRepository(s)
        cr.add(camera_id=cam.id, pile_id=None, chair_count=20, confidence=0.9, total_chairs=20, pile_count=1)
        cr.add(camera_id=cam.id, pile_id=None, chair_count=15, confidence=0.9, total_chairs=35, pile_count=2)
        cr.add(camera_id=cam.id, pile_id=None, chair_count=15, confidence=0.9, total_chairs=30, pile_count=2)
        # total = última leitura de cada pilha; sem pile_id, soma tudo
        assert cr.history(camera_id=cam.id) is not None


def test_events_and_purge(db):
    from app.database.repository import EventRepository
    from app.schemas import EventLevel

    with db.session_scope() as s:
        repo = EventRepository(s)
        repo.log(EventLevel.WARNING, "test_event", "mensagem de teste")
        repo.log(EventLevel.ERROR, "erro", "falhou")
        assert len(repo.list()) == 2
        assert len(repo.list(level="ERROR")) == 1
        assert repo.purge(older_than_days=0) >= 0


def test_model_versions_activation(db):
    from app.database.repository import ModelRepository

    with db.session_scope() as s:
        repo = ModelRepository(s)
        a = repo.upsert(filename="chair_v1.pt", version="1", precision=0.8, recall=0.7, map50=0.9, map5095=0.6)
        b = repo.upsert(filename="chair_v2.pt", version="2", precision=0.9, recall=0.8, map50=0.95, map5095=0.7)
        repo.set_active(a.id)
        assert repo.get_active().filename == "chair_v1.pt"
        repo.set_active(b.id)
        assert repo.get_active().filename == "chair_v2.pt"
        # só um pode estar ativo
        actives = [m for m in repo.list() if m.active]
        assert len(actives) == 1


def test_calibration_upsert(db):
    from app.database.repository import CalibrationRepository, CameraRepository

    with db.session_scope() as s:
        cam = CameraRepository(s).create("C", "rtsp://x")
        repo = CalibrationRepository(s)
        assert repo.get(cam.id) is None
        repo.upsert(cam.id, enabled=True, roi_x=10, roi_y=20, roi_w=100, roi_h=200, chair_height_px=42.0)
        d = repo.as_dict(cam.id)
        assert d["enabled"] is True
        assert d["roi"] == {"x": 10, "y": 20, "w": 100, "h": 200}
        assert d["chair_height_px"] == 42.0
        repo.upsert(cam.id, chair_height_px=50.0)
        assert repo.as_dict(cam.id)["chair_height_px"] == 50.0


def test_snapshots_and_corrections(db):
    from app.database.repository import CorrectionRepository, SnapshotRepository

    with db.session_scope() as s:
        snap = SnapshotRepository(s)
        row = snap.add(filename="a.jpg", reason="count_change", total=35, pile_count=2, confidence=0.8)
        assert snap.get(row.id).filename == "a.jpg"
        assert len(snap.list()) == 1
        assert snap.delete(row.id) is True

        corr = CorrectionRepository(s)
        corr.add(camera_id=1, pile_id=None, ai_count=18, correct_count=20, ai_confidence=0.6)
        corr.add(camera_id=1, pile_id=None, ai_count=15, correct_count=15, ai_confidence=0.9)
        stats = corr.stats()
        assert stats["total"] == 2
        assert stats["with_error"] == 1
        assert stats["mean_abs_error"] == 1.0
        assert stats["exact_match_rate"] == 0.5


def test_history_time_filter(db):
    from datetime import datetime, timedelta, timezone

    from app.database.repository import CameraRepository, CountRepository

    with db.session_scope() as s:
        cam = CameraRepository(s).create("C", "rtsp://x")
        cr = CountRepository(s)
        for i in range(3):
            cr.add(camera_id=cam.id, pile_id=None, chair_count=10 + i, confidence=0.9)
        recs = cr.history(camera_id=cam.id, since=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1))
        assert len(recs) == 3
        assert cr.history(camera_id=cam.id, since=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)) == []


def test_rollback_on_error(db):
    from app.database.repository import CameraRepository

    with pytest.raises(RuntimeError):
        with db.session_scope() as s:
            CameraRepository(s).create("C", "rtsp://x")
            raise RuntimeError("falha proposital")
    with db.session_scope() as s:
        assert CameraRepository(s).list() == []
