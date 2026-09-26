"""Testes da API REST e do WebSocket.

Sobe a aplicação FastAPI em modo de teste (sem câmera e sem modelo) e
verifica os endpoints, a segurança (senha mascarada) e o WebSocket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import settings
    from app.database import database

    monkeypatch.setattr(settings.web, "database_url", f"sqlite:///{tmp_path / 'api.db'}")
    database.reset_engine()
    database.init_db()

    import app.main as main_mod

    app = main_mod.build_app()
    with TestClient(app) as c:
        yield c
    database.reset_engine()


# ------------------------------------------------------------------ saúde
def test_health_and_docs(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # Sem modelo, o sistema continua saudável (modo de preparação)
    assert body["healthy"] is True
    assert body["ready"] is False


def test_dashboard_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CONTADOR" in r.text.upper()
    assert "dashboard.js" in r.text


def test_openapi_lists_endpoints(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for expected in (
        "/api/status", "/api/cameras", "/api/count", "/api/history", "/api/piles",
        "/api/models", "/api/dataset", "/api/training/status", "/api/calibration",
        "/api/events", "/api/snapshots", "/api/health",
    ):
        assert expected in paths, f"endpoint ausente: {expected}"


# ---------------------------------------------------------------- segurança
def test_rtsp_password_is_never_exposed(client):
    # enabled=false: não tentamos abrir conexão de verdade (o teste é sobre
    # mascaramento, não sobre rede).
    r = client.post(
        "/api/cameras",
        json={
            "name": "Teste",
            "rtsp_url": "rtsp://user:sup3rsegredo@10.0.0.5:554/onvif1",
            "enabled": False,
        },
    )
    assert r.status_code == 200
    assert "sup3rsegredo" not in r.text
    assert "****" in r.json()["camera"]["rtsp_url"]

    r2 = client.get("/api/cameras")
    assert "sup3rsegredo" not in r2.text
    r3 = client.get("/api/config")
    assert "sup3rsegredo" not in r3.text


def test_mask_url_function():
    from app.config import mask_url

    assert mask_url("rtsp://u:senha@1.2.3.4:554/x") == "rtsp://u:****@1.2.3.4:554/x"
    assert mask_url("rtsp://u@1.2.3.4/x") == "rtsp://u@1.2.3.4/x"
    assert mask_url("") == ""


def test_create_camera_requires_url(client):
    r = client.post("/api/cameras", json={"name": "Sem URL"})
    assert r.status_code == 400
    assert r.json()["ok"] is False


# ----------------------------------------------------------------- status
def test_status_shape(client):
    body = client.get("/api/status").json()
    assert body["ok"] is True
    for key in ("system", "camera", "piles", "total", "model", "history", "training"):
        assert key in body, key


def test_count_shape(client):
    body = client.get("/api/count").json()
    assert body["ok"] is True
    assert body["total"] == 0
    assert body["piles"] == []


def test_piles_and_history_empty(client):
    assert client.get("/api/piles").json()["piles"] == []
    hist = client.get("/api/history?hours=24").json()
    assert hist["ok"] is True
    assert hist["records"] == []


# ----------------------------------------------------------------- modelos
def test_models_endpoint_reports_not_trained(client):
    body = client.get("/api/models").json()
    assert body["ok"] is True
    assert "models" in body
    # Não deve inventar métricas
    for m in body["models"]:
        assert m.get("map50") is None or isinstance(m.get("map50"), (int, float))


def test_activate_unknown_model_fails_cleanly(client):
    r = client.post("/api/models/activate", json={"filename": "nao_existe.pt"})
    assert r.status_code == 400
    assert r.json()["ok"] is False


# ---------------------------------------------------------------- calibração
def test_calibration_roundtrip(client):
    r = client.post(
        "/api/calibration",
        json={
            "enabled": True,
            "roi": {"x": 100, "y": 50, "w": 800, "h": 600},
            "chair_height_px": 42.5,
            "stability_frames": 15,
            "counting_method": "periodicity",
            "count_offset": 1,
            "chair_classes": "chair",
        },
    )
    assert r.status_code == 200
    calib = r.json()["calibration"]
    assert calib["chair_height_px"] == 42.5
    assert calib["roi"]["w"] == 800
    assert calib["stability_frames"] == 15
    assert calib["count_offset"] == 1

    r2 = client.get("/api/calibration").json()
    assert r2["calibration"]["chair_height_px"] == 42.5
    assert r2["calibration"]["counting_method"] == "periodicity"

    r3 = client.post("/api/calibration/reset").json()
    assert r3["calibration"]["stability_frames"] != 15 or r3["calibration"]["chair_height_px"] == 0.0


def test_calibration_writes_json_file(client):
    client.post("/api/calibration", json={"chair_height_px": 33.0})
    from app.config import settings

    path = settings.config_dir / "camera_1.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["chair_height_px"] == 33.0


# ----------------------------------------------------------------- dataset
def test_dataset_stats_empty(client):
    body = client.get("/api/dataset").json()
    assert body["ok"] is True
    assert body["stats"]["total_images"] == 0
    assert body["stats"]["pending"] == 0


def test_dataset_split_rejects_bad_proportions(client):
    r = client.post("/api/dataset/split", json={"train": 0.5, "val": 0.5, "test": 0.5, "seed": 1})
    assert r.status_code == 400


def test_dataset_split_without_images(client):
    r = client.post("/api/dataset/split", json={"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42})
    assert r.status_code == 400
    assert "imagem" in r.json()["message"].lower()


# -------------------------------------------------------------- treinamento
def test_training_status_idle(client):
    body = client.get("/api/training/status").json()
    assert body["ok"] is True
    assert body["running"] is False


def test_training_start_without_dataset_fails(client):
    r = client.post("/api/training/start", json={"epochs": 1})
    assert r.status_code in (400, 409)
    assert "imagem" in r.json()["message"].lower()


def test_training_ignores_gitkeep_placeholders(client, monkeypatch: pytest.MonkeyPatch):
    """Regressão: .gitkeep não pode contar como imagem de treino.

    Sem isso, o botão de treinamento aceitava um dataset vazio (só o
    placeholder do git) e disparava um treino de verdade em processo
    separado - que continuava rodando depois do teste.
    """
    import subprocess

    from app.config import settings
    from app.services import training_service as training_service_mod

    train_dir = settings.dataset_dir / "images" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / ".gitkeep").write_text("", encoding="utf-8")
    (train_dir / "README.txt").write_text("não sou imagem", encoding="utf-8")

    def fail(*_a, **_k):  # pragma: no cover
        raise AssertionError("não deveria ter iniciado processo de treino")

    # Mira o `subprocess` do MÓDULO DO SERVIÇO, não o `subprocess` global.
    #
    # Com o monkeypatch global, qualquer biblioteca que internallyamente
    # invoque um processo durante um import preguiçoso dispara o guard. Foi
    # exatamente o que aconteceu: `cuda.pathfinder` chama
    # `subprocess.Popen(['/sbin/ldconfig', '-p'])` (via ctypes.util.
    # find_library) na primeira vez que o torch é carregado, e esse guard
    # acusava um "treino iniciado" que nunca aconteceu. O teste falhava de
    # forma intermitente, dependendo de o import já ter aquecido ou não -
    # um teste que às vezes mente é pior do que nenhum.
    monkeypatch.setattr(training_service_mod.subprocess, "Popen", fail)
    runs_dir = settings.base_dir / "training" / "runs"
    # Foto do antes: o teste precisa garantir que NENHUM treino novo nasceu, e
    # não que a pasta esteja vazia. `training/runs` é o diretório REAL
    # compartilhado: afirmar "não existe chair_counter*" quebra na segunda
    # execução da suíte, porque o teste anterior pode ter deixado um
    # diretório para trás. O sintoma é um teste que falha sozinho depois de
    # passar uma vez.
    before = set(runs_dir.glob("chair_counter*")) if runs_dir.is_dir() else set()
    r = client.post("/api/training/start", json={"epochs": 1})
    assert r.status_code in (400, 409)
    assert "imagem" in r.json()["message"].lower()
    after = set(runs_dir.glob("chair_counter*")) if runs_dir.is_dir() else set()
    assert after == before, f"treino indevido subiu: {after - before}"


# ------------------------------------------------------------------- logs
def test_logs_restricted_to_known_files(client):
    ok = client.get("/api/logs?file=app.log")
    assert ok.status_code == 200
    bad = client.get("/api/logs?file=../../etc/passwd")
    assert bad.status_code == 400
    assert "não permitido" in bad.json()["message"]


def test_dataset_image_path_traversal_blocked(client):
    r = client.get("/api/dataset/image", params={"path": "../../../etc/passwd"})
    assert r.status_code in (400, 404)


# --------------------------------------------------------------- eventos
def test_events_list_and_clear(client):
    assert client.get("/api/events").json()["events"] == []
    assert client.delete("/api/events").json()["ok"] is True


# ------------------------------------------------------------- snapshots
def test_snapshot_without_frame_returns_error(client):
    r = client.post("/api/snapshots")
    assert r.status_code in (400, 503)


# ------------------------------------------------------------- websocket
def test_websocket_sends_state(client):
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        state = ws.receive_json()
        assert state["type"] == "state"
        assert "piles" in state and "total" in state


def test_websocket_status_endpoint(client):
    body = client.get("/ws/status").json()
    assert "connections" in body


# ------------------------------------------------------------- correções
def test_correction_validation(client):
    r = client.post("/api/count/999/correct", json={"count": 20})
    assert r.status_code in (400, 503)  # pilha inexistente ou worker parado
    r2 = client.post("/api/count/1/correct", json={"count": "abc"})
    assert r2.status_code == 400
