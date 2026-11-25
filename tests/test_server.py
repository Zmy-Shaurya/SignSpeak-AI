"""API tests against the real app. Camera 9 doesn't exist, so no hardware is needed and the camera reports an error."""
import time

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(scope="module")
def client():
    server.config.update(camera=9, backend="any")
    with TestClient(server.app) as c:
        c.post("/api/settings", json={"mute": True})  # never play audio in tests
        yield c


def wait_for(check, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


def test_pages_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_health_reports_missing_camera(client):
    assert wait_for(lambda: client.get("/api/health").json()["camera"] == "error")
    body = client.get("/api/health").json()
    assert body["status"] == "degraded" and len(body["model"]) == 12


def test_state_and_metrics_shape(client):
    state = client.get("/api/state").json()
    assert {"camera", "label", "probs", "words", "sentence", "settings", "model", "quality"} <= state.keys()
    metrics = client.get("/api/metrics").json()
    assert {"uptime_s", "fps", "inference_ms", "frames_inferred", "model"} <= metrics.keys()


def test_settings_are_validated(client):
    assert client.post("/api/settings", json={"threshold": 0.8, "hold": 4, "stride": 1}).json()["hold"] == 4
    assert client.post("/api/settings", json={"threshold": 2}).status_code == 422
    assert client.post("/api/settings", json={"stride": 9}).status_code == 422
    assert client.post("/api/camera", json={"index": -1}).status_code == 422


def test_edit_words(client):
    assert client.put("/api/words", json={"words": ["water", "please"]}).status_code == 200
    assert client.get("/api/state").json()["words"] == ["water", "please"]
    assert client.put("/api/words", json={"words": ["banana"]}).status_code == 422
    assert client.put("/api/words", json={"words": ["nothing"]}).status_code == 422
    assert client.post("/api/clear").status_code == 200
    assert client.get("/api/state").json()["words"] == []


def test_speak_now_and_repeat(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # plain sentences, no network
    assert client.post("/api/speak").status_code == 409  # no words yet
    client.put("/api/words", json={"words": ["hello", "help"]})
    assert client.post("/api/speak").status_code == 200
    assert client.get("/api/state").json()["words"] == []
    assert wait_for(lambda: client.get("/api/state").json()["sentence"] == "Hello, help.")
    assert client.post("/api/repeat").status_code == 200
