import pytest
from fastapi.testclient import TestClient
from env.server import app
from env.models import DataAction, ActionType

client = TestClient(app)

def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_reset_returns_observation():
    resp = client.post("/reset?task_id=1")
    assert resp.status_code == 200
    assert "dataset_preview" in resp.json()

def test_step_returns_stepresult():
    action = {"action_type": "NOOP", "justification": "test noop"}
    resp = client.post("/step?task_id=1", json=action)
    assert resp.status_code == 200
    assert "reward" in resp.json()

def test_grader_range():
    resp = client.get("/grader?task_id=1")
    assert resp.status_code == 200
    score = resp.json()["score"]
    assert 0.0 <= score <= 1.0

def test_baseline():
    resp = client.get("/baseline")
    assert resp.status_code == 200
    assert resp.json()["score"] == 0.0
