"""Auth behavior of the http server, no LLM involved.

Only the request gate is tested here: the agent behind it is covered by the
rest of the suite, and running it would need a live model. fastapi is an
optional extra, so the whole module skips when it isn't installed.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from secret_agent import server  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_run_without_server_key_is_503(client, monkeypatch):
    monkeypatch.setattr(server, "SERVER_KEY", "")
    r = client.post("/run", json={"task": "hi"}, headers={"x-api-key": "anything"})
    assert r.status_code == 503


def test_run_with_wrong_key_is_401(client, monkeypatch):
    monkeypatch.setattr(server, "SERVER_KEY", "right")
    r = client.post("/run", json={"task": "hi"}, headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_run_with_missing_key_header_is_401(client, monkeypatch):
    monkeypatch.setattr(server, "SERVER_KEY", "right")
    r = client.post("/run", json={"task": "hi"})
    assert r.status_code == 401


def test_empty_task_is_422(client, monkeypatch):
    monkeypatch.setattr(server, "SERVER_KEY", "right")
    r = client.post("/run", json={"task": ""}, headers={"x-api-key": "right"})
    assert r.status_code == 422
