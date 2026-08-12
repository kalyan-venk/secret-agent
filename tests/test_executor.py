"""The executor split: the orchestrator reaches command execution over HTTP.

Two things get proven here. First, the HTTP contract: a command goes out as a
request, the output comes back, and a refusal comes back as an error the client
turns into the same ToolError an in-process run would raise. Second, that the
guardrails run on the executor side of the wire, not the caller's -- `cat
/etc/passwd` is refused by the service, so moving execution to another node did
not move the confinement off it.

The FastAPI app is driven in-process through an ASGI transport, so these are
deterministic and need no live socket. The genuinely-two-process proof (separate
PIDs, real uvicorn) is scripts/executor_demo.py, which is meant to be watched.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from secret_agent.executor import service  # noqa: E402
from secret_agent.executor.client import ExecutorClient  # noqa: E402
from secret_agent.tools.base import ToolError  # noqa: E402
from secret_agent.tools.shell import Bash  # noqa: E402


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_text("hello from the executor node\n")
    (root / ".env").write_text("SECRET_KEY=hunter2\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd.txt").write_text("root:x:0:0\n")
    monkeypatch.setenv("SA_ROOT", str(root))
    return root


@pytest.fixture
def executor(project, monkeypatch):
    """An ExecutorClient wired to the in-process FastAPI app with a real key."""
    monkeypatch.setattr(service, "EXECUTOR_KEY", "test-key")
    http = TestClient(service.app)
    client = ExecutorClient("http://executor", key="test-key", client=http)
    yield client
    http.close()


def test_a_command_runs_on_the_executor_and_the_output_comes_back(executor, project):
    out = executor.execute("cat notes.txt")
    assert "hello from the executor node" in out


def test_health_reports_the_executor_pid_and_root(project, monkeypatch):
    monkeypatch.setattr(service, "EXECUTOR_KEY", "test-key")
    http = TestClient(service.app)
    r = http.get("/health")
    http.close()
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["pid"] > 0
    assert str(project) in body["root"]


def test_path_confinement_runs_on_the_executor_side(executor):
    # the whole point: moving execution off-process did not move the guardrail
    with pytest.raises(ToolError, match="outside the project root"):
        executor.execute("cat /etc/passwd")


def test_credential_files_are_refused_on_the_executor_side(executor):
    with pytest.raises(ToolError, match="credential"):
        executor.execute("cat .env")


def test_non_allowlisted_program_is_refused_on_the_executor_side(executor):
    with pytest.raises(ToolError, match="allowlist"):
        executor.execute("rm -rf notes.txt")


def test_nonzero_exit_comes_back_as_output_not_an_error(executor):
    out = executor.execute("grep zzzznotpresent notes.txt")
    assert "[exit 1" in out


def test_a_missing_executor_key_is_a_503(project, monkeypatch):
    monkeypatch.setattr(service, "EXECUTOR_KEY", "")
    http = TestClient(service.app)
    client = ExecutorClient("http://executor", key="anything", client=http)
    with pytest.raises(ToolError, match="503"):
        client.execute("ls")
    http.close()


def test_a_wrong_executor_key_is_a_401(project, monkeypatch):
    monkeypatch.setattr(service, "EXECUTOR_KEY", "right")
    http = TestClient(service.app)
    client = ExecutorClient("http://executor", key="wrong", client=http)
    with pytest.raises(ToolError, match="401"):
        client.execute("ls")
    http.close()


def test_bash_tool_routes_to_the_executor_when_the_url_is_set(project, monkeypatch):
    # With SA_EXECUTOR_URL pointed at a dead port, bash must fail "unreachable"
    # rather than silently running the command in this process. That failure IS
    # the proof there is no local fallback: the command genuinely only runs on
    # the executor node.
    monkeypatch.setenv("SA_EXECUTOR_URL", "http://127.0.0.1:1")  # nothing listens
    monkeypatch.setenv("SA_EXECUTOR_KEY", "irrelevant")
    with pytest.raises(ToolError, match="unreachable"):
        Bash().run(command="ls")


def test_bash_tool_runs_in_process_when_no_executor_is_configured(project, monkeypatch):
    monkeypatch.delenv("SA_EXECUTOR_URL", raising=False)
    out = Bash().run(command="cat notes.txt")
    assert "hello from the executor node" in out
