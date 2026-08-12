"""The executor node: bash execution behind an HTTP endpoint.

One POST /execute endpoint, guarded by a static key in the x-executor-key
header, running the same validate -> confine -> spawn path (executor.core) that
the in-process bash tool runs. The orchestrator calls this instead of spawning
subprocesses itself, so a compromise of the command tool is contained to this
process, on this box, with this root -- not the process that holds the model
keys and the whole conversation.

Runs on localhost today; nothing here assumes it. Point SA_EXECUTOR_URL at
another host and the executor is genuinely a second node. SA_ROOT pins what this
node may touch, independently of wherever the orchestrator lives.

Auth follows server.py exactly, and for the same reason: a missing key refuses
to start serving rather than serving open. An unset env var must never mean "no
auth" on a service whose whole job is running commands.

Run with:
    SA_EXECUTOR_KEY=... SA_ROOT=/path/to/project \
        uvicorn secret_agent.executor.service:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..tools.base import ToolError
from .core import run_command

EXECUTOR_KEY = os.environ.get("SA_EXECUTOR_KEY", "")

log = logging.getLogger("secret_agent.executor")

# Opt-in request logging to stdout. Off by default (so importing this under a
# test suite stays silent); SA_EXECUTOR_LOG=1 attaches a handler so an operator
# -- or the two-process demo -- can watch which commands actually landed on this
# node. Guarded so repeated imports don't stack handlers.
if os.environ.get("SA_EXECUTOR_LOG", "").strip().lower() in ("1", "true", "yes", "on"):
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(_h)
    log.setLevel(logging.INFO)

app = FastAPI(title="secret-agent-executor")


class ExecRequest(BaseModel):
    command: str = Field(min_length=1, max_length=8000)


class ExecResponse(BaseModel):
    ok: bool
    output: str = ""
    error: str | None = None


def _root() -> Path:
    return Path(os.environ.get("SA_ROOT", os.getcwd())).resolve()


def _check_key(x_executor_key: str) -> None:
    if not EXECUTOR_KEY:
        raise HTTPException(status_code=503, detail="executor has no SA_EXECUTOR_KEY set")
    if not secrets.compare_digest(x_executor_key, EXECUTOR_KEY):
        raise HTTPException(status_code=401, detail="bad executor key")


@app.get("/health")
def health() -> dict:
    # pid and root are here so a caller can prove the command ran on this node
    # (a different process, with a different root) and not in the orchestrator.
    return {"status": "ok", "pid": os.getpid(), "root": str(_root())}


@app.post("/execute", response_model=ExecResponse)
def execute(req: ExecRequest, x_executor_key: str = Header(default="")) -> ExecResponse:
    _check_key(x_executor_key)
    try:
        out = run_command(req.command, _root())
    except ToolError as e:
        # A refusal is a normal outcome, not a 500. It goes back as ok=False so
        # the client re-raises it as the model-facing ToolError it would have
        # been in-process.
        log.info("executor pid=%s refused %r: %s", os.getpid(), req.command, e)
        return ExecResponse(ok=False, error=str(e))
    log.info("executor pid=%s ran %r", os.getpid(), req.command)
    return ExecResponse(ok=True, output=out)
