"""HTTP front door for the agent.

One POST /run endpoint, guarded by a static key in the x-api-key header, so the
agent can be called from anywhere the same way a hosted LLM API is called. Each
request builds a fresh Agent (fresh conversation, fresh permissions), runs the
task to completion, and returns the answer plus the run stats that matter when
you compare runs: iterations, tool calls, repair rate, prompt tokens.

Deliberate choices:

- Filesystem tools only by default. The CLI hands out the shell tool because a
  human is sitting at the permission prompt; an HTTP caller is not, and every
  server run is auto-approved. SA_SERVER_SHELL=1 opts back in for a box where
  that risk is understood.
- The key is compared with compare_digest, not ==, so the check doesn't leak
  timing. A missing SA_SERVER_API_KEY refuses to start rather than starting
  open: an unset env var should never mean "no auth".
- Runs are synchronous. The endpoint is a plain def, so FastAPI runs it on a
  worker thread and slow runs don't block the event loop. No queue, no job
  store; a demo box does not need them and pretending otherwise is more code
  to review.

Run with:
    SA_SERVER_API_KEY=... LLM_PROVIDER=hosted GROQ_API_KEY=... \
        uvicorn secret_agent.server:app --host 0.0.0.0 --port 8080
"""

import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent, AgentFailure
from .config import Config
from .permissions import default_permissions
from .tools.fs import FS_TOOLS
from .tools.registry import Registry

SERVER_KEY = os.environ.get("SA_SERVER_API_KEY", "")

app = FastAPI(title="secret-agent")


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)


class RunResponse(BaseModel):
    answer: str
    iterations: int
    tool_calls: int
    repair_rate: float
    prompt_tokens: int


def _check_key(x_api_key: str) -> None:
    if not SERVER_KEY:
        raise HTTPException(status_code=503, detail="server has no SA_SERVER_API_KEY set")
    if not secrets.compare_digest(x_api_key, SERVER_KEY):
        raise HTTPException(status_code=401, detail="bad api key")


def build_server_agent() -> Agent:
    cfg = Config.from_env()
    cfg.auto_approve = True  # nobody is at a prompt; the tool set is the guard
    tools = list(FS_TOOLS)
    if os.environ.get("SA_SERVER_SHELL", "") == "1":
        from .tools.shell import SHELL_TOOLS

        tools += list(SHELL_TOOLS)
    perms = default_permissions(auto_approve=True)
    return Agent(Registry(tools, permissions=perms), cfg=cfg)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest, x_api_key: str = Header(default="")) -> RunResponse:
    _check_key(x_api_key)
    agent = build_server_agent()
    try:
        result = agent.run(req.task)
    except AgentFailure as e:
        raise HTTPException(status_code=502, detail=str(e))
    return RunResponse(
        answer=result.answer,
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        repair_rate=result.repair_rate,
        prompt_tokens=sum(s.prompt_tokens for s in result.steps),
    )
