"""The orchestrator's side of the executor split.

The `bash` tool holds one of these when SA_EXECUTOR_URL is set. It POSTs the
command to the executor node and turns the JSON reply back into exactly what an
in-process run would have produced: a returned string on success, a raised
ToolError on a refusal. The registry catches the ToolError and hands it to the
model as a tool result, same as it always did, so the model cannot tell whether
the command ran here or on another box -- which is the point.

Transport errors (the executor is down, or unreachable) are raised as ToolError
too, deliberately. There is no fall back to running the command locally: the
whole reason the executor is a separate node is that this process should not run
untrusted commands, and quietly doing so because the network hiccuped would
undo that. A dead executor means bash fails, loudly, and the model is told so.
"""

from __future__ import annotations

import httpx

from ..tools.base import ToolError


class ExecutorClient:
    def __init__(self, url: str, key: str | None = None, timeout: float = 35.0,
                 client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.key = key or ""
        self.timeout = timeout
        # Injectable so a test can point this at the FastAPI app through an
        # in-memory ASGI transport instead of a real socket.
        self._client = client

    def execute(self, command: str) -> str:
        headers = {"x-executor-key": self.key} if self.key else {}
        target = f"{self.url}/execute"
        try:
            if self._client is not None:
                r = self._client.post(target, json={"command": command}, headers=headers)
            else:
                r = httpx.post(target, json={"command": command}, headers=headers,
                               timeout=self.timeout)
        except httpx.HTTPError as e:
            raise ToolError(
                f"the executor service at {self.url} is unreachable ({e}). "
                "bash runs on a separate node and there is no local fallback."
            ) from e

        if r.status_code != 200:
            raise ToolError(
                f"executor service returned {r.status_code}: {r.text[:200]}"
            )

        data = r.json()
        if not data.get("ok", False):
            # A refused command (allowlist, path escape, credential file, ...).
            # Re-raised as ToolError so it reaches the model identically to an
            # in-process refusal.
            raise ToolError(data.get("error") or "executor refused the command")
        return data.get("output", "")
