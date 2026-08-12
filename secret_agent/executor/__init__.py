"""Out-of-process command execution for the agent.

`core.run_command` is the execution path; `service.py` serves it over HTTP;
`client.py` is what the orchestrator's bash tool calls to reach it. The service
is an optional extra (fastapi/uvicorn); importing this package does not pull it
in, same as server.py.
"""

from .client import ExecutorClient
from .core import run_command

__all__ = ["ExecutorClient", "run_command"]
