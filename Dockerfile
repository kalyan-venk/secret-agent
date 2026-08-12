# One image, run as two different services in docker-compose: the orchestrator
# (secret_agent.server) and the executor node (secret_agent.executor.service).
# They differ only in the command and the env, which is the point -- the same
# code runs the loop on one box and the sandboxed subprocess on another.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY secret_agent ./secret_agent

# [server] pulls in fastapi + uvicorn, which both the orchestrator and the
# executor need. git is here so the executor's read-only git subcommands work.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -e ".[server,hosted]"
