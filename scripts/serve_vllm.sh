#!/usr/bin/env bash
# Serve an OpenAI-compatible endpoint with vLLM, the backend the agent's HTTP
# API uses under real concurrency (vLLM batches many requests at once, unlike
# Ollama's one-at-a-time local path). Needs a GPU. Point the agent at it with
# LLM_PROVIDER=vllm (defaults to http://localhost:8000/v1, no key).
set -euo pipefail
MODEL="${SA_VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${SA_VLLM_PORT:-8000}"
exec python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT"
