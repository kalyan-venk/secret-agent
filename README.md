# Secret-Agent

A small agent runtime I wrote from scratch, no LangChain, no LlamaIndex. It runs
a tool-calling loop against a local 8B model through Ollama, can search a
document corpus, and can talk to an external MCP server. Most of what's
interesting here is in the parts that broke and what the measurements said when
they disagreed with me.

## Run it

```
pip install -e ".[dev,vector]"
ollama pull llama3.1:8b && ollama pull nomic-embed-text

secret-agent --demo                        # same question, with and without retrieval
secret-agent --rag "when does Driftwood close a batch?"
python -m secret_agent.rag.eval --ablate   # the retrieval numbers
pytest                                     # 251 offline
pytest -m live -o addopts=""               # 9 against real ollama
```

Want a hosted model instead of local Ollama? `pip install -e ".[hosted]"`, set
`LLM_PROVIDER=hosted` and a `GROQ_API_KEY`, done. The provider sits behind a
single method, so the swap is 1 file and nothing else changes.

Serving the HTTP API under real concurrency? Run vLLM. Ollama answers one
request at a time, fine for local dev; vLLM batches many at once, which is what
an API endpoint needs. vLLM speaks the same OpenAI-compatible protocol, so it is
the hosted path with local defaults: `bash scripts/serve_vllm.sh` on a GPU box,
then `LLM_PROVIDER=vllm` points the agent at `http://localhost:8000/v1` with no
key. Override the target with `SA_HOSTED_BASE_URL` and the model with
`SA_HOSTED_MODEL`.

## Why a local 8B model

Ollama costs nothing per token, but the real reason is that small local models
write bad tool calls, and I wanted to build against the messy version. Fenced
JSON, single quotes, trailing commas, arguments double-encoded as a string,
invented key names, 2 calls when you asked for 1. My favourite one emits the
tool call and then narrates the result before the tool has even run. A frontier
model hides all of that behind constrained decoding, and then your parse and
retry code never gets tested.

Then I measured it and it went against me. On the 10 tasks in
`scripts/repair_rate.py`, `llama3.1:8b` and `llama3.2:3b` emitted clean JSON
every single time. Only `qwen2.5-coder:7b` needed repairs, on 70% of its calls,
because a model fine-tuned on code wraps anything that looks like code in
` ```json ` fences. So the messy-tool-call problem is real, but it tracks which
model you pick, not how small it is.

| model | temp | tool calls | repaired | rate |
|---|---|---|---|---|
| llama3.1:8b | 0.1 | 11 | 0 | 0% |
| llama3.2:3b | 0.1 | 20 | 0 | 0% |
| **qwen2.5-coder:7b** | 0.1 | 10 | **7** | **70%** |

This matters because the parser sits between the model and every measurement you
could make of the model. Benchmark llama against qwen on tool-call validity with
the repair turned off and qwen looks 70% worse. It isn't. That whole gap is my
parser, and it would read as a capability difference. So every repair is counted
and printed (`AgentRun.repair_rate`). I got burned by exactly this on an earlier
project, where a headline improvement turned out to be the eval harness stripping
markdown fences, not the models differing.

## The loop

The model asks for a tool, my code runs it, I feed the result back, and that
repeats until it stops asking. Everything else exists because that loop breaks in
4 ways:

| failure | what it looks like | what's done |
|---|---|---|
| runaway | model calls tools forever | hard cap at 15 iterations, then raises |
| malformed call | JSON that isn't JSON | repair ladder, bounded retries, then raises |
| tool throws | exception inside a tool | caught, handed back to the model as a readable error |
| context overflow | prompt exceeds the window | counted before every call, compacted if needed |

2 of those have a wrong fix that looks right. The loop stops on the absence of a
tool call, not on the presence of text, because small models narrate ("Sure, let
me check that file.") while also calling the tool, so stopping on text ends every
run on iteration 1. And running out of iterations raises instead of returning the
last message, because a run that hit the cap failed, and handing back its partial
output is how a bad answer gets mistaken for a real one.

## The sandbox, and how it broke

Every file path the model gives me goes through `safe_resolve()`: canonicalise,
then check it's inside the project root with `is_relative_to`. That catches `..`,
absolute paths, and symlinks pointing outward, which is the case a plain
string-prefix check misses.

An external review on 2026-07-25 broke this completely. I had put `python` on the
bash allowlist, so the model could run `python3 -c 'open("/tmp/pwned","w")...'`
and write straight outside the root. `shell=False` stops `/bin/sh`, it does not
stop you from handing the model a different interpreter. Worse, bash was never
calling `safe_resolve` on its arguments at all, so `cat /etc/passwd` and `cat
.env` both worked. The "two independent layers" I'd described were 1 layer, and
it had a hole.

The fix: no interpreters on the allowlist ever, every path-shaped bash argument
through the same `safe_resolve` the file tools use, read-only git only, `find
-exec` and `git --exec-path` refused, scrubbed environment, 30s timeout. The
lesson I kept is that an allowlist is only as safe as the least constrained
program on it, and deciding "is this binary safe" turned out to be as hard as the
thing the allowlist was supposed to avoid. It's defense in depth, not real
isolation. Real isolation needs the OS. The whole story is in `sandbox.py` and
`tools/shell.py`.

## Context management

The prompt grows every turn, and I guessed wrong about what happens when it
overflows. Ollama does not drop the oldest messages. It keeps the system prompt
and the most recent message and silently deletes the middle, and it returns HTTP
200 the whole time. In one probe 3,580 tokens of history became 53 with no error.

That's dangerous because the system prompt survives, so the model still sounds
like itself and follows instructions, it has just forgotten the task. There's no
signal downstream, so counting tokens before each call is the only defense. The
usual `chars/4` estimate undercounts JSON by 29%, which is the wrong direction
(you think you have room and you don't), so the estimator buckets by punctuation
density and errs high instead. Both findings come from scripts you can run
(`overflow_probe.py`, `calibrate_tokens.py`).

## Retrieval

The agent searches a corpus as a tool, not as a fixed preprocessing step, so it
decides whether to search, can reword the query, and can search twice for a
2-part question. The corpus is documentation for a data platform called Meridian
that does not exist. It's fake on purpose: with real docs you cannot tell
successful retrieval from the model already knowing the answer. `MER-4471`
appears nowhere on the internet, so if the model says it, it read it here.

Retrieval lands the right passage in the top 3 on 90% of 20 hand-labelled
questions, MRR 0.756. But the aggregate hides the honest number. When I split by
phrasing, questions worded like the document hit 1.00 at k=3, and questions
worded differently drop to 0.71, and I wrote both the documents and the questions,
so the easy half is me grading my own softballs. There are 3 vector stores
(`store_numpy.py` is 15 lines of dot-product and argsort, plus Chroma and
Qdrant), kept so a real vector DB stops looking like magic. A parity test pins all
3 to identical rankings, which is what catches the one real trap: Chroma returns a
distance and the others return a similarity, and converting is a subtraction that
silently inverts your results if you forget it.

## MCP

An MCP client connects to an external server over stdio and registers its tools
as ordinary tools in the registry. Because they go through the registry, every
guardrail the native tools already pass through governs them too, with no new
code in the loop: the same repair ladder, the same permission check, the same
path confinement runs before any call is forwarded. An MCP server can do nothing
`read_file` couldn't.

## Scope

Stops at retrieval plus a thin MCP client. No MCP server, no orchestration, no
Kubernetes. When a piece hit its "done" line, I stopped there instead of letting
it grow.
