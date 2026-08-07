# secret-agent

An agent runtime with no framework under it. Tool-calling loop, guardrails,
token-budgeted context management, and retrieval, built against a small local
model on purpose, because a small model emits messy output and messy output is
what forces the interesting code to exist.

No LangChain, no LlamaIndex. 5,263 lines of source, 2,368 of tests, 637 of
measurement scripts, and a large share of the source is comment, because the
reasoning behind a choice outlives the choice. The loop itself is 53 lines
including its error handling, and you can read the whole thing in an
afternoon, which is the point.

```
pip install -e ".[dev,vector]"
ollama pull llama3.1:8b && ollama pull nomic-embed-text

secret-agent --demo                            # retrieval on vs off
secret-agent --rag "when does Driftwood close a batch?"
python -m secret_agent.rag.eval --ablate       # the retrieval numbers
pytest                                          # 251 offline
pytest -m live -o addopts=""                    # 9 against real ollama
pytest -m "mcp or qdrant" -o addopts=""         # 6 for the MCP + Qdrant extensions

# optional: hosted provider instead of local Ollama for the chat model
pip install -e ".[hosted]"
export LLM_PROVIDER=hosted GROQ_API_KEY=...     # or copy .env.local.example -> .env.local
secret-agent "when does Driftwood close a batch?"
python scripts/hosted_eval.py                   # retrieval eval + one full agent task, hosted
```

---

## On the commit history

This was built as a marathon, one long push rather than an hour a night over
three weeks. So the timestamps cluster, and some phases have a dozen
commits while others have one. That's what the work actually looked like:
parsing took nine commits because it kept breaking in new ways, and the
permission layer took one because it didn't.

I'd rather the history show that than tidy it into a fake commit-a-day.

Two things in the source carry the same idea. Four docstrings stated a
confident claim that a later measurement contradicted, and each is walked back
in the file where it lives instead of quietly edited to look right (the parser
section below has the clearest one). And when the build was finished I ran a
hostile external review over it, with instructions that "strong work, minor
nits" would count as a failed review. It found a total sandbox escape. The
corrected design and the exploit that broke it are in the Guardrails section.

---

## Why a local 8B model instead of an API

Two reasons, and the second is the real one.

**Practical:** there's no API key on this machine to hand a Python process.
Ollama needs no key and costs nothing per token.

**The actual reason:** small local models produce *bad* tool calls: fenced
JSON, single quotes, trailing commas, arguments double-encoded as a string,
invented key names, tool names that don't exist, two calls when you asked for
one, and my favourite, a response that emits the tool call and then narrates
the result it expects *before the tool has run*. (That last one is real and
reproducible. The transcript is in `tests/test_agent.py`.)

A frontier model hides most of that behind constrained decoding. Building
against the messy one is what forces parse → validate → retry to be real
rather than aspirational.

**Caveat, since I measured it and it went against me:** on the ten-task set in
`scripts/repair_rate.py`, `llama3.1:8b` and `llama3.2:3b` emitted clean JSON
*every single time*. `qwen2.5-coder:7b` needed repair on 70%. The premise
holds, but the variable turned out to be which model, not how small. See
[Measuring the parser](#measuring-the-parser-not-just-writing-it).

The provider sits behind a one-method interface (`complete(messages, tools)`),
so swapping in the Anthropic SDK is one file, not a rewrite. Same interface,
different reason to use it: below is what swapping to a hosted provider
actually looked like.

---

## A hosted provider, behind the same interface

Everything above stays true by default: `LLM_PROVIDER` unset (or `ollama`)
gets you the exact `OllamaClient` this was always built against, nothing
changes underneath a run that doesn't ask for anything different.

Set `LLM_PROVIDER=hosted` and the same `complete(messages, tools)` interface
is served by `HostedClient` in `llm.py`, talking to any OpenAI-compatible
`chat.completions` endpoint through the `openai` SDK. Configured by env vars,
optionally read from a git-ignored `.env.local` (see `.env.local.example`):

| var | default | purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `hosted` |
| `GROQ_API_KEY` | none | the key, if using the Groq default |
| `SA_HOSTED_API_KEY` | none | explicit key, wins over `GROQ_API_KEY` |
| `SA_HOSTED_BASE_URL` | `https://api.groq.com/openai/v1` | any OpenAI-compatible base URL |
| `SA_HOSTED_MODEL` | `llama-3.1-8b-instant` | model id at that endpoint |

Groq is the default because it's free with no credit card and speaks the
OpenAI protocol. Swapping to Gemini's OpenAI-compatible endpoint or
OpenRouter is `SA_HOSTED_BASE_URL` + `SA_HOSTED_MODEL` + a different key,
nothing in `llm.py` changes; `.env.local.example` has both as commented-out
examples. `build_llm_client(cfg)` is the one place that picks a provider, so
`agent.py` and `cli.py` never name a concrete client class.

`openai` is an optional extra (`pip install -e ".[hosted]"`), not a base
dependency, so `pip install -e .` for the offline Ollama path stays exactly
as light as before.

**What moves and what doesn't.** Retrieval embeddings (`nomic-embed-text`)
have no free hosted equivalent wired up here, so `search_docs` still calls
local Ollama even with `LLM_PROVIDER=hosted` -- only the reasoning/tool-calling
loop moves to the hosted model. That means the RAG hit@k/MRR numbers below
are provider-independent by construction, not something the hosted run is
expected to change.

**Numbers**, `scripts/hosted_eval.py` (retrieval eval + one full agent task
through the hosted provider), run for real with a GROQ_API_KEY:

| | local (Ollama, `llama3.1:8b`) | hosted (Groq, `llama-3.1-8b-instant`) |
|---|---|---|
| RAG hit@3 / MRR / low-overlap hit@3 | 0.90 / 0.756 / 0.71 | identical -- retrieval stays on local Ollama regardless of `LLM_PROVIDER`, see note below |
| agent task -- search the docs for Meridian's legal-hold error code | -- | done in 2 iterations, 1 tool call, 0% repair rate, 1473 prompt tokens total |
| agent task -- answer | -- | "Meridian returns the error code `MER-4471` when trying to delete a dataset under legal hold, and this error is not retryable." |

The RAG row has one value, not two, on purpose: `search_docs` calls local
`nomic-embed-text` no matter which chat provider is configured, so there is
no separate "hosted" retrieval number to report -- see "What moves and what
doesn't" above.

Also true and worth saying plainly: re-running `python -m secret_agent.rag.eval
--ablate` today (the same command `scripts/hosted_eval.py` calls as its first
step) prints hit@1 0.65 / MRR 0.767, not the 0.60 / 0.756 committed above.
hit@3 (0.90) still matches. Corpus, eval code and gold labels are all
unmodified since those numbers were committed, so this is drift, not a
regression from this fix -- most likely the embed cache
(`.embed_cache/nomic-embed-text.json`) no longer matches whatever produced
the committed run. Flagging it here rather than touching the committed
numbers, since editing `corpus/` or the gold labels is outside what this fix
touches.

---

## The loop

In one sentence: **the model asks for a tool, my code runs it, I feed the
result back, and that repeats until it stops asking.**

Everything else exists because that loop can break in exactly four ways:

| failure | what it looks like | what's done about it |
|---|---|---|
| runaway | model calls tools forever | hard cap at 15 iterations, then raises |
| malformed call | JSON that isn't JSON | repair ladder + bounded retry, then raises |
| tool blows up | exception inside a tool | caught, returned to the model as a readable error |
| context overflow | payload exceeds the window | counted *before* every call, compacted if needed |

Two decisions in there worth stating, because both have a wrong answer that
looks right:

**The stop condition keys off the absence of a tool call, not the presence of
prose.** Small models narrate constantly (*"Sure, let me check that file."*)
while also emitting the call. Stopping when the model produces text ends every
run on iteration 1.

**Running out of iterations raises rather than returning the last text.** A
run that hit the cap failed. Handing back its partial output lets a caller
mistake a failure for an answer, which is how a bad number ends up in a
metric. Same reasoning for exhausted parse retries.

---

## Measuring the parser, not just writing it

Every repair the parser applies is recorded on the call and counted, because
the parser sits between the model and every measurement you could take *of*
the model, silently improving its apparent output quality.

This comes from being burned. On an earlier research project, a headline
improvement turned out on audit to be substantially an artifact of the eval
harness stripping markdown fences, rather than the models differing. The
parser was doing the work and the model was getting the credit.

`scripts/repair_rate.py` runs the same ten tasks through the real loop across
several models. **This is one sampled run, not a fixed measurement**. the
completion counts come from live generation and will differ if you re-run it.
The stable claim is the *direction*: llama emits clean JSON, the code-tuned
model fences almost everything.

| model | temp | completions | w/ calls | repaired | rate |
|---|---|---|---|---|---|
| llama3.1:8b | 0.1 | 21 | 11 | 0 | 0.0% |
| llama3.1:8b | 0.9 | 22 | 12 | 0 | 0.0% |
| llama3.2:3b | 0.1 | 29 | 20 | 0 | 0.0% |
| llama3.2:3b | 0.9 | 21 | 11 | 0 | 0.0% |
| **qwen2.5-coder:7b** | 0.1 | 20 | 10 | **7** | **70.0%** |

Which is not what I expected, and is the most useful thing I measured here.

**Repair rate is a property of the specific model, not of model size.** Both
llama models emitted clean JSON on every tool call at both temperatures. The
7B *code-tuned* model wrapped 70% of its calls in ` ```json ` fences, because
that is what a model fine-tuned on code does with anything resembling code.

So: benchmark llama3.1:8b against qwen2.5-coder:7b on tool-call validity with
fence-stripping disabled, and qwen scores ~70% worse. It is not 70% worse.
That entire gap would be my parser, and it would look exactly like a
capability difference.

One run per cell is thin, and the 0% cells are the weaker half of the
evidence: "0 out of 11" is consistent with a low rate, not only with zero.
The 70% cell is the reliable one: seven fenced calls out of ten is not a
sampling accident.

Which also means I have to walk back my own framing above. "A small local
model emits messy tool calls" is what I expected and it is not what these two
llama models did. The fixture suite in `tests/fixtures/malformed.py` covers
twenty failure shapes, and on this task set llama triggered none of them. The
repair ladder is insurance that mostly didn't get claimed. It got claimed hard
on exactly one model, which is the argument for having it *and* for always
printing the rate.

`AgentRun.repair_rate` and `parsing.STATS.summary()` exist for this.

---

## Guardrails

> **An external adversarial review on 2026-07-25 broke this section
> completely.** What follows is the corrected version, and the failure is more
> instructive than the design. Short version: `python` was on the bash
> allowlist, so the sandbox confined nothing, and bash never called the
> path-confinement function at all.

Path confinement, then a permission layer, and they're independent: approval
is not authorisation to escape the project root.

Every model-supplied path goes through `safe_resolve()`: canonicalise, then
check containment with `is_relative_to`. That catches `..`, absolute paths,
and symlinks pointing outward (the case that defeats a string-prefix check,
since the *unresolved* path looks fine). `expanduser` is deliberately never
called, so `~` is a literal directory name.

Percent-encoded traversal is rejected too, but the honest answer to "how do
you stop `%2e%2e%2f`" is **"by never URL-decoding anything"**. The rejection
is a tripwire, not the defense. What *isn't* handled: TOCTOU between resolve
and open, and hardlinks. Both are written down in `sandbox.py` rather than
quietly omitted.

`bash` is a different category from `write_file` even though both mutate.
`write_file`'s blast radius is one resolved path you can inspect. `bash` hands
off to a process whose reach is the whole filesystem and network, and no
argument inspection bounds that.

I wrote that paragraph, and then put `python` on the allowlist anyway. The
review's exploit:

```
python3 -c 'open("/tmp/pwned","w").write("OWNED")'   → wrote outside the root
cat /etc/passwd                                      → read outside the root
cat .env                                             → read a credential file
```

`shell=False` protects you from `/bin/sh`. It does not protect you from
handing the model a **different** shell, and an interpreter is one. And bash
never called `safe_resolve` on its arguments at all, so the "independent
layers" claim two paragraphs up was simply false for this tool. Approving one
bash call escaped the root.

What defends it now: no interpreters on the allowlist ever, **every
path-shaped argument through the same `safe_resolve` the file tools use**,
`looks_secret` on resolved arguments, flags like `find -exec` and
`git --exec-path` refused, git limited to read-only subcommands, `shell=False`,
scrubbed environment, 30s timeout.

The lesson worth keeping is not "I forgot about python". It's that **an
allowlist is only as strong as the least constrained program on it**, and
"is this binary safe" turns out to be about as hard as "is this binary
dangerous", which was the entire argument for preferring an allowlist. `find`
has `-exec`. `git` has `--exec-path`. I now maintain a blocklist *inside* the
allowlist, which is exactly the smell I claimed to be avoiding.

**This is defense in depth, not isolation, and the difference is not
rhetorical.** Real isolation needs the OS: seatbelt, landlock, a container.
Env scrubbing likewise: it stops env-var inheritance, but `HOME` is still
forwarded and `~/.aws/credentials` is a file, not a variable.

---

## Context management

The window is a budget and the payload grows every turn. What happens when
you exceed it depends on the provider, and I guessed wrong before measuring:

```
num_ctx   prompt_eval   system   first user   last user
   8192          3580     kept         kept        kept
   1024            53     kept      DROPPED        kept
    256            53     kept      DROPPED        kept
```

All three responses were HTTP 200 with no error field. Ollama doesn't drop
from the front. It keeps the system prompt *and* the most recent message and
silently discards the middle. 3,580 tokens became 53.

The system prompt surviving is what makes it dangerous: the model still
behaves like itself, still follows instructions, still sounds fine. It has
just forgotten the middle of the task. There is no downstream signal, so
counting before you send is the only defense. (`scripts/overflow_probe.py`.)

**Two strategies, so they can be compared.** On a 2,308-token history:

| strategy | before | after | saved | cost |
|---|---|---|---|---|
| truncate | 2308 | 540 | 1768 | 0 ms |
| summarize | 2308 | 624 | 1684 | 8537 ms |

Summarization is slower *and* saved fewer tokens. It doesn't win on
compression. It wins on what it keeps, and only sometimes. A fact stated
early and never repeated is destroyed by truncation and *may* survive
summarization. "May", because the summary is model output, and if it writes
"discussed the config file" the filename is just as gone.

Tool results get trimmed head-and-tail (the tail matters: the summary line
and the last error live there) with the full output spilled to `.spill/` and
the path named in the marker, so the model can `read_file` it back.

**Token counting.** `chars/4` is the usual rule of thumb and it's wrong in the
dangerous direction on exactly the content an agent history is full of:

| sample | punctuation | real tokens | chars/4 | error |
|---|---|---|---|---|
| prose | 1.1% | 78 | 91 | +16.7% |
| code | 4.0% | 94 | 99 | +5.3% |
| tool output | 4.9% | 479 | 461 | **−3.8%** |
| JSON | 34.8% | 96 | 68 | **−29.2%** |

Undercounting means you think you have room and you don't. The estimator
buckets by punctuation density and is tuned to err high.
(`scripts/calibrate_tokens.py`.) The per-message chat-template overhead is 10
tokens, not the 4 I'd assumed from memory of an OpenAI figure.

---

## Retrieval, and whether it helped

The corpus is documentation for **Meridian**, a data platform that does not
exist. Every fact in it is invented: service names, error codes, thresholds.

That's not whimsy. If you index real documentation you cannot tell successful
retrieval apart from the model already knowing the answer, and the claim "it
answered from the corpus" becomes unfalsifiable. `MER-4471` appears nowhere on
the internet. If the model says it, the model read it.

`secret-agent --demo` asks the same question with and without `search_docs`:

> **Without:** *"To find out what error code Meridian returns... I would use
> the Meridian API documentation. Let me check the documentation... According
> to the documentation, attempting to delete a dataset that is..."*
>
> **With:** *"The error code is MER-4471: deletion blocked by active legal
> hold. This error is not retryable and requires the legal hold to be released
> before deletion can proceed."*

**Retrieval is a tool the agent calls, not a preprocessing step.** A
preprocess step retrieves once, on the original wording, for every question
including "thanks", and when that search comes back wrong there is no second
attempt. As a tool, the model decides whether to search, reformulates the
query, and can search twice for a two-part question.

### The numbers

20 hand-labelled query→passage pairs, `python -m secret_agent.rag.eval`:

| | k=1 | k=3 | k=5 |
|---|---|---|---|
| precision | 0.60 | 0.38 | 0.23 |
| recall | 0.50 | 0.83 | 0.83 |
| **hit rate** | **0.60** | **0.90** | **0.90** |

MRR 0.756. **hit@k is the number that matters**: the model needs the fact
once, not every copy of it. Precision@5 is capped near 0.2-0.4 by how few
chunks contain any given fact, so don't read 0.23 as "77% wrong".

**And the number I'd actually lead with:**

| questions phrased… | n | hit@1 | hit@3 | hit@5 |
|---|---|---|---|---|
| in the document's own words | 13 | 0.69 | **1.00** | 1.00 |
| in someone else's words | 7 | 0.43 | **0.71** | 0.71 |

I wrote both the documents and the questions. Where the wording matches,
retrieval looks perfect. Where it doesn't, hit@3 falls from 1.00 to 0.71.
The second row is the honest one, and reporting only the aggregate 0.90 would
have hidden it. The two remaining misses are `"bad row"` → *poison record* and
`"bounce a stuck job without seeing customer records"` → *operator*, both
requiring a hop from a description to a term that embedding similarity
doesn't make.

### Ablations

Chunk size, overlap held at size/6:

| size | chunks | hit@1 | hit@3 | hit@5 | R@3 | MRR |
|---|---|---|---|---|---|---|
| 200 | 105 | 0.45 | 0.65 | 0.75 | 0.65 | 0.593 |
| 300 | 71 | 0.60 | 0.85 | 0.90 | 0.77 | 0.741 |
| **600** | **38** | 0.60 | **0.90** | 0.90 | **0.83** | 0.756 |
| 1200 | 18 | **0.65** | 0.80 | 0.85 | 0.70 | 0.744 |
| 2000 | 10 | 0.60 | 0.85 | **1.00** | 0.78 | **0.758** |

600 wins hit@3 and recall@3, and 200 is clearly worse. But 1200 wins hit@1,
2000 wins hit@5, and 2000 edges MRR by 0.002, those are one- and two-query
differences on a 20-query set. The defensible reading is "600 is a good
middle and 200 is genuinely worse", not "600 is optimal".

Second ablation, nomic-embed-text's asymmetric `search_document:` /
`search_query:` prefixes:

| variant | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| correct (doc/query) | 0.60 | 0.90 | 0.90 | 0.756 |
| none at all | **0.65** | **0.95** | **0.95** | **0.792** |
| same prefix both sides | 0.55 | 0.95 | 0.95 | 0.733 |

I had written a paragraph in `embed.py` asserting that skipping the prefixes
was measurably worse. Then I measured it, and no-prefix scored better on
every metric.

I did **not** change the default. That gap is 18 queries versus 19, on a
20-question set, over a corpus I wrote myself, comfortably inside the noise,
and "the model card is wrong" needs more than one query. (It was a two-query
gap until a review corrected two gold labels, which cuts in favour of the
"it's noise" reading: had I acted on the original result, I would now be
defending a default chosen on evidence that had since halved.) The documented
behaviour is the better prior when the evidence is this thin. It's logged as
an open question rather than quietly resolved in whichever direction happened
to win one run.

The ablation earned its place by contradicting me.

---

## Three vector stores

`store_numpy.py` is fifteen lines: a normalised matrix, a dot product, an
argsort. `store_chroma.py` and `store_qdrant.py` put the same interface over
Chroma and Qdrant. All three are kept.

The brute-force one exists so that a vector database stops being magic:
persistence, an HNSW index, metadata filters and concurrency are scaling
concerns layered on those fifteen lines, not a different idea. At 38 chunks a
real store buys nothing except the metadata filter. At ten million vectors the
ANN index is the whole game, and you pay for it by *sometimes getting the wrong
neighbours*.

The one thing the interface hides is a sign. Chroma returns a distance, Qdrant
and numpy return a similarity, and converting is a subtraction that inverts the
ranking with no error if you forget it. A parity test asserts all three return
identical rankings and scores to four decimals, which is what catches that.

## External tools over MCP

An MCP client (`mcp/client.py`, `mcp/adapter.py`) connects to an external MCP
server over stdio and registers each of its tools as an ordinary tool in the
Registry. Routing them through the Registry means every guardrail the native
tools already pass through governs the external ones with no new code in the
loop: the same parse-and-repair ladder, the same permission check before
execution, the same path confinement.

Confinement runs before the call is forwarded. `make_mcp_tool` walks every
string in the argument object and resolves it against the project root using
the same over-inclusive check bash uses, so an out-of-root or credential-named
path is refused here and the server never sees it. An MCP server can do nothing
`read_file` could not. Tool names are namespaced `mcp__{label}__{tool}`, both
to avoid colliding with a native `read_file` and to keep the permission key
unambiguous. Nothing over MCP defaults to allow.

---

## Layout

```
secret_agent/
  agent.py          the loop. read this first
  parsing.py        the repair ladder + instrumentation
  context.py        token budget, truncate vs summarize, spill
  sandbox.py        path confinement
  permissions.py    allow / ask / deny
  conversation.py   message history
  llm.py            provider interface + Ollama + hosted (OpenAI-compatible)
  prompts.py        system prompt construction
  cli.py
  tools/            base, registry, fs, shell
  rag/              chunking, embed, three stores, retrieve, eval
  mcp/              stdio client + adapter onto the Tool interface
corpus/             fictional docs for Meridian
scripts/            calibrate_tokens.py, overflow_probe.py, hosted_eval.py
tests/              251 offline, 9 live, 6 mcp/qdrant
```

## Scope

Stops at RAG plus a thin MCP client, on purpose. No MCP server, no distributed
tracing, no Kubernetes, no multi-agent orchestration: those are a different
project. When a phase started growing past its "done when" line I treated that
as the phase being finished rather than needing expansion.

## Reading the docs in here

The reasoning and the errors caught during the build live in the source
itself, next to the code they explain. `sandbox.py` and `tools/shell.py` carry
the security design and the escape that broke it. `context.py` carries the
token-counting and overflow findings. `corpus/overview-index.txt` explains why
the corpus is fictional and where the eval is weak.
