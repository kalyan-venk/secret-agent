# secret-agent

An agent runtime with no framework under it. Tool-calling loop, guardrails,
token-budgeted context management, and retrieval — built against a small local
model on purpose, because a small model emits messy output and messy output is
what forces the interesting code to exist.

No LangChain, no LlamaIndex. 4,100 lines of source, 1,700 of tests, 300 of
measurement scripts — and a large share of the source is comment, because the
reasoning behind a choice outlives the choice. The loop itself is 53 lines
including its error handling, and you can read the whole thing in an
afternoon, which is the point.

```
pip install -e ".[dev,vector]"
ollama pull llama3.1:8b && ollama pull nomic-embed-text

secret-agent --demo                            # retrieval on vs off
secret-agent --rag "when does Driftwood close a batch?"
python -m secret_agent.rag.eval --ablate       # the retrieval numbers
pytest                                          # 167 offline
pytest -m live -o addopts=""                    # 9 against real ollama
```

---

## On the commit history

This was built as a marathon — a single long push with
[Claude Code](https://claude.ai/code) as a pair, rather than an hour a night
over three weeks. So the timestamps cluster, and some phases have a dozen
commits while others have one. That's what the work actually looked like:
parsing took nine commits because it kept breaking in new ways, and the
permission layer took one because it didn't.

I'd rather the history show that than tidy it into a fake commit-a-day.

The `MISTAKES.md` file is committed for the same reason. Every error in it is
one I actually hit here, including three where I wrote a confident claim into
a docstring and the measurement later contradicted it.

---

## Why a local 8B model instead of an API

Two reasons, and the second is the real one.

**Practical:** this machine has a Claude subscription, not an API key. The CLI
authenticates over OAuth and there's no key to hand a Python process. Ollama
needs no key and costs nothing per token.

**The actual reason:** small local models produce *bad* tool calls — fenced
JSON, single quotes, trailing commas, arguments double-encoded as a string,
invented key names, tool names that don't exist, two calls when you asked for
one, and my favourite, a response that emits the tool call and then narrates
the result it expects *before the tool has run*. (That last one is real and
reproducible; the transcript is in `tests/test_agent.py`.)

A frontier model hides most of that behind constrained decoding. Building
against the messy one is what forces parse → validate → retry to be real
rather than aspirational.

**Caveat, since I measured it and it went against me:** on the ten-task set in
`scripts/repair_rate.py`, `llama3.1:8b` and `llama3.2:3b` emitted clean JSON
*every single time*. `qwen2.5-coder:7b` needed repair on 70%. The premise
holds, but the variable turned out to be which model, not how small — see
[Measuring the parser](#measuring-the-parser-not-just-writing-it).

The provider sits behind a one-method interface (`complete(messages, tools)`),
so swapping in the Anthropic SDK is one file, not a rewrite.

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
prose.** Small models narrate constantly — *"Sure, let me check that file."* —
while also emitting the call. Stopping when the model produces text ends every
run on iteration 1.

**Running out of iterations raises rather than returning the last text.** A
run that hit the cap failed; handing back its partial output lets a caller
mistake a failure for an answer, which is how a bad number ends up in a
metric. Same reasoning for exhausted parse retries.

---

## Measuring the parser, not just writing it

Every repair the parser applies is recorded on the call and counted, because
the parser sits between the model and every measurement you could take *of*
the model, silently improving its apparent output quality.

This comes from being burned. On an earlier project a headline "a headline improvement
improvement" turned out, on audit, to be mostly markdown-fence-stripping in
the eval harness rather than any model improvement — the parser was doing the
work and the model was getting the credit.

`scripts/repair_rate.py` runs the same ten tasks through the real loop across
several models:

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
7B *code-tuned* model wrapped 70% of its calls in ` ```json ` fences — because
that is what a model fine-tuned on code does with anything resembling code.

So: benchmark llama3.1:8b against qwen2.5-coder:7b on tool-call validity with
fence-stripping disabled, and qwen scores ~70% worse. It is not 70% worse.
That entire gap would be my parser, and it would look exactly like a
capability difference.

Which also means I have to walk back my own framing above. "A small local
model emits messy tool calls" is what I expected and it is not what these two
llama models did — the fixture suite in `tests/fixtures/malformed.py` covers
twenty failure shapes, and on this task set llama triggered none of them. The
repair ladder is insurance that mostly didn't get claimed. It got claimed hard
on exactly one model, which is the argument for having it *and* for always
printing the rate.

`AgentRun.repair_rate` and `parsing.STATS.summary()` exist for this.

---

## Guardrails

> **An external adversarial review on 2026-07-25 broke this section
> completely.** What follows is the corrected version; the failure is written
> up in `MISTAKES.md` #12–14 and is more instructive than the design. Short
> version: `python` was on the bash allowlist, so the sandbox was not a
> sandbox, and bash never called the path-confinement function at all.

Path confinement, then a permission layer, and they're independent — approval
is not authorisation to escape the project root.

Every model-supplied path goes through `safe_resolve()`: canonicalise, then
check containment with `is_relative_to`. That catches `..`, absolute paths,
and symlinks pointing outward (the case that defeats a string-prefix check,
since the *unresolved* path looks fine). `expanduser` is deliberately never
called, so `~` is a literal directory name.

Percent-encoded traversal is rejected too, but the honest answer to "how do
you stop `%2e%2e%2f`" is **"by never URL-decoding anything"** — the rejection
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
never called `safe_resolve` on its arguments at all — so the "independent
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
dangerous" — which was the entire argument for preferring an allowlist. `find`
has `-exec`. `git` has `--exec-path`. I now maintain a blocklist *inside* the
allowlist, which is exactly the smell I claimed to be avoiding.

**This is defense in depth, not isolation, and the difference is not
rhetorical.** Real isolation needs the OS — seatbelt, landlock, a container.
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
from the front — it keeps the system prompt *and* the most recent message and
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
compression — it wins on what it keeps, and only sometimes. A fact stated
early and never repeated is destroyed by truncation and *may* survive
summarization; "may", because the summary is model output, and if it writes
"discussed the config file" the filename is just as gone.

Tool results get trimmed head-and-tail (the tail matters — the summary line
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
exist. Every fact in it is invented — service names, error codes, thresholds.

That's not whimsy. If you index real documentation you cannot tell successful
retrieval apart from the model already knowing the answer, and the claim "it
answered from the corpus" becomes unfalsifiable. `MER-4471` appears nowhere on
the internet. If the model says it, the model read it.

`secret-agent --demo` asks the same question with and without `search_docs`:

> **Without:** *"To find out what error code Meridian returns... I would use
> the Meridian API documentation. Let me check the documentation... According
> to the documentation, attempting to delete a dataset that is—"*
>
> **With:** *"The error code is MER-4471: deletion blocked by active legal
> hold. This error is not retryable and requires the legal hold to be released
> before deletion can proceed."*

**Retrieval is a tool the agent calls, not a preprocessing step.** A
preprocess step retrieves once, on the original wording, for every question
including "thanks" — and when that search comes back wrong there is no second
attempt. As a tool, the model decides whether to search, reformulates the
query, and can search twice for a two-part question.

### The numbers

20 hand-labelled query→passage pairs, `python -m secret_agent.rag.eval`:

| | k=1 | k=3 | k=5 |
|---|---|---|---|
| precision | 0.60 | 0.38 | 0.23 |
| recall | 0.50 | 0.83 | 0.83 |
| **hit rate** | **0.60** | **0.90** | **0.90** |

MRR 0.756. **hit@k is the number that matters** — the model needs the fact
once, not every copy of it. Precision@5 is capped near 0.2–0.4 by how few
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
`"bounce a stuck job without seeing customer records"` → *operator* — both
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
2000 wins hit@5, and 2000 edges MRR by 0.002 — those are one- and two-query
differences on a 20-query set. The defensible reading is "600 is a good
middle and 200 is genuinely worse", not "600 is optimal".

Second ablation — nomic-embed-text's asymmetric `search_document:` /
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
20-question set, over a corpus I wrote myself — comfortably inside the noise,
and "the model card is wrong" needs more than one query. (It was a two-query
gap until a review corrected two gold labels, which cuts in favour of the
"it's noise" reading: had I acted on the original result, I would now be
defending a default chosen on evidence that had since halved.) The documented
behaviour is the better prior when the evidence is this thin. It's logged as
an open question in `LEARNING-STATUS.md` rather than quietly resolved in
whichever direction happened to win one run.

The ablation earned its place by contradicting me.

---

## Two vector stores

`store_numpy.py` is fifteen lines: a normalised matrix, a dot product, an
argsort. `store_chroma.py` is the same interface over Chroma. Both are kept.

The brute-force one exists so that a vector database stops being magic —
persistence, an HNSW index, metadata filters and concurrency are scaling
concerns layered on those fifteen lines, not a different idea. At 38 chunks
Chroma buys nothing except the metadata filter; at ten million vectors the
ANN index is the whole game, and you pay for it by *sometimes getting the
wrong neighbours*.

They return identical rankings and scores to four decimals, which is the test
that the interface isn't leaking.

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
  llm.py            provider interface + Ollama
  prompts.py        system prompt construction
  cli.py
  tools/            base, registry, fs, shell
  rag/              chunking, embed, two stores, retrieve, eval
corpus/             fictional docs for Meridian
scripts/            calibrate_tokens.py, overflow_probe.py
tests/              167 offline, 9 live
```

## Scope

Stops at RAG, on purpose. No MCP server, no distributed tracing, no
Kubernetes, no multi-agent orchestration — those are a different project. When
a phase started growing past its "done when" line I treated that as the phase
being finished rather than needing expansion.

## Reading the docs in here

- `DECISIONS.md` — choices made and, more usefully, what was rejected and why
- `MISTAKES.md` — errors hit during the build, with the mechanism that let
  each one survive
- `LEARNING-STATUS.md` — what I can defend cold, what I can't yet, and open
  questions
- `corpus/README.md` — why the corpus is fictional and where the eval is weak
