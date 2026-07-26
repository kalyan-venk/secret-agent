"""System prompt construction for prompted tool-calling mode.

Prompt text is code here. It gets edited, it changes behaviour, and it needs
the same "why is it like this" treatment as anything else -- so the notes are
inline rather than in a doc nobody opens.

One thing worth knowing before touching this: the system prompt sits at the
front of every request forever. Editing it mid-conversation invalidates any
prompt cache from the first differing token onward, which on a hosted API is
the difference between paying full price and paying a tenth of it. Locally
Ollama's KV cache behaves the same way -- it re-processes the prompt. So:
build the system prompt once, at the top, and leave it alone.
"""

TOOL_PREAMBLE = """You have tools. To use one, reply with a single JSON object and nothing else:

{"name": "<tool_name>", "arguments": {"<arg>": "<value>"}}

Rules:
- One JSON object per reply. If you truly need two tools at once, reply with a JSON array of objects.
- Use double quotes. No trailing commas. No comments.
- Do not wrap the JSON in markdown fences.
- Only use tool names from the list below. Do not invent tools.
- When you have enough information, reply with your answer in plain prose and NO JSON.

Available tools:

{tools}"""


# Notes on what's in there and why:
#
# "and nothing else" -- llama3.1 will still add prose about half the time.
# The line measurably reduces it but does not stop it, which is fine, the
# parser handles prose. It's a cost/benefit thing: one sentence of prompt for
# fewer repairs.
#
# "Do not wrap the JSON in markdown fences" -- ignored roughly as often as
# it's obeyed. Kept anyway for the same reason. Worth being honest that a
# prompt instruction is a nudge, not a constraint; if you need a guarantee
# you need constrained decoding, which is what native mode gets you.
#
# "reply with your answer in plain prose and NO JSON" -- this one matters a
# lot. Without it the model keeps calling tools after it has the answer,
# because nothing has told it what "done" looks like. The iteration cap
# catches that, but hitting the cap is a failure, not a stop condition.


def build_system_prompt(base: str, tools_block: str) -> str:
    if not tools_block.strip():
        return base
    return base.rstrip() + "\n\n" + TOOL_PREAMBLE.replace("{tools}", tools_block)


DEFAULT_BASE = """You are a careful assistant working inside a project directory.
Prefer looking things up with your tools over guessing. If a tool returns an
error, read it and adjust -- do not repeat the identical call."""


# Fed back when the model emits something that looked like a call but wasn't
# parseable. Short on purpose: a long correction gives the model more surface
# to latch onto and it starts apologising instead of retrying.
RETRY_NUDGE = """Your last reply could not be parsed as a tool call.

{problem}

Reply with ONLY a JSON object of the form {{"name": ..., "arguments": {{...}}}}, or with your final answer in plain prose if you are done."""
