"""Getting tool calls out of whatever the model actually said.

This file is the reason the project uses a small local model. Ask llama3.1:8b
for one JSON tool call and over a few hundred turns you will get, verbatim:

    {"name": "read_file", "arguments": {"path": "a.txt"}}      <- the good day
    ```json\n{...}\n```                                        <- fenced
    Sure! Here's the call:\n{...}\nLet me know if that works.   <- prose both sides
    {'name': 'read_file', 'arguments': {'path': 'a.txt'}}       <- python quotes
    {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"} <- args as a string
    {"tool": "read_file", "tool_input": {...}}                  <- invented key names
    {"name": "read_file", "arguments": {"path": "a.txt",}}      <- trailing comma
    {"name": "reed_file", ...}                                  <- name doesn't exist
    {"name": "read_file", "arguments": {}}                      <- required arg missing
    {...}\n{...}                                                <- two calls, no array

Frontier models hide almost all of this behind constrained decoding. Building
against the messy one is what forces the repair ladder below to exist.

### The measurement point

Every repair is recorded on the ToolCall and counted in STATS. That's
deliberate and it is the most important thing in this file.

I have been burned by exactly this before: on an earlier research project a
headline improvement turned out, on audit, to be substantially markdown-fence
stripping in the eval harness rather than a real difference between models.
The parser was silently doing the work and taking none of the credit.

So: if you ever compare two models with this runtime in the loop, print
STATS.summary() alongside the result. A model that needed repair on 40% of
turns and one that needed it on 2% did not have the same run, and the
difference will look like a capability difference if you don't look.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .conversation import new_call_id

# instrumentation


class ParseStats:
    def __init__(self):
        self.completions = 0
        self.completions_with_calls = 0
        self.completions_needing_repair = 0
        self.calls_found = 0
        self.repairs: dict[str, int] = {}
        self.unparseable = 0

    def note_repair(self, kind: str) -> None:
        self.repairs[kind] = self.repairs.get(kind, 0) + 1

    def reset(self) -> None:
        self.__init__()

    def summary(self) -> str:
        if not self.completions:
            return "parse stats: nothing parsed yet"
        pct = 100.0 * self.completions_needing_repair / max(1, self.completions_with_calls)
        lines = [
            f"parse stats: {self.completions} completions, "
            f"{self.completions_with_calls} contained tool calls, "
            f"{self.calls_found} calls extracted",
            f"  repaired: {self.completions_needing_repair} "
            f"({pct:.1f}% of tool-call completions were NOT clean JSON)",
        ]
        for k, v in sorted(self.repairs.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k}: {v}")
        if self.unparseable:
            lines.append(f"  gave up on: {self.unparseable}")
        return "\n".join(lines)


STATS = ParseStats()


# types


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=new_call_id)
    raw: str = ""
    repairs: list[str] = field(default_factory=list)

    @property
    def was_clean(self) -> bool:
        return not self.repairs


@dataclass
class ParseResult:
    calls: list[ToolCall] = field(default_factory=list)
    # prose with the tool-call blocks removed. Kept because when the model
    # both talks and calls a tool, the talking is often the useful part.
    text: str = ""
    # things that looked like calls but weren't usable; these get fed back
    problems: list[str] = field(default_factory=list)

    # Prose split around the calls, rather than merged into one blob. These
    # exist because the two halves mean opposite things and the agent needs
    # to treat them differently:
    #
    #   prose_before  reasoning. "I'll need to look at the config first."
    #   prose_after   narration of a result the tool has not returned yet.
    #                 "The echo tool returned: pineapple" -- written BEFORE
    #                 echo ran. See Agent.record_assistant_turn.
    prose_before: str = ""
    prose_after: str = ""

    def __bool__(self) -> bool:
        return bool(self.calls)


# fences

_FENCE = re.compile(
    r"```[ \t]*(?:json|JSON|tool_call|python)?[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def strip_fences(text: str) -> tuple[str, bool]:
    """Pull content out of ``` blocks and drop the fences.

    Returns (text, did_anything). The bool is not decoration -- it feeds
    STATS, which is the entire point of the module docstring.

    Unterminated fences are a real case: the model opens ```json, emits the
    object, and then hits its token limit before closing. Handled after the
    regex pass.
    """
    hit = False

    def _sub(m):
        nonlocal hit
        hit = True
        return "\n" + m.group(1) + "\n"

    out = _FENCE.sub(_sub, text)

    if "```" in out:
        # unterminated. Keep everything after the last opener.
        head, _, tail = out.rpartition("```")
        # strip a language tag if the model wrote one
        tail = re.sub(r"^[ \t]*(json|JSON|tool_call|python)[ \t]*\r?\n", "", tail)
        if tail.strip():
            out = head + "\n" + tail
            hit = True
        else:
            out = head
            hit = True

    return out, hit


# finding balanced JSON inside prose


def balanced_spans(text: str) -> list[tuple[int, int]]:
    """Every balanced {...} / [...] region at the top level.

    Regex can't do this -- nested braces and braces inside strings both break
    it, and I wrote the regex version first and watched it truncate an object
    at the first inner `}`. A depth counter that knows about string state is
    about fifteen lines and is actually correct.

    Note `depth == 0` skips quote tracking entirely. That is not an
    optimisation, it's a bug fix. Model prose contains apostrophes -- "I'll
    need to look at the file" -- and the first version treated that `'` as an
    opening quote, stayed in string mode for the rest of the completion, and
    found zero tool calls in a response that plainly contained one. Outside
    braces there is nothing to protect, so don't try.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_str = False
    quote = ""
    esc = False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue

        if depth > 0 and ch in "\"'":
            in_str = True
            quote = ch
            continue

        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, i + 1))
                    start = -1

    # unclosed trailing object -- model ran out of tokens mid-emit. Take the
    # rest and let the repair ladder try to close it.
    if depth > 0 and start >= 0:
        spans.append((start, len(text)))

    return spans


# the repair ladder

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def loads_forgiving(raw: str) -> tuple[Any | None, list[str]]:
    """Try increasingly desperate things to turn `raw` into a Python object.

    Ordered cheapest-and-safest first. Every rung records itself. If you find
    yourself adding a rung below `literal_eval`, stop and ask whether the
    prompt is the problem instead -- past that point you're guessing at
    intent, and a wrong guess executes a tool.
    """
    repairs: list[str] = []

    # 0. it's just fine
    try:
        return json.loads(raw), repairs
    except json.JSONDecodeError:
        pass

    # 1. trailing commas -- llama3.1 does this constantly
    cleaned = _TRAILING_COMMA.sub(r"\1", raw)
    if cleaned != raw:
        try:
            obj = json.loads(cleaned)
            repairs.append("trailing_comma")
            return obj, repairs
        except json.JSONDecodeError:
            pass

    # 2. unclosed braces/brackets -- count and append what's missing
    closed = _close_unbalanced(cleaned)
    if closed != cleaned:
        try:
            obj = json.loads(closed)
            repairs.append("unclosed_braces")
            return obj, repairs
        except json.JSONDecodeError:
            pass

    # 3. python literal syntax: single quotes, True/False/None.
    #    literal_eval and not eval, obviously -- this string came from a model
    #    and eval() on model output is how you get a very bad afternoon.
    for candidate, tag in ((cleaned, "python_literal"), (closed, "python_literal_closed")):
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, (dict, list)):
                repairs.append(tag)
                return obj, repairs
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass

    return None, repairs


def _close_unbalanced(s: str) -> str:
    stack = []
    in_str = False
    quote = ""
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    out = s
    if in_str:
        out += quote
    # strip a dangling `"key":` or trailing comma before closing, otherwise
    # we produce {"a": 1, "b":} which is still invalid
    out = re.sub(r',\s*"[^"]*"\s*:\s*$', "", out)
    out = re.sub(r",\s*$", "", out)
    return out + "".join(reversed(stack))


# normalising whatever shape came back

# The model does not reliably use the key names you asked for. Every one of
# these turned up in a real completion at some point.
_NAME_KEYS = ("name", "tool", "tool_name", "function", "action", "recipient_name")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "tool_input", "input", "action_input")


def normalize_call(obj: Any) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    """obj -> (name, args, repairs). (None, None, ...) means 'not a tool call'."""
    repairs: list[str] = []

    if not isinstance(obj, dict):
        return None, None, repairs

    # OpenAI nested shape: {"function": {"name": ..., "arguments": ...}}
    if isinstance(obj.get("function"), dict):
        inner = obj["function"]
        repairs.append("nested_function_key")
        n, a, r = normalize_call(inner)
        return n, a, repairs + r

    name = None
    for k in _NAME_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            if k != "name":
                repairs.append(f"name_key:{k}")
            break

    if name is None:
        return None, None, repairs

    args: Any = None
    for k in _ARG_KEYS:
        if k in obj:
            args = obj[k]
            if k != "arguments":
                repairs.append(f"arg_key:{k}")
            break

    if args is None:
        # No args key at all. Two readings: a genuinely zero-arg call, or the
        # model flattened the args up to the top level, which it does:
        #   {"name": "read_file", "path": "a.txt"}
        # Take the leftover keys as the args. Ambiguous, so it's recorded.
        leftovers = {k: v for k, v in obj.items() if k not in _NAME_KEYS + _ARG_KEYS}
        if leftovers:
            repairs.append("flattened_args")
            args = leftovers
        else:
            args = {}

    if isinstance(args, str):
        # args double-encoded as a JSON string. Extremely common, and the
        # official OpenAI API does this too, so it isn't even wrong.
        inner, r = loads_forgiving(args)
        if isinstance(inner, dict):
            repairs.append("args_double_encoded")
            repairs.extend(r)
            args = inner
        else:
            # a bare string arg. Can't guess the parameter name from here --
            # the registry can, since it knows the schema. Marker for it.
            repairs.append("args_bare_string")
            args = {"__positional__": args}

    if not isinstance(args, dict):
        repairs.append("args_not_object")
        return name, None, repairs

    # Models copy schema metadata back as if it were an argument -- "title"
    # especially, since it's in the JSON Schema they were shown.
    #
    # This loop used to have `pass` as its body and removed nothing at all.
    # Only `$schema` was ever popped, by the line underneath. It went unnoticed
    # because pydantic ignores unknown fields by default, so the junk was
    # silently dropped one layer down and nothing visibly broke. Masking by
    # luck, not by design -- and it would have surfaced the moment a tool
    # turned on strict extras. Caught in review 2026-07-25.
    #
    # `type` and `description` are NOT dropped: a tool is entitled to have
    # parameters with those names, and guessing wrong there would silently
    # delete a real argument. `title` and `$schema` are never plausible.
    for junk in ("title", "$schema"):
        args.pop(junk, None)

    return name, args, repairs


# the entry point


def _looks_deliberate(obj: Any, repairs: list[str]) -> bool:
    """Did the model mean this as a tool call, or is it just JSON?

    Only consulted for names the registry doesn't know. An explicit args key
    (or the nested-function wrapper) means yes. Args we inferred by flattening
    leftovers means no.
    """
    if "flattened_args" in repairs:
        return False
    return any(k in obj for k in _ARG_KEYS) or isinstance(obj.get("function"), dict)


def parse_tool_calls(text: str, known_names: set[str] | None = None) -> ParseResult:
    """Extract tool calls from raw model text.

    `known_names` is optional and is used only to decide whether a JSON blob
    is a tool call or just JSON the model was talking about. Without it, a
    model answering "what does package.json look like?" with a JSON snippet
    gets read as a tool call, which is a genuinely funny bug to debug.
    """
    STATS.completions += 1
    result = ParseResult(text=text)

    if not text or not text.strip():
        return result

    body, fenced = strip_fences(text)

    consumed: list[tuple[int, int]] = []
    all_repairs: list[str] = []

    for start, end in balanced_spans(body):
        raw = body[start:end]
        obj, repairs = loads_forgiving(raw)
        if obj is None:
            # Only complain if it plausibly wanted to be a call. Random braces
            # in prose shouldn't generate noise the model then has to read.
            if any(k in raw for k in ("name", "tool", "function")):
                result.problems.append(
                    f"couldn't parse this as JSON: {raw[:180]}"
                )
                STATS.unparseable += 1
            continue

        # a list of calls, or a list of anything
        candidates = obj if isinstance(obj, list) else [obj]
        matched_here = False

        for c in candidates:
            name, args, r = normalize_call(c)
            if name is None:
                continue
            if known_names is not None and name not in known_names:
                # Unknown name. Two very different situations:
                #
                #   {"name": "reed_file", "arguments": {...}}   <- typo'd call
                #   {"name": "my-app", "version": "1.0.0"}      <- package.json
                #
                # Keep the first: a hallucinated tool name is a genuine call
                # attempt, and the registry's "no such tool, here are the real
                # ones" is far more useful to the model than silence. Drop the
                # second: the model was quoting a file, and treating that as a
                # call is how you end up executing someone's package.json.
                #
                # The discriminator is an explicit args key. If the only thing
                # making it look like a call is that we flattened leftover
                # top-level keys into args, that's our inference, not the
                # model's intent, and it isn't enough on its own.
                if not _looks_deliberate(c, r):
                    continue
            call_repairs = repairs + r
            if fenced:
                call_repairs = ["fenced"] + call_repairs
            result.calls.append(
                ToolCall(
                    name=name,
                    arguments=args if args is not None else {},
                    raw=raw,
                    repairs=call_repairs,
                )
            )
            all_repairs.extend(call_repairs)
            matched_here = True

        if matched_here:
            consumed.append((start, end))

    # prose = everything not inside a consumed span
    if consumed:
        keep = []
        prev = 0
        for s, e in consumed:
            keep.append(body[prev:s])
            prev = e
        keep.append(body[prev:])
        result.text = "".join(keep).strip()

        # balanced_spans walks left to right, so consumed is already ordered
        result.prose_before = body[: consumed[0][0]].strip()
        result.prose_after = body[consumed[-1][1]:].strip()
    else:
        result.text = text.strip()

    # stats
    if result.calls:
        STATS.completions_with_calls += 1
        STATS.calls_found += len(result.calls)
        if all_repairs:
            STATS.completions_needing_repair += 1
            for r in set(all_repairs):
                STATS.note_repair(r)

    return result


def parse_native_tool_calls(raw_calls: list[dict[str, Any]]) -> ParseResult:
    """Ollama's native tools path. Structured already, so this is a rename.

    Kept alongside the messy path so the two can be run against each other.
    Without that comparison there is no way to say what the ladder above is
    buying you, only that it exists.
    """
    STATS.completions += 1
    res = ParseResult()
    for rc in raw_calls:
        fn = rc.get("function", rc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            # Native calls are meant to be structured, but Ollama occasionally
            # double-encodes the arguments as a string, and a malformed one must
            # not crash the loop. Same forgiving path the prompted parser uses;
            # an unparseable string becomes a positional marker the registry resolves.
            parsed, _ = loads_forgiving(args)
            args = parsed if isinstance(parsed, dict) else {"__positional__": args}
        # Use the provider's own call id when it supplied one (Groq/OpenAI
        # do; Ollama's tool_calls dicts don't carry one). This matters
        # because the hosted wire format pairs an assistant tool_calls[].id
        # against the following role=tool message's tool_call_id -- if we
        # discard the real id and hand back a locally-generated one instead,
        # a strict provider (real OpenAI more than Groq) has no way to know
        # the two belong together. Falls back to new_call_id() via the
        # dataclass default when there's no "id" key, same as before.
        call_id = rc.get("id")
        kwargs = {"name": fn.get("name", ""), "arguments": args, "raw": json.dumps(rc)}
        if call_id:
            kwargs["id"] = call_id
        res.calls.append(ToolCall(**kwargs))
    if res.calls:
        STATS.completions_with_calls += 1
        STATS.calls_found += len(res.calls)
    return res
