"""What a tool is.

A tool = a name, a description the model reads, a pydantic model describing
its arguments, and a run(). The JSON Schema handed to the model is generated
from the pydantic model rather than written by hand, so the schema the model
sees and the validation the args go through can't drift apart. Hand-writing
both is how you end up with a tool that documents an argument it rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Raised inside run() when the tool cannot do the thing.

    This is NOT a crash. The loop catches it and hands the message back to
    the model as a tool result, because 'file not found' is information the
    model can act on -- it might have guessed a path.
    """


class ToolDenied(ToolError):
    """The permission layer said no. Separate type so the loop can tell
    'you may not' apart from 'it failed', and so it never gets retried."""


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    ok: bool = True
    # set when the content got trimmed for context budget; the full thing is
    # on disk at spill_path
    truncated: bool = False
    spill_path: str | None = None
    duration_ms: float = 0.0

    def for_model(self) -> str:
        if self.ok:
            return self.content
        return f"ERROR: {self.content}"


class Tool:
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    Args: ClassVar[type[BaseModel]]

    # permission default; the policy in permissions.py can override per-tool
    default_policy: ClassVar[str] = "ask"

    def run(self, **kwargs) -> str:
        raise NotImplementedError


    @classmethod
    def schema(cls) -> dict[str, Any]:
        """OpenAI/Ollama-style function schema. Anthropic wants
        input_schema instead of parameters -- one rename if that day comes."""
        params = cls.Args.model_json_schema()
        # pydantic emits $defs + title keys that just burn tokens in the
        # system prompt. Small models also seem to latch onto "title" and
        # occasionally emit it as an argument, which is its own kind of funny.
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description.strip(),
                "parameters": params,
            },
        }

    @classmethod
    def validate_args(cls, raw: dict[str, Any]) -> BaseModel:
        """Separate from parsing on purpose.

        Parsing answers 'is this JSON'. This answers 'is this a legal call'.
        They fail for different reasons and the model needs different words
        back for each, otherwise the retry is a guess.
        """
        try:
            return cls.Args(**raw)
        except ValidationError as e:
            raise ToolError(_readable_validation_error(e)) from e
        except TypeError as e:
            # e.g. arguments came through as a list. pydantic raises TypeError
            # rather than ValidationError for that and it took me a while.
            raise ToolError(f"arguments must be an object: {e}") from e


def _readable_validation_error(e: ValidationError) -> str:
    """pydantic's default str() is several lines of URLs. The model does not
    need a link to docs.pydantic.dev; it needs the field name and what's
    wrong with it, in one line."""
    bits = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        bits.append(f"{loc}: {err['msg']}")
    return "invalid arguments -- " + "; ".join(bits)


def render_tools_for_prompt(tools: list[type[Tool]]) -> str:
    """Text block describing the tools, for prompted mode.

    Format matters more than you'd expect. Things I tried, in order:

      1. Dumping raw JSON Schema. llama3.1 started echoing schema fragments
         back as if they were tool calls. Too much shape to copy.
      2. Terse one-liners `name(a, b) - desc`. Under-specified; the model
         invented arguments constantly.
      3. What's below -- signature line, description, then required/optional
         args one per line. Fewest bad calls of the three by a clear margin.

    That's an eyeball comparison over maybe 30 runs, not an eval. Saying it
    out loud so nobody quotes it as a measurement.
    """
    out = []
    for t in tools:
        schema = t.schema()["function"]
        props = schema["parameters"].get("properties", {})
        required = set(schema["parameters"].get("required", []))
        sig = ", ".join(props.keys())
        out.append(f"{t.name}({sig})")
        out.append(f"  {schema['description']}")
        for pname, pinfo in props.items():
            tag = "required" if pname in required else "optional"
            desc = pinfo.get("description", "")
            ptype = pinfo.get("type", pinfo.get("anyOf", "any"))
            if isinstance(ptype, list):
                ptype = "any"
            out.append(f"    - {pname} ({ptype}, {tag}): {desc}")
        out.append("")
    return "\n".join(out).rstrip()
