"""Real-ish model outputs.

Most of these I collected by running the loop against llama3.1:8b and dumping
completions that the parser choked on at the time. A few (SINGLE_QUOTES,
BARE_STRING_ARG) I wrote from memory of seeing them, so treat those two as
plausible rather than captured.

Each entry is (label, raw_text, expected_calls) where expected_calls is a list
of (name, args) or None meaning "should find nothing".
"""

CLEAN = '{"name": "read_file", "arguments": {"path": "notes.txt"}}'

FENCED = """```json
{"name": "read_file", "arguments": {"path": "notes.txt"}}
```"""

FENCED_NO_LANG = """```
{"name": "read_file", "arguments": {"path": "notes.txt"}}
```"""

FENCE_UNTERMINATED = """Sure, here you go:
```json
{"name": "read_file", "arguments": {"path": "notes.txt"}}"""

PROSE_BOTH_SIDES = """Of course! To answer that I'll need to look at the file.

{"name": "read_file", "arguments": {"path": "notes.txt"}}

Let me know if you'd like me to check anything else."""

SINGLE_QUOTES = "{'name': 'read_file', 'arguments': {'path': 'notes.txt'}}"

TRAILING_COMMA = '{"name": "read_file", "arguments": {"path": "notes.txt",},}'

ARGS_DOUBLE_ENCODED = '{"name": "read_file", "arguments": "{\\"path\\": \\"notes.txt\\"}"}'

WRONG_KEY_NAMES = '{"tool": "read_file", "tool_input": {"path": "notes.txt"}}'

NESTED_FUNCTION = (
    '{"type": "function", "function": {"name": "read_file", '
    '"arguments": {"path": "notes.txt"}}}'
)

FLATTENED_ARGS = '{"name": "read_file", "path": "notes.txt"}'

BARE_STRING_ARG = '{"name": "read_file", "arguments": "notes.txt"}'

TWO_CALLS_NO_ARRAY = """{"name": "read_file", "arguments": {"path": "a.txt"}}
{"name": "read_file", "arguments": {"path": "b.txt"}}"""

ARRAY_OF_CALLS = """[
  {"name": "read_file", "arguments": {"path": "a.txt"}},
  {"name": "grep", "arguments": {"pattern": "TODO"}}
]"""

UNCLOSED_BRACES = '{"name": "read_file", "arguments": {"path": "notes.txt"'

NESTED_BRACES_IN_ARG = (
    '{"name": "write_file", "arguments": {"path": "x.json", '
    '"content": "{\\"a\\": {\\"b\\": 1}}"}}'
)

BRACE_INSIDE_STRING = (
    '{"name": "grep", "arguments": {"pattern": "func() { return }"}}'
)

HALLUCINATED_NAME = '{"name": "reed_file", "arguments": {"path": "notes.txt"}}'

MISSING_REQUIRED = '{"name": "read_file", "arguments": {}}'

EXTRA_ARGS = (
    '{"name": "read_file", "arguments": {"path": "notes.txt", '
    '"encoding": "utf-8", "title": "Read File"}}'
)

NO_CALL_AT_ALL = "The file contains a list of grocery items. Nothing else stood out."

JSON_THE_MODEL_IS_TALKING_ABOUT = """Your package.json looks like this:

{"name": "my-app", "version": "1.0.0", "dependencies": {"react": "^18"}}

Nothing looks wrong with it."""

EMPTY = ""

WHITESPACE_ONLY = "   \n\n  "


# (label, text, expected) -- expected is list[(name, args)] or [] for none
CASES = [
    ("clean", CLEAN, [("read_file", {"path": "notes.txt"})]),
    ("fenced", FENCED, [("read_file", {"path": "notes.txt"})]),
    ("fenced_no_lang", FENCED_NO_LANG, [("read_file", {"path": "notes.txt"})]),
    ("fence_unterminated", FENCE_UNTERMINATED, [("read_file", {"path": "notes.txt"})]),
    ("prose_both_sides", PROSE_BOTH_SIDES, [("read_file", {"path": "notes.txt"})]),
    ("single_quotes", SINGLE_QUOTES, [("read_file", {"path": "notes.txt"})]),
    ("trailing_comma", TRAILING_COMMA, [("read_file", {"path": "notes.txt"})]),
    ("args_double_encoded", ARGS_DOUBLE_ENCODED, [("read_file", {"path": "notes.txt"})]),
    ("wrong_key_names", WRONG_KEY_NAMES, [("read_file", {"path": "notes.txt"})]),
    ("nested_function", NESTED_FUNCTION, [("read_file", {"path": "notes.txt"})]),
    ("flattened_args", FLATTENED_ARGS, [("read_file", {"path": "notes.txt"})]),
    ("unclosed_braces", UNCLOSED_BRACES, [("read_file", {"path": "notes.txt"})]),
    (
        "two_calls_no_array",
        TWO_CALLS_NO_ARRAY,
        [("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})],
    ),
    (
        "array_of_calls",
        ARRAY_OF_CALLS,
        [("read_file", {"path": "a.txt"}), ("grep", {"pattern": "TODO"})],
    ),
    ("brace_inside_string", BRACE_INSIDE_STRING,
     [("grep", {"pattern": "func() { return }"})]),
    ("hallucinated_name", HALLUCINATED_NAME, [("reed_file", {"path": "notes.txt"})]),
    ("missing_required", MISSING_REQUIRED, [("read_file", {})]),
    ("no_call_at_all", NO_CALL_AT_ALL, []),
    ("empty", EMPTY, []),
    ("whitespace_only", WHITESPACE_ONLY, []),
]
