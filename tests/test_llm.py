"""Provider selection and hosted-client construction, all offline.

No test here talks to a real network. The hosted path is exercised against a
fake standing in for openai.OpenAI, so these pass with no GROQ_API_KEY and no
internet -- construction and request/response mapping are what's under test,
not Groq itself. That's what "at least one full agent task through the
hosted provider" in EXTENSIONS-2026-08-05.md needs a live key for; this file
is the part that has to hold without one.
"""

from __future__ import annotations

import os

import pytest

from secret_agent.config import Config, _load_dotenv_local
from secret_agent.llm import HostedClient, LLMError, OllamaClient, build_llm_client
import secret_agent.config as config_module
import secret_agent.llm as llm_module


# --- helpers ---------------------------------------------------------------

ENV_VARS = (
    "LLM_PROVIDER",
    "SA_HOSTED_BASE_URL",
    "SA_HOSTED_MODEL",
    "SA_HOSTED_API_KEY",
    "GROQ_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No test should pick up whatever's actually exported in this shell.

    Also no-ops the real .env.local loader for every test except the ones
    that test it directly below -- otherwise a real .env.local sitting in
    the repo root (e.g. once Kalyan puts his own GROQ_API_KEY there to run
    the hosted eval) would make these tests' results depend on what's on
    disk that day. The dedicated .env.local tests call `_load_dotenv_local`
    directly with an explicit tmp_path candidate instead of going through
    this patched version.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_load_dotenv_local", lambda *a, **k: None)


class FakeChoice:
    def __init__(self, content, tool_calls=None):
        self.message = FakeMessage(content, tool_calls)


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, id="call_fake123"):
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeUsage:
    def __init__(self, prompt_tokens=12, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, content, tool_calls=None, usage=None):
        self.choices = [FakeChoice(content, tool_calls)]
        self.usage = usage or FakeUsage()

    def model_dump(self):
        return {"fake": True}


class FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeChat:
    def __init__(self, response):
        self.completions = FakeCompletions(response)


class FakeOpenAI:
    """Stand-in for openai.OpenAI. Records constructor kwargs so tests can
    assert base_url/api_key/timeout were threaded through from Config, and
    hands back a scripted response from .chat.completions.create().
    """

    instances: list["FakeOpenAI"] = []

    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = FakeChat(FakeResponse("hi"))
        self.closed = False
        FakeOpenAI.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_instances():
    FakeOpenAI.instances = []
    yield
    FakeOpenAI.instances = []


# --- provider selection: the default must not move -------------------------


def test_default_provider_with_no_env_set_is_ollama():
    """The whole point of D-keep-ollama-default: LLM_PROVIDER unset must
    build exactly what OllamaClient() always built. No hosted config, no
    API key, no openai import required to reach this path.
    """
    cfg = Config()
    client = build_llm_client(cfg)
    assert isinstance(client, OllamaClient)
    assert cfg.llm_provider == "ollama"


def test_from_env_defaults_to_ollama_provider(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    cfg = Config.from_env()
    assert cfg.llm_provider == "ollama"
    assert isinstance(build_llm_client(cfg), OllamaClient)


def test_llm_provider_env_var_selects_hosted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-123")
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config.from_env()
    client = build_llm_client(cfg)
    assert isinstance(client, HostedClient)


def test_unknown_provider_raises():
    cfg = Config(llm_provider="anthropic")
    with pytest.raises(LLMError, match="unknown LLM_PROVIDER"):
        build_llm_client(cfg)


# --- hosted client construction --------------------------------------------


def test_hosted_client_needs_an_api_key(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="")
    with pytest.raises(LLMError, match="API key"):
        HostedClient(cfg)


def test_hosted_client_without_openai_installed_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", None)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key")
    with pytest.raises(LLMError, match="openai"):
        HostedClient(cfg)


def test_hosted_client_defaults_are_groq(monkeypatch):
    """Groq is the built-in default: no config beyond exporting
    GROQ_API_KEY should be needed to point at it.
    """
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key")
    HostedClient(cfg)
    assert len(FakeOpenAI.instances) == 1
    inst = FakeOpenAI.instances[0]
    assert inst.api_key == "fake-key"
    assert inst.base_url == "https://api.groq.com/openai/v1"


def test_hosted_client_swaps_provider_via_config_alone(monkeypatch):
    """The acceptance bar from the brief: pointing this at Gemini or
    OpenRouter is base_url + key + model, no code change. Simulated here by
    constructing Config with different values -- exactly what changing
    SA_HOSTED_BASE_URL / SA_HOSTED_MODEL / SA_HOSTED_API_KEY does.
    """
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(
        llm_provider="hosted",
        hosted_base_url="https://openrouter.ai/api/v1",
        hosted_model="meta-llama/llama-3.1-8b-instruct:free",
        hosted_api_key="or-fake-key",
    )
    client = HostedClient(cfg)
    inst = FakeOpenAI.instances[0]
    assert inst.base_url == "https://openrouter.ai/api/v1"
    assert client.cfg.hosted_model == "meta-llama/llama-3.1-8b-instruct:free"


def test_hosted_base_url_and_model_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setenv("SA_HOSTED_API_KEY", "fake-key")
    monkeypatch.setenv("SA_HOSTED_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("SA_HOSTED_MODEL", "gemini-1.5-flash")
    cfg = Config.from_env()
    assert cfg.hosted_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert cfg.hosted_model == "gemini-1.5-flash"
    assert cfg.hosted_api_key == "fake-key"


def test_sa_hosted_api_key_wins_over_groq_api_key(monkeypatch):
    """So a non-Groq provider can be configured without unsetting
    GROQ_API_KEY first."""
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("SA_HOSTED_API_KEY", "explicit-key")
    cfg = Config.from_env()
    assert cfg.hosted_api_key == "explicit-key"


def test_groq_api_key_used_when_sa_hosted_api_key_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    cfg = Config.from_env()
    assert cfg.hosted_api_key == "groq-key"


# --- request/response mapping ----------------------------------------------


def test_complete_maps_text_response(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key", hosted_model="llama-3.1-8b-instant")
    client = HostedClient(cfg)
    client._client.chat.completions._response = FakeResponse(
        "the answer is 4", usage=FakeUsage(prompt_tokens=20, completion_tokens=6)
    )

    out = client.complete([{"role": "user", "content": "2+2"}])

    assert out.text == "the answer is 4"
    assert out.usage.prompt_tokens == 20
    assert out.usage.completion_tokens == 6
    assert out.native_tool_calls == []
    sent = client._client.chat.completions.calls[0]
    assert sent["model"] == "llama-3.1-8b-instant"
    assert sent["messages"] == [{"role": "user", "content": "2+2"}]


def test_complete_maps_native_tool_calls_into_the_shape_parsing_expects(monkeypatch):
    """secret_agent.parsing.parse_native_tool_calls reads
    rc["function"]["name"] and rc["function"]["arguments"] (a JSON string).
    HostedClient must hand back exactly that shape, not the openai SDK's
    object, or the native-mode path silently breaks for hosted models.

    Also covers the id: Groq assigns a real tool_call id (tc.id), and
    HostedClient used to discard it and let parse_native_tool_calls generate
    a fresh local one instead. That breaks the tool_call_id pairing a strict
    OpenAI-compatible provider requires (Groq happened to tolerate it; real
    OpenAI does not) -- see test_hosted_tool_loop_wire_format in
    test_conversation.py for the end-to-end version of this.
    """
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key", tool_mode="native")
    client = HostedClient(cfg)
    client._client.chat.completions._response = FakeResponse(
        None,
        tool_calls=[FakeToolCall("read_file", '{"path": "README.md"}', id="call_real_abc123")],
    )

    out = client.complete([{"role": "user", "content": "read the readme"}])

    assert out.text == ""
    assert out.native_tool_calls == [
        {
            "id": "call_real_abc123",
            "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
        }
    ]

    from secret_agent.parsing import parse_native_tool_calls

    parsed = parse_native_tool_calls(out.native_tool_calls)
    assert len(parsed.calls) == 1
    assert parsed.calls[0].name == "read_file"
    assert parsed.calls[0].arguments == {"path": "README.md"}
    # The real provider id, not a locally-generated one.
    assert parsed.calls[0].id == "call_real_abc123"


def test_tools_only_forwarded_in_native_mode(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key", tool_mode="prompted")
    client = HostedClient(cfg)
    client.complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    sent = client._client.chat.completions.calls[0]
    assert "tools" not in sent


def test_tools_forwarded_in_native_mode(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key", tool_mode="native")
    client = HostedClient(cfg)
    schema = [{"type": "function", "function": {"name": "read_file"}}]
    client.complete([{"role": "user", "content": "hi"}], tools=schema)
    sent = client._client.chat.completions.calls[0]
    assert sent["tools"] == schema


def test_sdk_exception_is_wrapped_as_llmerror(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key")
    client = HostedClient(cfg)

    def blow_up(**kwargs):
        raise RuntimeError("401 unauthorized")

    client._client.chat.completions.create = blow_up

    with pytest.raises(LLMError, match="hosted provider call failed"):
        client.complete([{"role": "user", "content": "hi"}])


def test_close_delegates_to_the_underlying_client(monkeypatch):
    monkeypatch.setattr(llm_module, "_OpenAI", FakeOpenAI)
    cfg = Config(llm_provider="hosted", hosted_api_key="fake-key")
    client = HostedClient(cfg)
    client.close()
    assert client._client.closed is True


# --- .env.local -------------------------------------------------------
#
# These call _load_dotenv_local directly with an explicit candidate list
# (bypassing the autouse fixture's no-op patch and from_env's own default
# repo-root/cwd lookup) so the result depends only on the tmp file each
# test writes, never on whatever is or isn't on disk in the real repo.


def test_env_local_populates_missing_vars(tmp_path):
    dotenv = tmp_path / ".env.local"
    dotenv.write_text("GROQ_API_KEY=from-dotenv\nLLM_PROVIDER=hosted\n")

    _load_dotenv_local(candidates=[dotenv])

    assert os.environ["GROQ_API_KEY"] == "from-dotenv"
    assert os.environ["LLM_PROVIDER"] == "hosted"
    cfg = Config.from_env()
    assert cfg.hosted_api_key == "from-dotenv"
    assert cfg.llm_provider == "hosted"


def test_env_local_does_not_override_a_real_env_var(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env.local"
    dotenv.write_text("GROQ_API_KEY=from-dotenv\n")
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")

    _load_dotenv_local(candidates=[dotenv])

    assert os.environ["GROQ_API_KEY"] == "from-shell"


def test_missing_env_local_is_a_silent_no_op(tmp_path):
    dotenv = tmp_path / ".env.local"  # deliberately never created
    _load_dotenv_local(candidates=[dotenv])  # must not raise
    assert "GROQ_API_KEY" not in os.environ
    cfg = Config.from_env()
    assert cfg.llm_provider == "ollama"
    assert cfg.hosted_api_key == ""


def test_env_local_lines_without_equals_or_comments_are_skipped(tmp_path):
    dotenv = tmp_path / ".env.local"
    dotenv.write_text(
        "# a comment\n\nGROQ_API_KEY=real-value\nnot a kv line\n"
    )
    _load_dotenv_local(candidates=[dotenv])
    assert os.environ["GROQ_API_KEY"] == "real-value"


def test_env_local_strips_inline_trailing_comment(tmp_path):
    """`KEY=val  # note` used to store the comment as part of the value.
    Caught in review 2026-08-05."""
    dotenv = tmp_path / ".env.local"
    dotenv.write_text("SA_HOSTED_MODEL=llama-3.1-8b-instant  # groq's free-tier model\n")
    _load_dotenv_local(candidates=[dotenv])
    assert os.environ["SA_HOSTED_MODEL"] == "llama-3.1-8b-instant"


def test_env_local_quoted_value_with_inline_comment(tmp_path):
    dotenv = tmp_path / ".env.local"
    dotenv.write_text('GROQ_API_KEY="gsk_abc123"  # personal key, don\'t commit\n')
    _load_dotenv_local(candidates=[dotenv])
    assert os.environ["GROQ_API_KEY"] == "gsk_abc123"


def test_env_local_hash_inside_quotes_is_not_treated_as_a_comment(tmp_path):
    dotenv = tmp_path / ".env.local"
    dotenv.write_text('SOME_TOKEN="abc#def"\n')
    _load_dotenv_local(candidates=[dotenv])
    assert os.environ["SOME_TOKEN"] == "abc#def"
