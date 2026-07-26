"""Command line entry point.

    secret-agent "what is the retention default for bronze?"
    secret-agent --repl
    secret-agent --demo

Deliberately small. The interesting code is in agent.py; this is argument
parsing and printing.
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent, AgentFailure
from .config import Config
from .llm import LLMError, OllamaClient
from .parsing import STATS
from .permissions import default_permissions
from .tools.fs import FS_TOOLS
from .tools.registry import Registry
from .tools.shell import SHELL_TOOLS

BANNER = """secret-agent -- {model}, {n} tools, num_ctx={ctx}, mode={mode}
/dump   print the full conversation as the model sees it
/stats  parse repair rate, context compactions, permissions
/quit"""


def build_agent(args) -> Agent:
    cfg = Config.from_env()
    cfg.verbose = args.verbose
    if args.model:
        cfg.model = args.model
    if args.max_iter:
        cfg.max_iterations = args.max_iter
    if args.strategy:
        cfg.strategy = args.strategy
    cfg.auto_approve = args.yes

    tools = list(FS_TOOLS) + list(SHELL_TOOLS)
    if args.rag:
        from .rag import RAG_TOOLS
        tools += list(RAG_TOOLS)

    perms = default_permissions(auto_approve=cfg.auto_approve)
    reg = Registry(tools, permissions=perms)
    agent = Agent(reg, OllamaClient(cfg), cfg)
    agent._perms = perms  # for /stats
    return agent


def show_stats(agent) -> None:
    print("\n" + STATS.summary())
    print(agent.ctx.report())
    if hasattr(agent, "_perms"):
        print("permissions: " + agent._perms.summary())


def run_once(agent, task: str) -> int:
    try:
        run = agent.run(task)
    except AgentFailure as e:
        print(f"\nfailed: {e}", file=sys.stderr)
        print(f"(partial transcript has {len(e.conversation)} messages; "
              f"re-run with --verbose to watch it)", file=sys.stderr)
        return 2
    except LLMError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print("\n" + run.answer)
    print(f"\n[{run.summary()}]", file=sys.stderr)
    return 0


def repl(agent) -> int:
    print(BANNER.format(model=agent.cfg.model, n=len(agent.registry),
                        ctx=agent.cfg.num_ctx, mode=agent.cfg.tool_mode))
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line == "/dump":
            print(agent.conversation.transcript())
            continue
        if line == "/stats":
            show_stats(agent)
            continue

        try:
            run = agent.run(line)
            print("\n" + run.answer)
            if agent.cfg.verbose:
                print(f"[{run.summary()}]", file=sys.stderr)
        except AgentFailure as e:
            print(f"\nfailed: {e}", file=sys.stderr)
        except LLMError as e:
            print(f"\n{e}", file=sys.stderr)
            return 1


def demo() -> int:
    """The one that shows retrieval doing something, rather than existing.

    Asks the same question twice, once with search_docs available and once
    without. MER-4471 is invented, so the model cannot answer it from its
    parameters -- which makes the difference visible instead of assumed.
    """
    from .rag import RAG_TOOLS

    q = ("What error code does Meridian return when you try to delete a dataset "
         "under legal hold, and is it retryable?")
    cfg = Config(max_iterations=6)
    client = OllamaClient(cfg)

    print(f"Q: {q}\n")

    print("=" * 70)
    print("WITHOUT retrieval -- no tools at all")
    print("=" * 70)
    bare = Agent(Registry([]), client, cfg)
    print(bare.run(q).answer.strip()[:600])

    print("\n" + "=" * 70)
    print("WITH retrieval -- search_docs available")
    print("=" * 70)
    reg = Registry(RAG_TOOLS, permissions=default_permissions(auto_approve=True))
    run = Agent(reg, client, cfg).run(q)
    print(run.answer.strip())
    print(f"\n[{run.summary()}]")

    print("\nMER-4471 appears nowhere outside corpus/retention.md. If the second")
    print("answer has it and the first doesn't, retrieval is the difference.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="secret-agent", description=__doc__)
    ap.add_argument("task", nargs="*", help="what to do; omit for a REPL")
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--demo", action="store_true", help="retrieval on vs off")
    ap.add_argument("--rag", action="store_true", help="add the search_docs tool")
    ap.add_argument("--model")
    ap.add_argument("--strategy", choices=["truncate", "summarize"])
    ap.add_argument("--max-iter", type=int)
    ap.add_argument("-y", "--yes", action="store_true",
                    help="auto-approve every permission prompt. Know what you're running.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.demo:
        return demo()

    agent = build_agent(args)
    task = " ".join(args.task)
    if task and not args.repl:
        return run_once(agent, task)
    return repl(agent)


if __name__ == "__main__":
    raise SystemExit(main())
