"""Context management. Phase 5 fills this in.

For now it's the seam, not the implementation: the loop calls ensure_fits()
before every model call and trim_tool_result() after every tool, and both are
pass-throughs. Putting the call sites in first means phase 5 is a change to
one file instead of surgery on the loop.
"""

from __future__ import annotations

from .config import Config
from .conversation import Conversation
from .tools.base import ToolResult


class ContextManager:
    def __init__(self, cfg: Config, client=None):
        self.cfg = cfg
        self.client = client

    def ensure_fits(self, conv: Conversation) -> bool:
        """Returns True if it had to compact. TODO(phase5)"""
        return False

    def trim_tool_result(self, result: ToolResult) -> ToolResult:
        """TODO(phase5) -- a grep returning 4000 lines should not enter
        history whole."""
        return result
