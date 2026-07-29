"""Model Context Protocol support.

Client side (primary): connect to any MCP server over stdio, discover its
tools, and map them into the runtime's tool schema + dispatch so they run
through the same repair ladder, permission layer and sandbox as native tools.

Server side (optional): expose this runtime's own tools behind an MCP server
interface, so another MCP client can drive them under the same guardrails.
"""

from .adapter import make_mcp_tool, make_mcp_tools
from .client import MCPError, MCPStdioClient

__all__ = ["MCPStdioClient", "MCPError", "make_mcp_tools", "make_mcp_tool"]
