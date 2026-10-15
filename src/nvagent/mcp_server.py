"""MCP server exposing the virtual NV lab to Claude Code over stdio.

    python -m nvagent.mcp_server --role planner --run-dir runs/<id>

Each agent role gets only its own tools, so the separation of duties does not rely on the
system prompt alone: the analyst cannot measure, the reviewer can neither measure nor fit.
"""
from __future__ import annotations

import argparse
import os

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from . import tools

_COMMON = [tools.get_run_summary, tools.add_notebook_entry]
ROLE_TOOLS = {
    "planner": _COMMON + [
        tools.get_instrument_info, tools.measure_odmr_sweep, tools.measure_odmr_points,
        tools.find_odmr_lines, tools.run_adaptive_odmr, tools.measure_rabi, tools.measure_t1,
    ],
    "analyst": _COMMON + [
        tools.find_odmr_lines, tools.fit_odmr, tools.fit_field_vector, tools.fit_rabi, tools.fit_decay, tools.view_fit_plot,
    ],
    "reviewer": _COMMON + [tools.get_fit_report, tools.check_field_consistency, tools.view_fit_plot],
}
ROLE_TOOLS["all"] = list(dict.fromkeys(fn for fns in ROLE_TOOLS.values() for fn in fns))


def build_server(role: str = "all") -> MCPServer:
    if role not in ROLE_TOOLS:
        raise ValueError(f"unknown role {role!r}; choose from {sorted(ROLE_TOOLS)}")
    server = MCPServer("nvlab", instructions=f"Virtual NV-centre ODMR laboratory: {role} tools.")
    for fn in ROLE_TOOLS[role]:
        server.add_tool(fn)
    return server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", default=os.environ.get("NVAGENT_ROLE", "all"), choices=sorted(ROLE_TOOLS))
    parser.add_argument("--run-dir", default=os.environ.get("NVAGENT_RUN_DIR"))
    args = parser.parse_args(argv)
    if not args.run_dir:
        parser.error("--run-dir (or NVAGENT_RUN_DIR) is required")
    tools.init(args.run_dir)
    build_server(args.role).run("stdio")


if __name__ == "__main__":
    main()
