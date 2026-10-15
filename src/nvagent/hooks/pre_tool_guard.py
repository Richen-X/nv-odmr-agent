"""PreToolUse hook: the instrument interlock between the model and the (virtual) hardware.

Claude Code runs this before every tool call matching ``mcp__nvlab__.*`` and passes the call
as JSON on stdin. Exit code 2 blocks the call and returns stderr to the model so it can
correct the request; nothing the model says can override the limits. Every decision is
written to the run's audit log.
"""
from __future__ import annotations

import json
import sys

from ..runstate import RunState
from ..safety import DEFAULT_LIMITS, INSTRUMENT_TOOLS, check_request

PREFIX = "mcp__nvlab__"


def main(stdin=None) -> int:
    try:
        event = json.loads((stdin or sys.stdin).read())
        name = str(event.get("tool_name", ""))
        args = event.get("tool_input") or {}
    except (ValueError, AttributeError):
        print("nvagent interlock: unreadable hook input, refusing the instrument call.", file=sys.stderr)
        return 2
    tool = name[len(PREFIX):] if name.startswith(PREFIX) else None
    if tool not in INSTRUMENT_TOOLS:
        return 0
    state = RunState.from_env_or_none()
    problems = check_request(tool, args, state.limits() if state else DEFAULT_LIMITS, state.points_used() if state else 0)
    if state:
        state.audit("guard", tool=tool, args=args, decision="deny" if problems else "allow", reasons=problems, session=event.get("session_id"))
    if problems:
        print("Blocked by the instrument interlock: " + "; ".join(problems) + ". Adjust the request and try again.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
