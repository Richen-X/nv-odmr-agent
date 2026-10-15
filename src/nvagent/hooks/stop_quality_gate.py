"""Stop hook: the analyst cannot finish until its fits pass the quality gate.

Claude Code runs this when the agent tries to end its turn. Printing
{"decision": "block", "reason": ...} returns the reason to the model and keeps it working,
which turns "fit this data" into a verify-and-retry loop. After ``max_blocks`` refusals the
stop is allowed and the failure recorded, so a run that cannot pass still terminates and
the reviewer sees why.
"""
from __future__ import annotations

import json
import sys

from ..qualitygate import gate_issues, gate_settings
from ..runstate import RunState


def main(stdin=None) -> int:
    try:
        event = json.loads((stdin or sys.stdin).read())
    except ValueError:
        event = {}
    state = RunState.from_env_or_none()
    if state is None:
        return 0
    session = str(event.get("session_id") or "default")
    issues = gate_issues(state)
    if not issues:
        state.record_gate(True, [], session)
        state.audit("quality_gate", session=session, passed=True)
        return 0
    max_blocks = gate_settings(state)["max_blocks"]
    attempt = state.count_gate_block(session)
    if attempt > max_blocks:
        state.record_gate(False, issues, session)
        state.audit("quality_gate", session=session, passed=False, issues=issues, gave_up=True)
        return 0
    state.audit("quality_gate", session=session, passed=False, issues=issues, attempt=attempt)
    reason = f"Quality gate not passed (attempt {attempt} of {max_blocks}): " + " ".join(issues)
    print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
