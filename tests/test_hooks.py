"""Claude Code hooks, exercised as real subprocesses with the JSON Claude Code sends them."""
import json
import os
import subprocess
import sys

from nvagent import tools

GUARD = "nvagent.hooks.pre_tool_guard"
GATE = "nvagent.hooks.stop_quality_gate"


def run_hook(module, payload, run_dir):
    env = {**os.environ, "NVAGENT_RUN_DIR": str(run_dir)}
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run([sys.executable, "-m", module], input=stdin, capture_output=True, text=True, env=env, timeout=120)


def pre_tool(tool, **args):
    return {"hook_event_name": "PreToolUse", "session_id": "s1", "tool_name": f"mcp__nvlab__{tool}", "tool_input": args}


def stop(session):
    return {"hook_event_name": "Stop", "session_id": session, "stop_hook_active": False}


def test_guard_allows_a_safe_instrument_call_and_audits_it(run_state):
    p = run_hook(GUARD, pre_tool("measure_odmr_sweep", start_mhz=2640, stop_mhz=3100, step_mhz=3, mw_power_dbm=10), run_state.dir)
    assert p.returncode == 0
    last = json.loads((run_state.dir / "audit.jsonl").read_text().splitlines()[-1])
    assert last["event"] == "guard" and last["decision"] == "allow"


def test_guard_blocks_excess_mw_power_and_tells_the_model_why(run_state):
    p = run_hook(GUARD, pre_tool("measure_odmr_sweep", start_mhz=2640, stop_mhz=3100, step_mhz=3, mw_power_dbm=20), run_state.dir)
    assert p.returncode == 2
    assert "mw_power_dbm" in p.stderr


def test_guard_blocks_requests_beyond_the_point_budget(run_state):
    tools.measure_odmr_sweep(2640.0, 3100.0, 0.25, 0.0)  # 1841 of 3000 points
    p = run_hook(GUARD, pre_tool("measure_odmr_sweep", start_mhz=2640, stop_mhz=3100, step_mhz=0.25, mw_power_dbm=0), run_state.dir)
    assert p.returncode == 2
    assert "budget" in p.stderr


def test_guard_ignores_analysis_tools(run_state):
    assert run_hook(GUARD, {"tool_name": "mcp__nvlab__fit_odmr", "tool_input": {}}, run_state.dir).returncode == 0


def test_guard_fails_closed_on_unreadable_input(run_state):
    assert run_hook(GUARD, "not json", run_state.dir).returncode == 2


def test_stop_gate_blocks_until_the_fit_passes(run_state):
    p = run_hook(GATE, stop("a1"), run_state.dir)
    assert json.loads(p.stdout)["decision"] == "block"
    assert "fit_odmr" in p.stdout
    tools.measure_odmr_sweep(2640.0, 3100.0, 0.25, 0.0)
    fit = tools.fit_odmr()
    tools.fit_field_vector(fit["fit_id"])
    p = run_hook(GATE, stop("a1"), run_state.dir)
    assert p.returncode == 0 and p.stdout.strip() == ""


def test_stop_gate_gives_up_after_max_blocks_and_records_the_failure(run_state):
    outputs = [run_hook(GATE, stop("b2"), run_state.dir).stdout for _ in range(4)]
    assert all('"block"' in out for out in outputs[:3])
    assert outputs[3].strip() == ""
    gate = json.loads((run_state.dir / "state" / "quality_gate.json").read_text())
    assert gate["passed"] is False
