"""Test double for `claude -p`: checks the invocation and plays each role through the real tools.

The reviewer asks for a revision in round 1 so that the orchestrator's feedback loop runs.
"""
import json
import os
import sys
from pathlib import Path


def arg(name):
    return sys.argv[sys.argv.index(name) + 1]


def main():
    prompt = sys.stdin.read()
    if os.environ.get("FAKE_CLAUDE_SLEEP"):
        import time

        time.sleep(float(os.environ["FAKE_CLAUDE_SLEEP"]))
    server = json.loads(Path(arg("--mcp-config")).read_text())["mcpServers"]["nvlab"]
    role, run_dir = server["env"]["NVAGENT_ROLE"], server["env"]["NVAGENT_RUN_DIR"]
    # The invocation must isolate the agent: no parent session, no built-in tools, only MCP tools.
    assert not os.environ.get("CLAUDECODE"), "child agent inherited the parent Claude Code session"
    assert arg("--tools") == "" and arg("--permission-mode") == "dontAsk" and "--strict-mcp-config" in sys.argv
    assert all(t.startswith("mcp__nvlab__") for t in arg("--allowedTools").split(","))
    # No argument may contain a newline (Windows launchers re-parse the command line).
    assert "--system-prompt" not in sys.argv and not any(chr(10) in a for a in sys.argv)
    assert Path(arg("--system-prompt-file")).read_text().startswith("You are")
    json.loads(arg("--json-schema"))
    hooks = json.loads(Path(arg("--settings")).read_text())["hooks"]
    assert ("Stop" in hooks) == (role == "analyst")

    from nvagent import tools

    tools.init(run_dir)
    first_round = "Round 1 " in prompt
    if role == "planner":
        if first_round:
            survey = tools.measure_odmr_sweep(2640.0, 3100.0, 3.0, 10.0)
            centres = [c["center_mhz"] for c in tools.find_odmr_lines(z_min=6.0)["candidates"]]
            refine = tools.run_adaptive_odmr(centres, 384)
            done = [f"survey {survey['dataset_id']}", f"adaptive {refine['dataset_id']}"]
        else:
            centres = [c["center_mhz"] for c in tools.find_odmr_lines(z_min=4.0)["candidates"]]
            done = [f"adaptive {tools.run_adaptive_odmr(centres, 96)['dataset_id']}"]
        out = {"measurements": done, "points_used_total": tools.get_run_summary()["points_used"], "ready_for_analysis": True, "rationale": "fake planner"}
    elif role == "analyst":
        fit = tools.fit_odmr()
        field = tools.fit_field_vector(fit["fit_id"])
        out = {
            "odmr_fit_id": fit["fit_id"], "field_fit_id": field["fit_id"],
            "lines": [{"center_mhz": ln["center_mhz"], "center_err_mhz": ln["center_err_mhz"]} for ln in fit["lines"]],
            "b_canonical_g": field["b_canonical_g"], "b_err_g": field["b_err_g"], "b_magnitude_g": field["b_magnitude_g"],
            "D_mhz": field["D_mhz"], "reduced_chi2": fit["reduced_chi2"], "quality_gate_passed": True, "summary": "fake analyst",
        }
    else:
        check = tools.check_field_consistency()
        verdict = "revise" if first_round else "accept"
        out = {
            "verdict": verdict, "issues": check["quality_gate_issues"],
            "requests_for_planner": ["measure more points around every line"] if first_round else [],
            "requests_for_analyst": [], "summary": f"fake reviewer: {verdict}",
        }
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done",
                      "structured_output": out, "total_cost_usd": 0.0, "num_turns": 1, "session_id": "fake"}))


if __name__ == "__main__":
    main()
