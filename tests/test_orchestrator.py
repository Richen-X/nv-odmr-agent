"""Orchestrator: role wiring, the feedback loop and scoring, with a stand-in for `claude -p`."""
import json
import sys
from pathlib import Path

import pytest

from nvagent import tools
from nvagent.orchestrator import AgentConfig, run_experiment, score, write_role_config
from nvagent.runstate import RunState
from nvagent.simulator import Sample

FAKE_CLAUDE = Path(__file__).with_name("fake_claude.py").resolve()


def test_closed_loop_runs_all_roles_and_scores_against_truth(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")  # as when launched from inside a Claude Code session
    report = run_experiment(seed=4, cfg=AgentConfig(claude_cmd=[sys.executable, str(FAKE_CLAUDE)]), runs_dir=tmp_path)
    assert [a["tag"] for a in report["agents"]] == [
        "r1-planner", "r1-analyst", "r1-reviewer", "r2-planner", "r2-analyst", "r2-reviewer",
    ]
    assert all(a["ok"] for a in report["agents"])
    assert report["accepted"]
    s = report["score"]
    assert s["n_lines"] == 8
    assert s["field_error_g"] < 0.1
    run_dir = Path(report["run_dir"])
    assert "measure more points" in (run_dir / "agents" / "r2-planner.prompt.txt").read_text()
    assert (run_dir / "report.md").read_text().startswith("# Run seed 4")


def test_only_the_analyst_gets_the_quality_gate(tmp_path):
    state = RunState.create(tmp_path / "run", Sample.random(seed=1))
    for role in ("planner", "analyst", "reviewer"):
        mcp_path, settings_path = write_role_config(state.dir, role)
        server = json.loads(mcp_path.read_text())["mcpServers"]["nvlab"]
        hooks = json.loads(settings_path.read_text())["hooks"]
        assert server["args"][server["args"].index("--role") + 1] == role
        assert hooks["PreToolUse"][0]["matcher"] == "mcp__nvlab__.*"
        assert ("Stop" in hooks) == (role == "analyst")


def test_a_hung_agent_times_out_and_the_report_is_still_written(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "60")
    cfg = AgentConfig(claude_cmd=[sys.executable, str(FAKE_CLAUDE)], timeout_s=3)
    report = run_experiment(seed=4, cfg=cfg, runs_dir=tmp_path)
    assert [a["tag"] for a in report["agents"]] == ["r1-planner"]
    assert not report["agents"][0]["ok"] and "timed out" in report["agents"][0]["error"]
    assert (Path(report["run_dir"]) / "report.json").exists()


def test_windows_batch_launchers_are_refused(tmp_path):
    with pytest.raises(ValueError, match="cmd"):
        run_experiment(seed=4, cfg=AgentConfig(claude_cmd=[str(tmp_path / "claude.cmd")]), runs_dir=tmp_path)


def test_scoring_ignores_a_field_fit_id_that_names_an_odmr_fit(run_state):
    tools.measure_odmr_sweep(2640.0, 3100.0, 0.25, 0.0)
    odmr = tools.fit_odmr()
    tools.fit_field_vector(odmr["fit_id"])
    result = score(run_state, odmr["fit_id"])
    assert result["scored"] and result["field_fit_id"] != odmr["fit_id"]
