import pytest

from nvagent import tools
from nvagent.runstate import RunState
from nvagent.simulator import Sample


@pytest.fixture
def run_state(tmp_path):
    """A fresh run directory with a hidden sample, wired into the tool layer."""
    state = RunState.create(tmp_path / "run", Sample.random(seed=4), goal="test", budget_points=3000)
    tools.init(state.dir)
    yield state
    tools.STATE = None
