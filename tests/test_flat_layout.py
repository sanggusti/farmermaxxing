"""The submission runs flat, from a foreign working directory.

Kaggle unpacks a submission into /kaggle_simulations/agent/ with no package
around it, and kaggle_environments execs main.py with empty globals, so there is
no __file__. Issue #2 was exactly this and its symptom is a failed validation
episode on the ladder, not a bad score, which costs a submission slot and a day
of rating convergence to diagnose.

test_contract.py runs agent/main.py in place from the repo root, where `agent/`
is importable and the cwd is helpful. That cannot catch a regression which only
appears in the real layout, so this builds the layout and runs from elsewhere.
"""

import os
import shutil

import pytest

from kaggle_environments import make

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_FILES = ["main.py", "policy.py", "params.py", "market.py", "rules.py"]


@pytest.fixture
def flat_agent(tmp_path):
    """A copy of the submission, flat, exactly as Kaggle would unpack it."""
    for name in AGENT_FILES:
        shutil.copy(os.path.join(REPO, "agent", name), tmp_path / name)
    params = os.path.join(REPO, "agent", "params.json")
    if os.path.exists(params):
        shutil.copy(params, tmp_path / "params.json")
    return tmp_path


def test_runs_from_a_foreign_cwd(flat_agent, tmp_path, monkeypatch):
    """Run from a directory that is neither the repo nor the agent directory."""
    elsewhere = tmp_path.parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)

    env = make("kaggriculture",
               configuration={"episodeSteps": 48, "seed": 11}, debug=True)
    env.run([str(flat_agent / "main.py"), "pass"])

    final = env.steps[-1]
    assert final[0].status == "DONE", (
        f"flat submission failed with {final[0].status}; "
        "this is what a validation-episode failure looks like"
    )


def test_no_sibling_package_is_required(flat_agent):
    """There is no agent/ package in the flat layout, so imports must be flat."""
    assert not (flat_agent / "__init__.py").exists()
    source = (flat_agent / "main.py").read_text()
    assert "from ." not in source, "relative imports break in the flat layout"


def test_params_json_is_picked_up_from_beside_main(flat_agent, tmp_path, monkeypatch):
    """A tuned params.json next to main.py must be loaded, not silently ignored.

    Silently falling back to defaults would mean submitting an untuned agent
    while every local number said otherwise.
    """
    import json

    (flat_agent / "params.json").write_text(json.dumps({"target_geese": 3}))
    monkeypatch.chdir(tmp_path.parent)

    import runpy
    ns = runpy.run_path(str(flat_agent / "main.py"))
    assert ns["_PARAMS"].target_geese == 3
