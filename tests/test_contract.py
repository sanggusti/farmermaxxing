"""Guard the submission contract and the per-turn time budget.

If any of these fail, the submission errors out on the ladder rather than
scoring badly -- which costs a day of rating convergence to discover.
"""

import os
import runpy
import time

import pytest

from kaggle_environments import make

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO, "agent", "main.py")

ACT_TIMEOUT = 1.0        # kaggriculture.json: one second per turn
TIME_BUDGET = 0.30       # our own ceiling, on a 2 vCPU submission box


def test_agent_is_the_last_callable():
    """kaggle_environments takes the LAST callable defined in the file.

        return [v for v in env.values() if callable(v)][-1]

    Kaggle's own harness instead wants a function literally named `agent`.
    Defining `agent` last satisfies both; importing anything after it breaks
    the first without breaking the second, which would be a confusing failure.
    """
    ns = runpy.run_path(MAIN)
    callables = [(k, v) for k, v in ns.items() if callable(v)]
    assert callables[-1][0] == "agent", (
        f"last callable is {callables[-1][0]!r}, not 'agent' -- "
        "the local file loader would pick the wrong function"
    )


def test_agent_survives_a_full_episode():
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3},
               debug=True)
    env.run([MAIN, "starter"])
    final = env.steps[-1]
    assert final[0].status == "DONE", f"agent ended {final[0].status}"
    assert final[0].reward is not None


def test_action_shape_is_valid():
    ns = runpy.run_path(MAIN)
    agent = ns["agent"]

    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 1})
    env.reset(2)
    obs = env.steps[0][0].observation
    action = agent(obs)

    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["farmer"], list) and action["farmer"]
    assert isinstance(action["hands"], list)
    assert isinstance(action["market"], list)
    assert len(action["market"]) <= 10, "market orders past 10 are silently dropped"


@pytest.mark.slow
def test_per_turn_time_within_budget():
    """Measure decision time only, excluding the engine's own step cost."""
    ns = runpy.run_path(MAIN)
    agent = ns["agent"]

    timings = []

    def timed(obs):
        t0 = time.perf_counter()
        out = agent(obs)
        timings.append(time.perf_counter() - t0)
        return out

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 5})
    env.run([timed, "pass"])

    worst = max(timings)
    p99 = sorted(timings)[int(len(timings) * 0.99)]
    assert p99 < TIME_BUDGET, f"p99 turn time {p99*1000:.1f}ms over budget"
    assert worst < ACT_TIMEOUT, f"worst turn {worst*1000:.1f}ms would time out"
