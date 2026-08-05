"""Pin the champion's score on fixed seeds.

Nothing else in the suite fails if a refactor quietly makes the agent worse.
Parity, contract and timing all pass for an agent that has become 20% less
profitable, and 20% is not hypothetical: fixing two bugs in PR #35 moved the
then-champion from 82,759 to 66,620, and that was noticed only because it was
being measured by hand at the time.

Episodes are deterministic given a seed, so the only things that can move these
numbers are our own code, the tuned parameters, or the pinned engine. All three
are exactly what we want to be told about.

When a change is intentional, re-record with:

    python -m tests.record_scores
"""

import json
import os

import pytest

from params import Params
from sim.harness import play, make_agent

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXPECTED_PATH = os.path.join(HERE, "expected_scores.json")

# Generous enough to absorb nothing (episodes are deterministic), tight enough
# that any real behavioural change trips it. Non-zero only to survive float
# formatting differences across platforms.
TOLERANCE = 0.005


def load_expected():
    with open(EXPECTED_PATH) as f:
        return json.load(f)


@pytest.mark.slow
@pytest.mark.parametrize("seed", sorted(load_expected()))
def test_champion_score_is_unchanged(seed):
    params_path = os.path.join(REPO, "agent", "params.json")
    if not os.path.exists(params_path):
        pytest.skip("no tuned params.json; nothing to pin")

    expected = load_expected()[seed]
    result = play(make_agent(Params.from_json(params_path)), "starter",
                  seed=int(seed), steps=720)
    actual = result["banks"][0]

    drift = abs(actual - expected) / expected
    assert drift <= TOLERANCE, (
        f"seed {seed}: bank moved {expected:,.0f} -> {actual:,.0f} "
        f"({(actual - expected) / expected:+.1%}). If this change is intended, "
        f"re-record with `python -m tests.record_scores`."
    )
