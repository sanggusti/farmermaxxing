"""fast_play must produce byte-identical results to env.run().

It exists to skip kaggle_environments' per-step bookkeeping (deepcopy for the
replay history, schema re-validation, re-structify, stdout capture), which
profiling showed dominates episode cost. It calls the engine's own
`interpreter()`, so the game logic is the same code that scores the ladder, and
these tests are what make that claim checkable rather than assumed.

`random` is deliberately excluded: `random_agent` draws from the unseeded global
`random` module, so its action sequence depends on how many draws the runner
itself performed. That is a property of the opponent, not a divergence in game
logic, and no search uses it.
"""

import pytest

from params import Params
from sim.harness import make_agent, play
from sim.fastplay import fast_play

DETERMINISTIC_OPPONENTS = ["pass", "starter"]


@pytest.mark.slow
@pytest.mark.parametrize("opponent", DETERMINISTIC_OPPONENTS)
@pytest.mark.parametrize("seed", [0, 7, 20000])
def test_matches_env_run_exactly(opponent, seed):
    params = Params()
    slow = play(make_agent(params), opponent, seed=seed, steps=720)
    fast = fast_play(make_agent(params), opponent, seed=seed, steps=720)

    assert fast["banks"] == slow["banks"], (
        f"{opponent} seed {seed}: fast_play {fast['banks']} != "
        f"env.run {slow['banks']}"
    )
    assert fast["winner"] == slow["winner"]


@pytest.mark.slow
def test_matches_with_tuned_params():
    """The champion exercises far more of the engine than defaults do."""
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "agent", "params.json")
    if not os.path.exists(path):
        pytest.skip("no tuned params to check")

    params = Params.from_json(path)
    slow = play(make_agent(params), "starter", seed=20001, steps=720)
    fast = fast_play(make_agent(params), "starter", seed=20001, steps=720)
    assert fast["banks"] == slow["banks"]


def test_short_episode_matches():
    """Cheap enough to run on every commit, not just under -m slow."""
    params = Params()
    slow = play(make_agent(params), "pass", seed=3, steps=120)
    fast = fast_play(make_agent(params), "pass", seed=3, steps=120)
    assert fast["banks"] == slow["banks"]


@pytest.mark.slow
def test_frozen_params_opponent_actually_plays():
    """A Params opponent must be resolved, not silently passed through.

    fast_play resolved built-in names but let a Params instance through
    uncalled, so every frozen champion became a no-op: the episode finished,
    the numbers looked plausible, and the opponent never played. Measured at the
    time, same seed: env.run gave [39,969, 15,949] and fast_play [118,385, 3,000].

    The tell is the opponent banking its 3,000 starting money untouched.
    """
    import os

    from sim.harness import play

    pool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sim", "opponents")
    snapshots = sorted(f for f in os.listdir(pool)) if os.path.isdir(pool) else []
    if not snapshots:
        pytest.skip("no frozen opponents to check against")

    opponent = Params.from_json(os.path.join(pool, snapshots[0]))
    slow = play(make_agent(Params()), opponent, seed=20000, steps=720)
    fast = fast_play(make_agent(Params()), opponent, seed=20000, steps=720)

    assert fast["banks"] == slow["banks"], (
        f"fast_play {fast['banks']} != env.run {slow['banks']}; "
        "the frozen opponent is probably not being resolved"
    )
    assert fast["banks"][1] != 3000.0, "opponent never acted"


def test_unresolvable_opponent_raises_rather_than_going_quiet():
    """Failing loudly beats a plausible-looking wrong number."""
    with pytest.raises(TypeError, match="not callable"):
        fast_play(make_agent(Params()), object(), seed=0, steps=48)
