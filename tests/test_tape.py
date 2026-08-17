"""Recorded ladder agents must stay usable, and must fail loudly if not.

The frozen pool is our own lineage end to end, so "beats every opponent in the
pool" only ever meant "beats earlier versions of itself". v8 did that 100% of
the time and lost all five head-to-heads against real ladder agents, banking
about half. These tests guard the fix.
"""

import pickle

import pytest

from sim import tape
from sim.opponents import resolve_pool, TAPE_PREFIX


def test_tape_agent_is_picklable():
    """Modal pickles opponents to send them to workers.

    A closure would not survive that, and the failure would arrive as a worker
    crash mid-search rather than anything readable here.
    """
    a = tape.TapeAgent([None, {"farmer": ["PASS"], "hands": [], "market": []}])
    assert pickle.loads(pickle.dumps(a)).actions == a.actions


def _obs(day, hour):
    return {"day": day, "hour": hour}


def test_one_instance_replays_many_episodes_identically():
    """The bug this file exists for.

    `resolve_pool` builds ONE opponent instance and `arena.evaluate` reuses it
    across every seed and seat. With an internal turn counter the tape
    exhausted after the first episode and PASSed for the rest of the sweep --
    while the sweep still completed and still reported per-opponent banks, so a
    losing agent measured as an 87.5% winner. Indexing off the observation
    makes reuse impossible to get wrong.
    """
    acts = [None] + [{"farmer": [str(i)], "hands": [], "market": []}
                     for i in range(1, 100)]
    a = tape.TapeAgent(acts)

    first_episode = [a(_obs(0, h)) for h in range(5)]
    # A second episode through the SAME instance must replay from the start.
    for h in range(5):
        a(_obs(0, h))
    second_episode = [a(_obs(0, h)) for h in range(5)]
    assert first_episode == second_episode


def test_tape_replays_the_next_action_not_the_current_one():
    """Index t+1 is the action taken FROM the observation at index t.

    Off by one still yields a complete episode with plausible banks, so it has
    to be pinned rather than eyeballed.
    """
    first = {"farmer": ["NORTH"], "hands": [], "market": []}
    second = {"farmer": ["SOUTH"], "hands": [], "market": []}
    a = tape.TapeAgent([{"farmer": ["PASS"], "hands": [], "market": []},
                        first, second])
    assert a(_obs(0, 0)) == first
    assert a(_obs(0, 1)) == second


def test_tape_indexes_across_day_boundaries():
    acts = [None] * 30
    acts[25] = {"farmer": ["MARK"], "hands": [], "market": []}
    a = tape.TapeAgent(acts)
    # day 1 hour 0 is turn 24; the action taken from it is index 25.
    assert a(_obs(1, 0))["farmer"] == ["MARK"]


def test_tape_passes_rather_than_raising_past_the_end():
    a = tape.TapeAgent([None, {"farmer": ["NORTH"], "hands": [], "market": []}])
    assert a(_obs(0, 0))["farmer"] == ["NORTH"]
    assert a(_obs(29, 23)) == tape.PASS_ACTION


def test_pool_passes_tapes_through_as_names_not_objects():
    """Payload size, not style.

    A tape pickles to ~126 KB and the Modal fan-out ships one opponent per
    episode -- at population 384 that is 1.16 GB per generation, which stalls
    the fan-out. sim/tapes/ is inside the mounted sim/ directory, so the name
    is enough and the worker loads it locally.
    """
    import pickle

    names = tape.names()
    if not names:
        pytest.skip("no tapes recorded")
    spec = f"{TAPE_PREFIX}{names[0]}"
    opps, labels = resolve_pool(spec)
    assert labels == [spec]
    assert opps[0] == spec
    assert len(pickle.dumps(opps[0])) < 200


def test_fastplay_resolves_a_tape_name_to_a_working_agent():
    """The name only helps if the other end can turn it back into an agent."""
    from sim.fastplay import _resolve
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    names = tape.names()
    if not names:
        pytest.skip("no tapes recorded")
    agent = _resolve(f"{TAPE_PREFIX}{names[0]}", engine)
    assert callable(agent)
    assert agent({"day": 0, "hour": 0}) is not None


def test_unknown_tape_name_is_rejected_at_resolution():
    with pytest.raises(ValueError, match="unknown tape"):
        resolve_pool(f"{TAPE_PREFIX}not-a-real-tape")


def test_unknown_opponent_names_the_tapes_too():
    with pytest.raises(ValueError, match="tapes"):
        resolve_pool("definitely-not-an-opponent")


@pytest.mark.slow
def test_recorded_tapes_are_not_degenerate_on_unseen_seeds():
    """A tape that collapses off its own seed is not an opponent.

    The tell is a bank near the 3,000 starting money -- an unresolved or
    exhausted opponent still finishes the episode and still reports a number.
    """
    for name in tape.names():
        v = tape.verify_tape(name, seeds=(20000, 20001))
        assert v["ok"], f"{name} collapsed on unseen seeds: {v['banks']}"


def test_tape_accepts_the_engine_slow_path_calling_convention():
    """`kaggle_environments` passes (observation, configuration) to a class.

    `Agent.act` builds `[observation, configuration]` and truncates it only when
    the agent has a `__code__` attribute:

        if hasattr(self.agent, "__code__"): args = args[:co_argcount]

    A class instance has none, so `TapeAgent.__call__` receives TWO arguments on
    the slow path. When it took one, the TypeError was swallowed by `Agent.act`'s
    own `except Exception` into a no-op action and the seat still finished DONE:
    on seed 20000 vs meta-a, `fast_play` gave [105,504, 151,737] while
    `harness.play` gave [114,521, 3,000]. Nothing errored, nothing warned, and
    every number derived from the slow path was wrong.
    """
    names = tape.names()
    if not names:
        pytest.skip("no tapes recorded")
    agent = tape.load(names[0])
    obs = {"day": 0, "hour": 3}
    assert agent(obs) == agent(obs, {"episodeSteps": 720}), (
        "the tape behaves differently depending on how many arguments the "
        "harness passes it"
    )


@pytest.mark.slow
def test_the_fast_and_slow_paths_agree_on_a_tape_opponent():
    """Same episode, both harnesses, identical banks.

    `tests/test_fastplay.py` pins this for built-ins and for frozen `Params`
    opponents. Tapes were the one opponent kind not covered, and they were the
    one that silently disagreed.
    """
    import os
    import sys

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(REPO, "agent"))
    from params import Params
    from sim.fastplay import fast_play
    from sim.harness import play, make_agent

    p = Params.from_json(os.path.join(REPO, "agent", "params.json"))
    name = tape.names()[0]
    fast = fast_play(p, f"{TAPE_PREFIX}{name}", seed=20000)["banks"]
    slow = play(make_agent(p), f"{TAPE_PREFIX}{name}", seed=20000)["banks"]
    assert fast == slow, f"fast {fast} vs slow {slow} against {name}"
