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

# Our own p99 ceiling. Was 0.30 s, which measured p99 turn times of 0.25 *ms*
# against a bound 1,200x larger -- an assertion that could not detect a 100x
# slowdown, i.e. exactly the regime a forward-sim planner enters. Tightened to
# 50 ms on 2026-08-17: still 200x the observed p99, so it is not flaky on a
# loaded laptop, but a real regression now fails instead of passing silently.
TIME_BUDGET = 0.05

# The reciprocal number, reported rather than asserted. We use ~0.03% of the
# per-turn allowance while the field uses ~0.0006%, and the top of the ladder
# replays frozen open-loop sequences with no lookahead at all. Nothing in the
# suite made that visible, so nobody costed the unused budget (issue #10).
UTILISATION_NOTE = "p99 as a fraction of the 1 s/turn actTimeout"


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
@pytest.mark.parametrize("opponent", ["pass", "tape:band-vishnu"])
def test_per_turn_time_within_budget(opponent, capsys):
    """Measure decision time only, excluding the engine's own step cost.

    Run against a real ladder tape as well as `pass`. `pass` is the cheapest
    possible opponent: it never trades, so the market stays near base price and
    our shed, tile count and task list all stay small. A strong opponent is the
    worst case -- against one of these the market collapses (milk 160 -> 7,
    melon 250 -> 31), which is where price-scanning and sale planning do the
    most work.
    """
    from sim.tape import load as load_tape

    ns = runpy.run_path(MAIN)
    agent = ns["agent"]

    timings = []

    def timed(obs):
        t0 = time.perf_counter()
        out = agent(obs)
        timings.append(time.perf_counter() - t0)
        return out

    opp = (load_tape(opponent[len("tape:"):])
           if opponent.startswith("tape:") else opponent)
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 5})
    env.run([timed, opp])

    worst = max(timings)
    p99 = sorted(timings)[int(len(timings) * 0.99)]
    with capsys.disabled():
        print(f"\n  vs {opponent}: p99 {p99*1000:.2f}ms  worst {worst*1000:.2f}ms  "
              f"{UTILISATION_NOTE} {p99 / ACT_TIMEOUT:.4%}")
    assert p99 < TIME_BUDGET, f"p99 turn time {p99*1000:.1f}ms over budget"
    assert worst < ACT_TIMEOUT, f"worst turn {worst*1000:.1f}ms would time out"


def test_hand_targets_above_ten_are_reachable():
    """Regression for the silent hiring cap (issue #34).

    Market orders truncate at MAX_MARKET_ORDERS (10) and hiring used to run only
    at hour 0, so every target above 10 produced exactly 10 hands. Targets of
    12, 14 and 16 gave byte-identical episodes, which made three search
    dimensions dead above 10.

    This test used to *also* assert `hires_at_hour_0 == MAX_MARKET_ORDERS`, i.e.
    that hour 0 spends the entire order budget on HIRE. That assertion was
    removed on 2026-08-17: docs/6 measured the resulting crowd-out at -37,000 of
    margin (`hands_late` 12/15/18 score identically because HIRE saturates the
    budget before SELL is reached), so the test was pinning behaviour we have
    since measured as harmful. Reserving order slots for SELL is on the roadmap
    and must not be blocked by a test.
    """
    import sys as _sys
    from dataclasses import replace

    _sys.path.insert(0, os.path.join(REPO, "agent"))
    from params import Params
    from policy import Policy

    p = replace(Params(), hands_early=16, hands_mid=16, hands_late=16)
    policy = Policy(p)

    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 2})
    env.reset(2)
    obs = env.steps[0][0].observation

    assert sum(1 for o in policy.act(obs)["market"] if o[0] == "HIRE") > 0, (
        "no hiring at all on hour 0"
    )

    # Hour 1 with 10 already hired must issue the remainder, not zero.
    obs["hour"] = 1
    obs["farms"][0]["hires_today"] = 10
    obs["farms"][0]["money"] = 100_000
    hires_at_hour_1 = sum(1 for o in policy.act(obs)["market"] if o[0] == "HIRE")
    assert hires_at_hour_1 > 0, (
        "targets above 10 are unreachable: hiring stopped after hour 0"
    )


def test_low_priority_crops_are_not_starved():
    """Regression for _wanted_crop returning the first crop under target.

    One-time crops vacate their tile on harvest, so a cycling crop sat under
    target permanently and monopolised every free tile. A champion configured
    for 26 carrot tiles planted tomato zero times across a real season.

    Narrowed on 2026-08-17. The original version also asserted that with melon
    at 9/10 and carrot at 0/26 the winner must NOT be melon, which hardcodes
    pure relative-shortfall ranking and forbids price from deciding -- and
    `crop_price_elasticity` exists precisely so that it can. Melon is also our
    largest earner (cutting it costs -49,283 of margin), so a value-weighted
    rule legitimately picking it is not a bug. What remains is the actual
    starvation guard: a SATISFIED crop must not win over unplanted ones.
    """
    import sys as _sys
    from dataclasses import replace

    _sys.path.insert(0, os.path.join(REPO, "agent"))
    from params import Params
    from policy import Policy

    p = replace(Params(), target_wheat_tiles=0, target_melon_tiles=10,
                target_carrot_tiles=26, target_tomato_tiles=8,
                target_strawberry_tiles=0)
    policy = Policy(p)
    counts = {"GOOSE": 0, "COW": 0, "SHEEP": 0, "COOP": 0, "PASTURE": 0,
              "empty": 30, "weeds": 0}

    # Melon at target; carrot and tomato completely unplanted. A crop already at
    # its target must not take the tile.
    crops = {"WHEAT": 0, "MELON": 10, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0}
    assert policy._wanted_crop(counts, crops, day=5, scale=1) in ("CARROT", "TOMATO")

    # Melon genuinely empty and everything else satisfied: melon must win.
    crops = {"WHEAT": 0, "MELON": 0, "CARROT": 26, "TOMATO": 8, "STRAWBERRY": 0}
    assert policy._wanted_crop(counts, crops, day=5, scale=1) == "MELON"
