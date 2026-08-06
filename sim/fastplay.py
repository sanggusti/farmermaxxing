"""Run episodes without kaggle_environments' per-step bookkeeping.

Profiling a single 720-turn episode showed 30.7M function calls, dominated by
`copy.deepcopy` (~2.4s cumulative) and `structify` (~1.6s). None of that is game
logic. `core.py` deep-copies the whole state every step to build the replay
history, re-validates the action against a JSON schema, re-structifies the
state, and wraps each step in stdout/stderr capture. We need none of it to score
a candidate; we only want the final bank.

This calls the engine's own `interpreter()` directly on a state we hold. The
game logic is therefore byte-identical to the one that scores the ladder, which
is the property a hand-written simulator would give up. `tests/test_fastplay.py`
asserts the final banks match `env.run()` exactly across seeds.

Use `sim.harness.play` when you need a replay or per-step history; use this when
you only need the score, which is every search episode.
"""

import os
import sys

from sim.census import EpisodeCensus, TURNS_PER_DAY

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))


def _resolve(spec, engine):
    """Built-in name, Params, or callable -> callable.

    Anything else raises. make_agent() would happily wrap an arbitrary object
    and hand back a callable that only fails once the episode is running, which
    is the failure mode this whole function exists to prevent.
    """
    from params import Params
    from sim.harness import make_agent

    if isinstance(spec, str):
        # Recorded ladder tapes travel as a NAME, never as an object. A tape
        # pickles to ~126 KB, and the Modal fan-out ships one opponent per
        # episode -- at population 384 that is 1.16 GB of payload per
        # generation. sim/tapes/ is inside the mounted sim/ directory, so a
        # worker can load it locally from a short string instead.
        if spec.startswith("tape:"):
            from sim.tape import load as load_tape
            return load_tape(spec[len("tape:"):])

        resolved = engine.agents.get(spec)
        if resolved is None:
            raise TypeError(
                f"unknown built-in agent {spec!r}; "
                f"expected one of {sorted(engine.agents)}"
            )
        return resolved
    if isinstance(spec, Params):
        return make_agent(spec)
    if callable(spec):
        return spec
    raise TypeError(
        f"opponent is {type(spec).__name__}, not callable, a Params or a "
        "built-in agent name. An unresolved opponent silently does nothing "
        "for the whole episode instead of erroring."
    )


def fast_play(agent_a, agent_b, seed=0, steps=720, metrics=False):
    """Play one episode and return {'banks', 'winner', 'statuses'}.

    Same return shape as `sim.harness.play` minus the `env`, which cannot be
    provided because no step history is kept. That omission is the entire point.

    `metrics=True` additionally returns 'metrics': a per-player dict of land and
    action statistics (see sim.census). It is off by default because every
    search episode goes through here and the sampling, though cheap, is not
    free -- and because the banks must be provably unaffected by asking for it.
    """
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    state = env.state
    # Opponents arrive as built-in names ("starter"), Params instances (frozen
    # champions), or callables. Anything not resolved to a callable raises on
    # every turn, which the loop below turns into a silent no-op opponent: the
    # episode finishes, the numbers look plausible, and the opponent never
    # played. That happened to every frozen Params opponent (issue #56).
    agents = [_resolve(a, engine) for a in (agent_a, agent_b)]

    # Name-mangled because core.py has no public accessor. This merges the
    # fields marked `shared` in the env spec (farms, market, town) from state[0]
    # into each agent's own observation, which is what an agent actually sees.
    shared = env._Environment__get_shared_state

    census = EpisodeCensus(2) if metrics else None

    for step in range(steps - 1):
        actions = []
        for i in range(2):
            obs = shared(i).observation
            if census is not None and i == 0 and step % TURNS_PER_DAY == 0:
                # Sample once per day, from the shared farms view, before any
                # action this turn. Both players come off player 0's obs.
                census.sample_day(obs["farms"], obs["day"])
            try:
                actions.append(agents[i](obs))
            except Exception:
                # Mirror core.py: a raising agent is ERROR, not a crash.
                state[i].status = "ERROR"
                actions.append(None)

        if census is not None:
            for i in range(2):
                census.record_action(i, actions[i])

        # Mutate in place rather than rebuilding and re-structifying the state
        # each step; structify was a large share of the remaining overhead.
        for i in range(2):
            state[i].action = actions[i]
            state[i]["action"] = actions[i]

        state = engine.interpreter(state, env)
        state[0].observation.step = step + 1
        # `shared()` reads env.state, not our local handle. Without this the
        # agent sees the step-0 observation for the whole episode and banks 0.
        env.state = state

        if all(s.status != "ACTIVE" for s in state):
            break

    banks = [float(s.reward) if s.reward is not None else float("nan")
             for s in state]
    winner = -1 if banks[0] == banks[1] else (0 if banks[0] > banks[1] else 1)
    out = {"seed": seed, "banks": banks, "winner": winner,
           "statuses": [s.status for s in state]}
    if census is not None:
        census.finish([state[i].observation["private"].get("shed", {})
                       for i in range(2)],
                      town=state[0].observation.get("town"))
        out["metrics"] = [census.result(i) for i in range(2)]
        out["shop_order"] = census.shop_order
    return out
