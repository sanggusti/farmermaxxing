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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))


def fast_play(agent_a, agent_b, seed=0, steps=720):
    """Play one episode and return {'banks', 'winner', 'statuses'}.

    Same return shape as `sim.harness.play` minus the `env`, which cannot be
    provided because no step history is kept. That omission is the entire point.
    """
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    state = env.state
    # Built-in opponents arrive as names ("starter", "pass", "random"). Passing
    # the string straight through makes every call raise, which the loop below
    # turns into a silent no-op opponent, so the episode still finishes and the
    # numbers look plausible while the opponent never plays.
    agents = [engine.agents.get(a, a) if isinstance(a, str) else a
              for a in (agent_a, agent_b)]

    # Name-mangled because core.py has no public accessor. This merges the
    # fields marked `shared` in the env spec (farms, market, town) from state[0]
    # into each agent's own observation, which is what an agent actually sees.
    shared = env._Environment__get_shared_state

    for step in range(steps - 1):
        actions = []
        for i in range(2):
            obs = shared(i).observation
            try:
                actions.append(agents[i](obs))
            except Exception:
                # Mirror core.py: a raising agent is ERROR, not a crash.
                state[i].status = "ERROR"
                actions.append(None)

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
    return {"seed": seed, "banks": banks, "winner": winner,
            "statuses": [s.status for s in state]}
