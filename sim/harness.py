"""Shared helpers for running episodes locally.

Everything that needs to play a game -- run.py, arena.py, the Modal workers --
goes through here so there is exactly one definition of "play an episode".
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent"))

from params import Params      # noqa: E402
from policy import Policy      # noqa: E402

BUILTIN = ("pass", "random", "starter")


def make_agent(params=None):
    """An agent callable for the given params. Must take exactly one arg."""
    policy = Policy(params or Params())

    def _agent(obs):
        return policy.act(obs)

    return _agent


def resolve(spec):
    """'starter' -> the builtin name; a Params -> a callable; a path -> itself.

    `tape:<name>` is loaded here too, mirroring `sim.fastplay._resolve`. Tapes
    travel through the pool as short strings so the Modal fan-out ships ~10
    bytes per episode instead of ~126 KB, but the slow path is what anything
    needing a step history uses (`sim.mix`, `sim.arena --fast False`), and it
    used to hand `"tape:meta-a"` straight to kaggle_environments as a filename.
    """
    if isinstance(spec, Params):
        return make_agent(spec)
    if isinstance(spec, str) and spec.startswith("tape:"):
        from sim.tape import load as load_tape
        return load_tape(spec[len("tape:"):])
    return spec


def play(a, b, seed=0, steps=720, debug=False):
    """Run one episode. Returns banks, the winner, and each agent's status."""
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed},
        debug=debug,
    )
    env.run([resolve(a), resolve(b)])

    final = env.steps[-1]
    banks = [float(s.reward) if s.reward is not None else float("nan") for s in final]
    statuses = [s.status for s in final]

    if banks[0] == banks[1]:
        winner = -1               # tie
    else:
        winner = 0 if banks[0] > banks[1] else 1

    return {
        "seed": seed,
        "banks": banks,
        "winner": winner,
        "statuses": statuses,
        "env": env,
    }
