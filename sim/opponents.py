"""The opponent pool used for evaluation.

The built-in agents are a very low bar: `starter` farms a single carrot tile,
never hires a hand and never buys land. Beating them says the agent runs, not
that it is competitive. Real gating needs to be against our own promoted
versions, so each accepted params.json is frozen into `sim/opponents/` and
becomes a permanent member of the pool.

Keeping every past champion, rather than only the latest, matters because the
final leaderboard is a single Bradley-Terry tournament over many episodes. A
candidate that beats the current best but collapses against an older archetype
is a regression, and a pool of one cannot show that.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The agent modules are flat in agent/ because that is how Kaggle unpacks a
# submission. Set the path here rather than relying on sim.harness having been
# imported first, so `python -m sim.opponents` works on its own.
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))

BUILTIN = ("starter", "pass", "random")

# Recorded ladder agents, replayed from their action tapes (see sim/tape.py).
# The frozen Params pool is entirely our own lineage, so "beats every opponent
# in the pool" only ever meant "beats earlier versions of itself". Measured
# against the real thing, v8 lost all five head-to-heads and banked about half.
TAPE_PREFIX = "tape:"

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponents")


def tape_names():
    from sim import tape
    return tape.names()


def frozen_names():
    """Names of every snapshotted opponent, oldest first."""
    if not os.path.isdir(POOL_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(POOL_DIR) if f.endswith(".json"))


def load(name):
    """A frozen opponent by name, as a Params instance."""
    from params import Params

    return Params.from_json(os.path.join(POOL_DIR, f"{name}.json"))


def freeze(params, name, notes=None):
    """Snapshot `params` into the pool under `name`."""
    from dataclasses import asdict

    os.makedirs(POOL_DIR, exist_ok=True)
    path = os.path.join(POOL_DIR, f"{name}.json")
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; pick a new name")

    payload = asdict(params)
    if notes:
        payload["_notes"] = notes
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def resolve_pool(spec):
    """Turn a comma-separated spec into a list of opponents.

    Accepts built-in names, frozen snapshot names, recorded ladder tapes as
    `tape:<name>`, and the shorthands `all` (built-ins plus snapshots),
    `frozen` (snapshots only), `tapes` (recorded ladder agents only), and
    `real` (tapes plus our two most recent snapshots).

    `real` exists because it is the only spec that answers the question that
    matters. The frozen pool is our own lineage end to end, so beating all of
    it means beating earlier versions of ourselves; v8 did that 100% of the
    time and still lost every head-to-head against a recorded ladder agent.
    """
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for name in names:
        if name == "all":
            out += list(BUILTIN) + frozen_names()
        elif name == "frozen":
            out += frozen_names()
        elif name == "tapes":
            out += [TAPE_PREFIX + t for t in tape_names()]
        elif name == "real":
            out += [TAPE_PREFIX + t for t in tape_names()] + frozen_names()[-2:]
        else:
            out.append(name)

    resolved = []
    for name in out:
        if name in BUILTIN:
            resolved.append(name)
        elif name.startswith(TAPE_PREFIX):
            # Passed through as the NAME. `sim.fastplay._resolve` loads it
            # where the episode actually runs, so the Modal fan-out ships a
            # short string rather than ~126 KB of actions per episode --
            # 1.16 GB per generation at population 384.
            from sim import tape
            if name[len(TAPE_PREFIX):] not in tape.names():
                raise ValueError(f"unknown tape {name!r}; have {tape.names()}")
            resolved.append(name)
        elif name in frozen_names():
            resolved.append(load(name))
        else:
            raise ValueError(
                f"unknown opponent {name!r}; built-ins {BUILTIN}, "
                f"frozen {frozen_names()}, tapes {tape_names()}"
            )
    return resolved, out


def main():
    """Freeze the current agent/params.json into the pool.

        python -m sim.opponents --name v1-cem --notes "first Modal search"
    """
    import argparse
    from params import Params

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="snapshot name, e.g. v1-cem")
    ap.add_argument("--params", default=os.path.join(REPO, "agent", "params.json"))
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    path = freeze(Params.from_json(args.params), args.name, args.notes)
    print(f"frozen -> {path}")
    print(f"pool now: {frozen_names()}")


if __name__ == "__main__":
    main()
