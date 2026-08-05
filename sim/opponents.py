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

BUILTIN = ("starter", "pass", "random")

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponents")


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

    Accepts built-in names, frozen snapshot names, and the shorthand `all`
    (every built-in plus every snapshot) or `frozen` (snapshots only).
    """
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for name in names:
        if name == "all":
            out += list(BUILTIN) + frozen_names()
        elif name == "frozen":
            out += frozen_names()
        else:
            out.append(name)

    resolved = []
    for name in out:
        if name in BUILTIN:
            resolved.append(name)
        elif name in frozen_names():
            resolved.append(load(name))
        else:
            raise ValueError(
                f"unknown opponent {name!r}; "
                f"built-ins {BUILTIN}, frozen {frozen_names()}"
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
    ap.add_argument("--params", default=os.path.join(
        os.path.dirname(POOL_DIR), "..", "agent", "params.json"))
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    path = freeze(Params.from_json(args.params), args.name, args.notes)
    print(f"frozen -> {path}")
    print(f"pool now: {frozen_names()}")


if __name__ == "__main__":
    main()
