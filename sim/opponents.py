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


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Recorded ladder agents, replayed from their action tapes (see sim/tape.py).
# The frozen Params pool is entirely our own lineage, so "beats every opponent
# in the pool" only ever meant "beats earlier versions of itself". Measured
# against the real thing, v8 lost all five head-to-heads and banked about half.
TAPE_PREFIX = "tape:"

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opponents")


def tape_names():
    from sim import tape
    return tape.names()


def _age_key(name):
    """Sort key that puts v2 before v10.

    Plain `sorted()` is lexicographic, so `v10-wheatfix` sorts *second*, right
    after `v1-warmbase`. That is not cosmetic: `resolve_pool("real")` took
    `frozen_names()[-2:]` and called it "our two most recent snapshots", so from
    the moment v10 was frozen the reigning champion was silently excluded from
    every `--opponents real` evaluation -- including v10's own promotion gate,
    which read 13 sigma and lost 108 points of rating.

    Keyed on the leading integer of the name, with `_frozen_at` (written by
    `freeze()` from now on) taking precedence when present, because the naming
    convention is free text and will eventually drift.
    """
    stamp = _frozen_at(name)
    n = ""
    for ch in name.lstrip("v"):
        if not ch.isdigit():
            break
        n += ch
    return (stamp or "", int(n) if n else 0, name)


def _frozen_at(name):
    try:
        with open(os.path.join(POOL_DIR, f"{name}.json")) as fh:
            return json.load(fh).get("_frozen_at")
    except (OSError, ValueError):
        return None


def frozen_names():
    """Names of every snapshotted opponent, oldest first.

    "Oldest first" is load-bearing -- see `_age_key`. It was previously
    lexicographic and the docstring claimed this anyway.
    """
    if not os.path.isdir(POOL_DIR):
        return []
    names = [f[:-5] for f in os.listdir(POOL_DIR) if f.endswith(".json")]
    return sorted(names, key=_age_key)


def champion_name():
    """The reigning champion, from the tracked CHAMPION pointer file.

    A pointer file rather than "the newest snapshot" because the two came apart:
    v11 and v12 were submitted and never frozen, so on 2026-08-06 the newest
    snapshot was not what was actually deployed. Whatever judges a candidate has
    to be named explicitly.
    """
    path = os.path.join(REPO, "CHAMPION")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if line:
                return line
    return None


def champion_params():
    """The champion as a Params instance. Raises if the pointer is unusable."""
    from params import Params

    name = champion_name()
    if name is None:
        raise ValueError(
            "no CHAMPION file at the repo root. Gating against nothing means "
            "gating against Params() dataclass defaults, which any candidate "
            "beats trivially -- write the reigning snapshot's name into it.")
    path = os.path.join(POOL_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise ValueError(
            f"CHAMPION names {name!r} but {path} does not exist; "
            f"pool has {frozen_names()}")
    return Params.from_json(path)


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
    # Recorded so pool ordering does not depend on the `vN-` naming convention
    # holding forever; see `_age_key`.
    payload["_frozen_at"] = _now()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


# Tape naming carries the rating band the tape was cut from, because which band
# an opponent came from is the single most important thing about it: the same
# agent banks 102,773 and wins 100% against sub-750 opponents and banks 74,752
# and wins 20% against 1000+.
#
#   band-*  our own matchmaking band, ~840-1050 rated
#   top-*   the prize band, ~2900-3250 rated (minted by `sim.ladder mine`)
#   meta-*  the 2026-08-05 top of the ladder, when the leader was 3047
BAND_PREFIXES = {"band": ("band-",), "top": ("top-", "meta-")}


def tapes_in_band(band):
    prefixes = BAND_PREFIXES[band]
    return [t for t in tape_names() if t.startswith(prefixes)]


def resolve_pool(spec):
    """Turn a comma-separated spec into a list of opponents.

    Accepts built-in names, frozen snapshot names, recorded ladder tapes as
    `tape:<name>`, and these shorthands:

    | spec | members |
    |---|---|
    | `all` | built-ins plus every snapshot |
    | `frozen` | snapshots only |
    | `tapes` | every recorded ladder agent |
    | `band` | tapes from our own matchmaking band, ~840-1050 rated |
    | `top` | tapes from the prize band, ~2900-3250 rated |
    | `real` | every tape plus our two most recent snapshots |
    | `champion` | just the reigning champion, per the CHAMPION file |

    `top` is the spec that matters and the one that did not exist until
    2026-08-17. The frozen pool is our own lineage end to end, so beating all of
    it means beating earlier versions of ourselves: v8 did that 100% of the time
    and lost every head-to-head against a recorded ladder agent. But `band` has
    a ceiling too -- the champion goes 3W-3L against it, which is precisely the
    746 rating it earned. Optimising against a 900-rated band cannot produce an
    agent that beats a 3200-rated one, and against `top` we currently go 0W-4L
    while selling a fifth of the units they do.
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
        elif name in BAND_PREFIXES:
            members = tapes_in_band(name)
            if not members:
                raise ValueError(
                    f"pool {name!r} is empty: no tape in sim/tapes/ matches "
                    f"{BAND_PREFIXES[name]}. Run `make refresh-tapes` to mint "
                    f"them from the latest daily episode dataset.")
            out += [TAPE_PREFIX + t for t in members]
        elif name == "champion":
            out.append(champion_name())
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
