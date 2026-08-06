"""Read the ladder instead of guessing at it.

Local evaluation and the ladder disagree, and docs/4 states the open question
plainly: v4 banked 137,684 against `starter` and 46,454 against our own
strongest frozen opponent, and "which of those numbers is comparable to 170k is
not known". Everything downstream of that -- how much headroom is left, whether
a promotion will hold, what the field is actually doing -- is guesswork until it
is measured.

Three things make that measurement possible, and all three are verified:

  * A replay carries its seed at `info.seed`, so any ladder episode can be
    re-run locally against the same weed draws and the same market.
  * Both farms are public in `steps[t][0].observation.farms`, so a replay is a
    complete record of what the opponent did, not just what we saw.
  * The recorded actions for both seats are in `steps[t][seat].action`, so an
    episode can be replayed exactly -- which is the check that tells us the
    other two are being read correctly.

    python -m sim.ladder verify   --replay replays/episode-90070208-replay.json
    python -m sim.ladder census   --replay replays/*.json
    python -m sim.ladder fetch    --date 2026-08-05 --top 20
    python -m sim.ladder calibrate --params agent/params.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "agent"))

from sim.census import TURNS_PER_DAY, tile_stats            # noqa: E402

DATASET = "kaggle/kaggriculture-episodes-{date}"
INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
KAGGLE = os.path.join(REPO, ".venv", "bin", "kaggle")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def seed_of(replay):
    """The episode's RNG seed. `configuration.seed` is null; `info.seed` is not."""
    return replay.get("info", {}).get("seed")


def names_of(replay):
    return replay.get("info", {}).get("TeamNames") or ["?", "?"]


def recorded_agent(replay, seat):
    """A callable replaying one seat's recorded actions, by step index.

    Returns PASS past the end rather than raising, so a truncated replay
    degrades into a quiet agent instead of an ERROR status that would be
    mistaken for a crash.
    """
    actions = [step[seat].get("action") for step in replay["steps"]]
    state = {"t": 0}

    def agent(_obs):
        t = state["t"]
        state["t"] += 1
        # Action at index t is the one taken from the observation at index t.
        if t + 1 < len(actions) and actions[t + 1] is not None:
            return actions[t + 1]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def verify(path):
    """Re-run a replay from its recorded actions and check the banks match.

    This is the load-bearing test. If it passes, the seed, the action encoding
    and the step alignment are all being read correctly, and every other number
    this module produces can be trusted. If it fails, they cannot -- and the
    failure mode is silent: an off-by-one in step alignment still produces a
    complete episode with entirely plausible banks.
    """
    from sim.fastplay import fast_play

    replay = load(path)
    seed = seed_of(replay)
    expected = [float(x) for x in replay["rewards"]]
    steps = len(replay["steps"])

    r = fast_play(recorded_agent(replay, 0), recorded_agent(replay, 1),
                  seed=seed, steps=steps)
    got = r["banks"]
    ok = all(abs(a - b) < 1.0 for a, b in zip(got, expected))
    return {"path": path, "seed": seed, "expected": expected, "got": got,
            "match": ok, "names": names_of(replay)}


def census(path, days=(3, 9, 15, 21, 27, 29)):
    """Per-day land and portfolio census for both seats of a replay."""
    replay = load(path)
    names = names_of(replay)
    steps = replay["steps"]
    out = {"path": path, "seed": seed_of(replay), "names": names,
           "rewards": replay["rewards"], "days": {}}

    for day in days:
        idx = min(day * TURNS_PER_DAY, len(steps) - 1)
        obs = steps[idx][0]["observation"]
        per_seat = []
        for p in range(2):
            farm = obs["farms"][p]
            s = tile_stats(farm)
            crops = {}
            for row in farm["tiles"]:
                for t in row:
                    if isinstance(t, dict):
                        k = t.get("crop") or t.get("animal") or t.get("kind")
                        crops[k] = crops.get(k, 0) + 1
            per_seat.append({
                "money": farm["money"],
                "quadrants": len(farm["unlocked_quadrants"]),
                "util": ((s["planted"] + s["animals"]) / s["unlocked"]
                         if s["unlocked"] else 0.0),
                "unlocked": s["unlocked"],
                "empty": s["empty"],
                "weeds": s["weeds"],
                "mix": crops,
            })
        out["days"][day] = per_seat

    final = steps[-1][0]["observation"]
    out["shops"] = list(final.get("town", {}).get("unlocked_shops") or ())
    return out


def fetch(date, top, dest):
    """Download the top-N episodes for a day.

    Per-file downloads on purpose: a daily dataset is ~750 files at ~28 MB, so
    the whole thing is over 20 GB. The index CSV is small and carries the
    ratings, so the selection happens before any episode is pulled.
    """
    os.makedirs(dest, exist_ok=True)
    slug = DATASET.format(date=date)

    listing = subprocess.run(
        [KAGGLE, "datasets", "files", "-d", slug, "--page-size", "200"],
        capture_output=True, text=True, check=True).stdout
    # The filename is the FIRST column, followed by size and creation date, so
    # match on the leading token rather than the end of the line.
    names = [ln.split()[0] for ln in listing.splitlines()
             if ln.split() and ln.split()[0].endswith(".json")]
    if not names:
        raise SystemExit(f"no episodes listed in {slug}; got:\n{listing[:400]}")

    # No rating column is needed. The host builds each daily dataset by ordering
    # every episode by the average rating of the agents playing and keeping the
    # top ~20 GB, so membership in the dataset *is* the rating filter -- any
    # file in it is a top-of-ladder episode. The listing is alphabetical by
    # episode id, which is arbitrary with respect to strength, so taking the
    # first N is an unbiased sample of that day's strong games.
    got = []
    for name in names[:top]:
        if not os.path.exists(os.path.join(dest, name)):
            subprocess.run([KAGGLE, "datasets", "download", "-d", slug,
                            "-f", name, "-p", dest, "--force", "--unzip"],
                           check=True)
        got.append(name)
    return got


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="replay recorded actions, check the banks")
    v.add_argument("--replay", nargs="+", required=True)

    c = sub.add_parser("census", help="per-day land and portfolio census")
    c.add_argument("--replay", nargs="+", required=True)

    f = sub.add_parser("fetch", help="download top-N episodes for a date")
    f.add_argument("--date", required=True)
    f.add_argument("--top", type=int, default=20)
    f.add_argument("--dest", default=os.path.join(REPO, "replays", "ladder"))

    args = ap.parse_args()
    paths = []
    for pat in getattr(args, "replay", []) or []:
        paths.extend(sorted(glob.glob(pat)) or [pat])

    if args.cmd == "verify":
        bad = 0
        for path in paths:
            r = verify(path)
            mark = "ok " if r["match"] else "FAIL"
            print(f"[{mark}] {os.path.basename(r['path'])}  seed {r['seed']}")
            print(f"       expected {r['expected']}")
            print(f"       replayed {r['got']}")
            bad += 0 if r["match"] else 1
        if bad:
            print(f"\n{bad} replay(s) did not reproduce. Every number this "
                  f"module derives is untrustworthy until that is fixed.")
        return 1 if bad else 0

    if args.cmd == "census":
        for path in paths:
            r = census(path)
            print(f"\n{os.path.basename(path)}  seed {r['seed']}")
            print(f"  {r['names'][0]} {r['rewards'][0]:,.0f}   vs   "
                  f"{r['names'][1]} {r['rewards'][1]:,.0f}")
            print(f"  shops unlocked: {' '.join(s[:4] for s in r['shops'])}")
            for day, seats in r["days"].items():
                for p, s in enumerate(seats):
                    mix = " ".join(f"{k[:3].lower()}{v}" for k, v in
                                   sorted(s["mix"].items(), key=lambda kv: -kv[1])[:5])
                    print(f"   d{day:>2} p{p}  ${s['money']:>8,.0f}  q{s['quadrants']}"
                          f"  util {s['util']:>4.0%}  empty {s['empty']:>2}"
                          f"  weeds {s['weeds']:>2}  {mix}")
        return 0

    if args.cmd == "fetch":
        got = fetch(args.date, args.top, args.dest)
        print(f"fetched {len(got)} episodes into {args.dest}")
        print("verify them before deriving anything:")
        print(f"  python -m sim.ladder verify --replay '{args.dest}/*.json'")
        return 0


if __name__ == "__main__":
    sys.exit(main())
