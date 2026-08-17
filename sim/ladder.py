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

    python -m sim.ladder verify    --replay replays/episode-90070208-replay.json
    python -m sim.ladder census    --replay 'replays/**/*.json'
    python -m sim.ladder fetch     --date 2026-08-05 --top 20
    python -m sim.ladder clone     --replay R --seat 1 --name top-someone
    python -m sim.ladder sync                       # ratings -> ledger.json
    python -m sim.ladder calibrate                  # local claim vs realised rating
    python -m sim.ladder mine      --date 2026-08-16 --top 12
"""

import argparse
import csv
import glob
import io
import json
import os
import re
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
    return census_steps(replay["steps"], replay["rewards"], names_of(replay),
                        seed_of(replay), days=days, path=path)


def census_steps(steps, rewards, names, seed, days=(3, 9, 15, 21, 27, 29),
                 path="<live>"):
    """The census, over a step list rather than a file.

    Split out so a LIVE episode can be censused with the same code that censuses
    a ladder replay. Our own realised tile mix had no instrument at all -- the
    project could read the meta's build off a replay but could only read its own
    intentions off `params.json`, and those turned out to be different things
    (105 tiles of crop target against 75 tiles of farm). Comparing intention to
    intention is how six target sweeps in a row produced confusing results.
    """
    out = {"path": path, "seed": seed, "names": names,
           "rewards": rewards, "days": {}}

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


# --------------------------------------------------------------------------
# The ladder as a number, not a vibe.
#
# Everything below exists because the project spent eleven days not knowing that
# its 13-sigma champion had lost 108 points of rating. A local number is a
# hypothesis about the ladder; `sync` records the outcome and `calibrate` scores
# the hypothesis. Both are cheap and neither existed.
# --------------------------------------------------------------------------

COMP = "kaggriculture"


def _kaggle_csv(*args):
    """Run a kaggle CLI subcommand with --format csv and parse it.

    CSV rather than the default table: the table is fixed-width and the
    description column contains commas, spaces and arrows, so column positions
    shift with content. Parsing that was never going to survive.
    """
    out = subprocess.run([KAGGLE, *args, "--format", "csv"],
                         capture_output=True, text=True, check=True).stdout
    # The CLI prints a paging token line before the header on some endpoints.
    lines = out.splitlines()
    while lines and not lines[0].startswith(("ref,", "teamId,")):
        lines.pop(0)
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def our_submissions():
    """Every submission Kaggle has for us, newest first."""
    return _kaggle_csv("competitions", "submissions", COMP, "--page-size", "200")


def our_standing():
    """(rank, score, n_teams) for us, from the full public leaderboard.

    The full CSV rather than `leaderboard -s`, which shows only the top 20 -- and
    we are 2455th. Downloading it is one request and gives the percentile
    context that makes a rating mean something.
    """
    import tempfile
    import zipfile

    me = subprocess.run([KAGGLE, "config", "view"], capture_output=True,
                        text=True, check=True).stdout
    m = re.search(r"username:\s*(\S+)", me)
    username = m.group(1) if m else None

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([KAGGLE, "competitions", "leaderboard", COMP, "-d",
                        "-p", tmp, "-q"], check=True)
        rows = None
        for name in os.listdir(tmp):
            path = os.path.join(tmp, name)
            if name.endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    inner = [n for n in zf.namelist() if n.endswith(".csv")][0]
                    rows = list(csv.DictReader(
                        io.TextIOWrapper(zf.open(inner), encoding="utf-8-sig")))
            elif name.endswith(".csv"):
                with open(path, encoding="utf-8-sig") as fh:
                    rows = list(csv.DictReader(fh))
    if not rows:
        return None

    scores = [float(r["Score"]) for r in rows]
    mine = next((r for r in rows
                 if username and username.lower()
                 in (r.get("TeamMemberUserNames") or "").lower()), None)
    return {
        "n_teams": len(rows),
        "rank": int(mine["Rank"]) if mine else None,
        "score": float(mine["Score"]) if mine else None,
        "team": mine["TeamName"] if mine else None,
        "leader": scores[0],
        "top10_cutoff": scores[9] if len(scores) > 9 else None,
        "p10": scores[len(scores) // 10],
        "median": scores[len(scores) // 2],
    }


def sync():
    """Pull ratings into the ledger and report the slot layout."""
    from sim import ledger

    led = ledger.load()
    today = _today()
    subs = our_submissions()
    seen = 0
    for row in subs:
        sid = int(row["ref"])
        if ledger.find(led, sid) is None:
            # A submission nobody wrote down. Record it rather than ignoring it:
            # an unrecorded submission is exactly how v10's result was lost.
            ledger.upsert(led, sid, date=row["date"], role="challenger",
                          note=f"auto-recorded by `sim.ladder sync`: "
                               f"{row['description'][:200]}")
        score = row.get("publicScore")
        if score not in (None, ""):
            ledger.observe(led, sid, float(score), today)
            seen += 1
    ledger.save(led)

    standing = our_standing()
    print(ledger.format_table(led))
    if standing:
        print()
        print(f"team      : {standing['team']}")
        print(f"rank      : {standing['rank']} of {standing['n_teams']}"
              f"   score {standing['score']}")
        print(f"leader    : {standing['leader']}"
              f"   top-10 cutoff {standing['top10_cutoff']}")
        print(f"benchmarks: 10th pct {standing['p10']}"
              f"   median {standing['median']}")
    print()
    print(f"{seen} rating(s) observed on {today}. A submission restarts at 600 "
          f"and needs\n~24-48h of episodes to converge, so do not read one "
          f"younger than that.")
    return 0


def calibrate():
    """Does our local evidence predict the ladder? Print the join and find out.

    This is the meta-instrument. Every promotion so far was decided by a local
    number, and no one had ever checked whether that number and the rating move
    in the same direction. They have not, consistently: v3 beat v2 91.7%
    head-to-head and scored 22 points lower.

    The honest caveat is printed with the table. n is tiny, the pools differ
    between rows so the local column is not even the same quantity throughout,
    and a rating that has not converged is not a measurement. The point is not a
    correlation coefficient -- it is to keep the comparison in front of us
    instead of rediscovering it in eleven days.
    """
    from sim import ledger

    led = ledger.load()
    rows = sorted(led["submissions"], key=lambda r: r.get("date") or "")
    usable = [r for r in rows if r.get("local") and ledger.latest_rating(r)[0]]

    print(f"{'ver':>5} {'local claim':>16} {'pool':<28} {'rating':>8} "
          f"{'d(local)':>10} {'d(rating)':>10}  agree")
    prev = None
    agree = disagree = 0
    for r in usable:
        loc = r["local"]
        rating = ledger.latest_rating(r)[0]
        d_loc = d_rat = None
        mark = ""
        if prev is not None and prev["local"]["pool"] == loc["pool"] \
                and prev["local"]["metric"] == loc["metric"]:
            d_loc = loc["value"] - prev["local"]["value"]
            d_rat = rating - ledger.latest_rating(prev)[0]
            same = (d_loc > 0) == (d_rat > 0)
            mark = "yes" if same else "NO"
            agree += same
            disagree += not same
        print(f"{r.get('version', '?'):>5} {loc['value']:>16,.0f} "
              f"{loc['pool'][:28]:<28} {rating:>8.1f} "
              f"{(f'{d_loc:+,.0f}' if d_loc is not None else '-'):>10} "
              f"{(f'{d_rat:+.1f}' if d_rat is not None else '-'):>10}  {mark}")
        prev = r

    print()
    if agree + disagree == 0:
        print("No two consecutive submissions share a metric and a pool, so the")
        print("local column cannot be differenced. That is the finding: six")
        print("promotions were each justified against a different yardstick.")
        print("Fix it going forward -- record `--metric` and `--pool` per")
        print("submission and keep them the same while comparing.")
    else:
        print(f"local and ladder agreed on direction {agree} time(s), "
              f"disagreed {disagree}.")
    print()
    print("Caveats, which are larger than the signal: ratings start at 600 and")
    print("climb, so a submission that held a slot briefly is under-measured")
    print("(v8's 853.9 came from 2h45m); and the pool column changes between")
    print("rows, so the local numbers are not all the same quantity.")
    return 0


def mine(date, top, dest, min_bank, keep):
    """Fetch a day's top episodes and mint tape opponents from them.

    This is the pool refresh. The `meta-*` tapes date from 2026-08-05, when the
    leader was 3047.8; the top of the ladder has since converged and sat flat at
    ~3200 for five days, which means the target is both stronger and no longer
    moving -- worth copying, and stable enough to copy.

    Both seats of a strong episode are candidates, but only seats that actually
    banked well are kept: a top-rated episode still has a loser in it, and the
    point of this pool is to be beaten by nothing we can currently build.
    """
    from sim import tape

    os.makedirs(dest, exist_ok=True)
    files = fetch(date, top, dest)
    minted, skipped = [], []

    for name in files:
        path = os.path.join(dest, name)
        try:
            rp = load(path)
        except (OSError, ValueError) as exc:
            skipped.append((name, f"unreadable: {exc}"))
            continue
        for seat in (0, 1):
            bank = float(rp["rewards"][seat])
            team = names_of(rp)[seat]
            if bank < min_bank:
                skipped.append((f"{name}#{seat}", f"banked {bank:,.0f} < {min_bank:,.0f}"))
                continue
            # Team names are arbitrary Unicode -- the leader on 2026-08-16 is
            # "カワシギ", which strips to the empty string and produced a tape
            # literally named `top-`. Fall back to the episode id so the tape is
            # still identifiable and two such teams cannot collide.
            stem = re.sub(r"[^a-z0-9]+", "", team.lower())[:16]
            slug = "top-" + (stem or f"ep{rp.get('info', {}).get('EpisodeId')}s{seat}")
            if slug in tape.names():
                skipped.append((f"{name}#{seat}", f"{slug} already in the pool"))
                continue
            actions = tape.extract(rp, seat)
            tape.save(actions, slug, meta={
                "episode": rp.get("info", {}).get("EpisodeId"),
                "seed": seed_of(rp), "seat": seat,
                "ladder_bank": bank, "team": team, "mined_date": date,
            })
            # Scale the non-degeneracy floor to what this seat actually banked.
            # `verify_tape`'s default floor of 40,000 was calibrated against the
            # meta tapes and is nearly vacuous for an agent that banks 120k: a
            # tape could lose two thirds of its output on a fresh seed and still
            # pass. Half of the recorded bank is a real check.
            v = tape.verify_tape(slug, floor=0.5 * bank)
            if not v["ok"]:
                # A tape that only works on its own seed is a recording, not an
                # opponent. Removed rather than left in the directory, because
                # `resolve_pool("top")` globs the directory.
                os.remove(os.path.join(tape.TAPE_DIR, f"{slug}.json"))
                skipped.append((f"{name}#{seat}", f"{slug} degenerate on fresh seeds"))
                continue
            minted.append((slug, team, bank, v["banks"]))
            if len(minted) >= keep:
                break
        if len(minted) >= keep:
            break

    print(f"\nminted {len(minted)} tape(s) from {date}:")
    for slug, team, bank, banks in minted:
        print(f"  {slug:<24} {team[:24]:<24} ladder {bank:>10,.0f}   "
              f"unseen seeds {' '.join(f'{b:,.0f}' for b in banks)}")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for what, why in skipped[:20]:
            print(f"  {what:<28} {why}")
    print("\nverify before using:  python -m sim.ladder verify --replay "
          f"'{dest}/*.json'")
    return 0 if minted else 1


def _today():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="replay recorded actions, check the banks")
    v.add_argument("--replay", nargs="+", required=True)

    c = sub.add_parser("census", help="per-day land and portfolio census")
    c.add_argument("--replay", nargs="+", required=True)

    cl = sub.add_parser("clone", help="freeze a replayed seat as a tape opponent")
    cl.add_argument("--replay", required=True)
    cl.add_argument("--seat", type=int, required=True)
    cl.add_argument("--name", required=True)

    f = sub.add_parser("fetch", help="download top-N episodes for a date")
    f.add_argument("--date", required=True)
    f.add_argument("--top", type=int, default=20)
    f.add_argument("--dest", default=os.path.join(REPO, "replays", "ladder"))

    sub.add_parser("sync", help="pull ratings into ledger.json and show the slots")
    sub.add_parser("calibrate", help="local claim vs realised rating, per submission")

    m = sub.add_parser("mine", help="fetch a day's top episodes and mint tapes")
    m.add_argument("--date", required=True)
    m.add_argument("--top", type=int, default=12,
                   help="how many episode files to download")
    m.add_argument("--keep", type=int, default=8,
                   help="stop after this many usable tapes")
    m.add_argument("--min-bank", type=float, default=110_000,
                   help="a seat must have banked at least this much to be worth "
                        "cloning. The default sits above our own best (~104k "
                        "against band opponents) on purpose: a pool we can "
                        "already beat teaches us nothing.")
    m.add_argument("--dest", default=os.path.join(REPO, "replays", "ladder"))

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

    if args.cmd == "clone":
        from sim import tape

        rp = load(args.replay)
        actions = tape.extract(rp, args.seat)
        path = tape.save(actions, args.name, meta={
            "episode": rp.get("info", {}).get("EpisodeId"),
            "seed": seed_of(rp), "seat": args.seat,
            "ladder_bank": rp["rewards"][args.seat],
            "team": names_of(rp)[args.seat],
        })
        print(f"wrote {path}")
        v = tape.verify_tape(args.name)
        print(f"on unseen seeds: {['%,.0f' % b if False else f'{b:,.0f}' for b in v['banks']]}")
        if not v["ok"]:
            print("  DEGENERATE -- the tape does not survive a fresh seed, so it "
                  "is not a usable opponent. Do not add it to the pool.")
            return 1
        print("  non-degenerate; usable as a frozen opponent")
        return 0

    if args.cmd == "fetch":
        got = fetch(args.date, args.top, args.dest)
        print(f"fetched {len(got)} episodes into {args.dest}")
        print("verify them before deriving anything:")
        print(f"  python -m sim.ladder verify --replay '{args.dest}/*.json'")
        return 0

    if args.cmd == "sync":
        return sync()

    if args.cmd == "calibrate":
        return calibrate()

    if args.cmd == "mine":
        return mine(args.date, args.top, args.dest, args.min_bank, args.keep)


if __name__ == "__main__":
    sys.exit(main())
