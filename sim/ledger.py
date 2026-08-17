"""The submission ledger: what we shipped, and what the ladder said about it.

This file exists because of one gap. On 2026-08-06 we promoted v10 on a local
gate reading `+4,500 mean, 11 sigma` (docs/6 says `+4,138 at 13 sigma`) and v11
as a tied-but-different basin, replacing v8. Then nobody wrote down what
happened next, and for eleven days the project had no idea that:

    v8   853.9   <- replaced after 2h45m
    v10  746.1   <- the 13-sigma champion
    v11  730.2

A 13-sigma local improvement cost 108 points of rating. That is the single most
important measurement in the project's history and it was recoverable only by
re-querying Kaggle on 2026-08-17. It will not be lost again.

So: every submission gets a row here, the same day it is submitted, with the
local number that justified it *and the pool that number was measured on*.
`sim.ladder sync` appends observed ratings; `sim.ladder calibrate` joins the two
columns and reports whether our local evidence predicts the ladder at all. That
join is the only thing that can tell us whether a gate is worth obeying.

The ledger is tracked in git, unlike `runs/` and `wandb/`, because it is the one
artifact whose whole value is being durable.

    python -m sim.ledger show
    python -m sim.ledger add --id 55294870 --version v11 --params agent/params.json \
        --pool band --metric margin --value -2821 --note "35.1% wins, 14 seeds x 2 seats"
    python -m sim.ledger role --id 55290443 --role anchor
"""

import argparse
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO, "ledger.json")

# Kaggle scores only the two most recent submissions, so `role` is not
# cosmetic: it records the intent behind a slot. The policy (AGENTS.md) is one
# anchor we believe in plus one challenger per day.
ROLES = ("anchor", "challenger", "retired", "probe")


def load():
    if not os.path.exists(LEDGER):
        return {"submissions": []}
    with open(LEDGER) as fh:
        return json.load(fh)


def save(ledger):
    """Write sorted by date, so a diff of this file reads as a timeline."""
    ledger["submissions"].sort(key=lambda r: (r.get("date") or "", r["submission_id"]))
    with open(LEDGER, "w") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=True)
        fh.write("\n")


def params_sha256(path):
    """Hash of the parameter *values*, not the file bytes.

    The version name (`v10`, `v11`) is free text and has already drifted: v11
    and v12 were submitted but only v10 was ever frozen into sim/opponents/.
    A hash is the only identifier that cannot be wrong about which parameters
    actually shipped.

    Keys beginning with `_` are excluded so that a frozen snapshot (which
    carries `_notes`) hashes identically to the flat `agent/params.json` that
    shipped from it. Hashing raw bytes would make the same agent look like two
    different ones, which defeats the point.
    """
    with open(path) as fh:
        values = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def find(ledger, submission_id):
    for row in ledger["submissions"]:
        if row["submission_id"] == int(submission_id):
            return row
    return None


def upsert(ledger, submission_id, **fields):
    row = find(ledger, submission_id)
    if row is None:
        row = {"submission_id": int(submission_id), "ratings": []}
        ledger["submissions"].append(row)
    row.update({k: v for k, v in fields.items() if v is not None})
    row.setdefault("ratings", [])
    return row


def observe(ledger, submission_id, score, observed_at, rank=None):
    """Append a rating observation, but only when it actually moved.

    Ratings are queried far more often than they change, and a row that grows
    an identical entry per sync makes the trajectory -- which is the thing we
    care about, because a fresh submission starts at 600 and climbs -- unreadable.
    """
    row = find(ledger, submission_id)
    if row is None:
        return None
    prev = row["ratings"][-1] if row["ratings"] else None
    if prev and abs(prev["score"] - float(score)) < 1e-9:
        prev["observed"] = observed_at          # refresh the age, not the series
        return row
    entry = {"observed": observed_at, "score": float(score)}
    if rank is not None:
        entry["rank"] = int(rank)
    row["ratings"].append(entry)
    return row


def latest_rating(row):
    """(score, observed) of the most recent observation, or (None, None).

    Callers must respect `age`: a submission restarts at 600 and needs roughly
    24-48h of episodes to converge, so a young rating is not evidence. v8's
    853.9 was measured over 2h45m and is a lower bound, not a verdict.
    """
    if not row.get("ratings"):
        return None, None
    last = row["ratings"][-1]
    return last["score"], last["observed"]


def scored_ids(ledger, n=2):
    """The submission ids Kaggle is currently scoring: the n most recent."""
    rows = sorted(ledger["submissions"], key=lambda r: r.get("date") or "")
    return [r["submission_id"] for r in rows[-n:]]


def format_table(ledger):
    rows = sorted(ledger["submissions"], key=lambda r: r.get("date") or "")
    scored = set(scored_ids(ledger))
    out = [f"{'':2} {'id':>9}  {'date':10} {'ver':>5} {'params':>16} "
           f"{'role':>10} {'rating':>7}  local"]
    for r in rows:
        score, _ = latest_rating(r)
        local = r.get("local") or {}
        claim = ""
        if local:
            claim = (f"{local.get('metric', '?')}={local.get('value', '?')} "
                     f"on {local.get('pool', '?')}")
        out.append(
            f"{'*' if r['submission_id'] in scored else ' ':2} "
            f"{r['submission_id']:>9}  {(r.get('date') or '')[:10]:10} "
            f"{r.get('version') or '?':>5} {r.get('params_sha256') or '-':>16} "
            f"{r.get('role') or '-':>10} "
            f"{(f'{score:.1f}' if score is not None else '-'):>7}  {claim}")
    out.append("")
    out.append("* = currently scored by Kaggle (the two most recent)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sim.ledger", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="the whole ledger as a table")

    a = sub.add_parser("add", help="record a submission")
    a.add_argument("--id", required=True, type=int)
    a.add_argument("--version", required=True)
    a.add_argument("--date", help="ISO date; defaults to whatever Kaggle reports on sync")
    a.add_argument("--params", help="path to the params.json that shipped")
    a.add_argument("--role", choices=ROLES, default="challenger")
    a.add_argument("--pool", help="the opponent pool the local number came from")
    a.add_argument("--metric", help="bank | margin | win_rate | ...")
    a.add_argument("--value", type=float)
    a.add_argument("--note", default="")

    r = sub.add_parser("role", help="change a submission's role")
    r.add_argument("--id", required=True, type=int)
    r.add_argument("--role", required=True, choices=ROLES)

    args = ap.parse_args(argv)
    ledger = load()

    if args.cmd == "show":
        print(format_table(ledger))
        return 0

    if args.cmd == "role":
        if find(ledger, args.id) is None:
            print(f"no ledger row for submission {args.id}")
            return 1
        upsert(ledger, args.id, role=args.role)
        save(ledger)
        print(f"{args.id} -> {args.role}")
        return 0

    local = None
    if args.metric is not None:
        # The pool is not optional bookkeeping. "90,842 clean bank" meant three
        # different things across v3/v4/v5 because the pool changed underneath
        # it, and a bank figure without its opponent is not comparable to
        # anything -- we bank 102,773 against sub-750 opponents and 74,752
        # against 1000+ with the same agent.
        if not args.pool:
            print("--metric requires --pool: a local number without its "
                  "opponent pool cannot be calibrated against anything")
            return 1
        local = {"metric": args.metric, "value": args.value, "pool": args.pool}

    upsert(ledger, args.id,
           version=args.version,
           date=args.date,
           params_sha256=params_sha256(args.params) if args.params else None,
           role=args.role,
           local=local,
           note=args.note or None)
    save(ledger)
    print(format_table(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
