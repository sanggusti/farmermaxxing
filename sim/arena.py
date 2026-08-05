"""Evaluate a parameter set against frozen opponents. The promotion gate.

    python -m sim.arena --seeds 12 --opponents starter,pass --wandb

Every matchup is played from BOTH seats. The engine quotes both players against
the same pre-commit market inventory, so seats are nearly symmetric -- but weed
spawns are drawn from one RNG stream in seat order, so they are not exactly, and
playing both removes that bias for free.

Two metrics, used for different jobs:

  mean_bank -- near-deterministic and nearly opponent-independent, so it has far
               lower variance than win-rate. This is what CEM optimises.
  win_rate  -- what the ladder actually scores. Used as the final gate before a
               submission, never for tuning (too noisy at this sample size).
"""

import argparse
import statistics
import sys

from sim.harness import play, make_agent
from sim.opponents import resolve_pool
from params import Params
from obs import wandb_setup


def evaluate(params, opponents, seeds, steps=720, on_episode=None, labels=None):
    """Play params against each opponent across seeds, from both seats.

    `opponents` may hold built-in names or Params instances. `labels` gives the
    display name for each; without it, Params opponents are all labelled the
    same and per-opponent breakdowns become useless.
    """
    if labels is None:
        labels = [o if isinstance(o, str) else f"params-{i}"
                  for i, o in enumerate(opponents)]

    rows = []
    for opp, label in zip(opponents, labels):
        for seed in seeds:
            for seat in (0, 1):
                me = make_agent(params)
                a, b = (me, opp) if seat == 0 else (opp, me)
                r = play(a, b, seed=seed, steps=steps)

                my_bank = r["banks"][seat]
                their_bank = r["banks"][1 - seat]
                row = {
                    "opponent": label,
                    "seed": seed,
                    "seat": seat,
                    "bank": my_bank,
                    "opp_bank": their_bank,
                    "win": 1 if my_bank > their_bank else (0 if my_bank < their_bank else 0.5),
                    "status": r["statuses"][seat],
                }
                rows.append(row)
                if on_episode:
                    on_episode(row)
    return rows


def per_opponent(rows):
    """Win rate and mean bank broken down by opponent.

    An aggregate can hide a regression: beating the built-ins by more while
    losing to the reigning champion nets out to roughly no change.
    """
    out = {}
    for row in rows:
        out.setdefault(row["opponent"], []).append(row)
    return {
        name: {
            "n": len(rs),
            "mean_bank": statistics.mean([r["bank"] for r in rs]),
            "win_rate": statistics.mean([r["win"] for r in rs]),
        }
        for name, rs in out.items()
    }


def summarise(rows):
    banks = [r["bank"] for r in rows]
    wins = [r["win"] for r in rows]
    errors = sum(1 for r in rows if r["status"] != "DONE")
    return {
        "n": len(rows),
        "mean_bank": statistics.mean(banks),
        "median_bank": statistics.median(banks),
        "min_bank": min(banks),
        "stderr": statistics.stdev(banks) / len(banks) ** 0.5 if len(banks) > 1 else 0.0,
        "win_rate": statistics.mean(wins),
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--opponents", default="starter",
                    help="built-ins, frozen snapshot names, or 'all'/'frozen'")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--params", default=None)
    ap.add_argument("--wandb", action="store_true", help="log this sweep to W&B")
    ap.add_argument("--group", default=None)
    args = ap.parse_args()

    params = Params.from_json(args.params) if args.params else Params()
    opponents, labels = resolve_pool(args.opponents)
    seeds = list(range(args.seeds))

    if not args.wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    from dataclasses import asdict
    with wandb_setup.start("arena", config=asdict(params), group=args.group,
                           tags=["arena"]) as run:
        table = wandb_setup.table(
            ["opponent", "seed", "seat", "bank", "opp_bank", "win", "status"])

        def record(row):
            if table is not None:
                table.add_data(*[row[c] for c in
                                 ("opponent", "seed", "seat", "bank",
                                  "opp_bank", "win", "status")])

        rows = evaluate(params, opponents, seeds, args.steps,
                        on_episode=record, labels=labels)
        stats = summarise(rows)

        breakdown = per_opponent(rows)
        for k, v in stats.items():
            run.summary[k] = v
        for name, b in breakdown.items():
            run.summary[f"vs_{name}/mean_bank"] = b["mean_bank"]
            run.summary[f"vs_{name}/win_rate"] = b["win_rate"]
        if table is not None:
            run.log({"episodes": table})

    print(f"episodes    : {stats['n']}  ({len(opponents)} opponents x "
          f"{len(seeds)} seeds x 2 seats)")
    print(f"mean bank   : {stats['mean_bank']:>12,.0f}  +/- {stats['stderr']:,.0f}")
    print(f"median bank : {stats['median_bank']:>12,.0f}")
    print(f"worst bank  : {stats['min_bank']:>12,.0f}")
    print(f"win rate    : {stats['win_rate']:>12.1%}")
    print("per opponent:")
    for name, b in sorted(breakdown.items()):
        print(f"  {name:<22} bank {b['mean_bank']:>11,.0f}   win {b['win_rate']:>6.1%}   n={b['n']}")
    if stats["errors"]:
        print(f"ERRORS      : {stats['errors']} episodes did not finish cleanly")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
