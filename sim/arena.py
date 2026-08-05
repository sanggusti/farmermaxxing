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
from params import Params
from obs import wandb_setup


def evaluate(params, opponents, seeds, steps=720, on_episode=None):
    """Play params against each opponent across seeds, from both seats."""
    rows = []
    for opp in opponents:
        for seed in seeds:
            for seat in (0, 1):
                me = make_agent(params)
                a, b = (me, opp) if seat == 0 else (opp, me)
                r = play(a, b, seed=seed, steps=steps)

                my_bank = r["banks"][seat]
                their_bank = r["banks"][1 - seat]
                row = {
                    "opponent": opp if isinstance(opp, str) else "params",
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
    ap.add_argument("--opponents", default="starter")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--params", default=None)
    ap.add_argument("--wandb", action="store_true", help="log this sweep to W&B")
    ap.add_argument("--group", default=None)
    args = ap.parse_args()

    params = Params.from_json(args.params) if args.params else Params()
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
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

        rows = evaluate(params, opponents, seeds, args.steps, on_episode=record)
        stats = summarise(rows)

        for k, v in stats.items():
            run.summary[k] = v
        if table is not None:
            run.log({"episodes": table})

    print(f"episodes    : {stats['n']}  ({len(opponents)} opponents x "
          f"{len(seeds)} seeds x 2 seats)")
    print(f"mean bank   : {stats['mean_bank']:>12,.0f}  +/- {stats['stderr']:,.0f}")
    print(f"median bank : {stats['median_bank']:>12,.0f}")
    print(f"worst bank  : {stats['min_bank']:>12,.0f}")
    print(f"win rate    : {stats['win_rate']:>12.1%}")
    if stats["errors"]:
        print(f"ERRORS      : {stats['errors']} episodes did not finish cleanly")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
