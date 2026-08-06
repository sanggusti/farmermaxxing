"""Promotion gate: decide whether a candidate is genuinely better.

    python -m sim.gate --candidate agent/params.json --champion sim/opponents/v1.json

Runs on holdout seeds only, against the full frozen pool, from both seats, and
exits non-zero unless every check passes. `make submit` sits behind it.

The checks exist because of two specific ways a candidate reads as better
without being better:

  * **Mean up, floor down.** A livestock ablation scored mean 51,131 against a
    champion's 50,588 while its worst seed was 40,173 against 48,542. The ladder
    scores win/loss and ends in one Bradley-Terry tournament, so a collapsing
    floor costs real matches that a mean conceals.
  * **Gain inside noise.** At 8 seeds the standard error runs around 1,000
    coins, so a 500-coin improvement is not evidence.
"""

import argparse
import statistics
import sys

from sim.arena import evaluate, summarise, per_opponent, shop_order_confound
from sim.opponents import resolve_pool
from params import Params
from obs import wandb_setup

# Must match search/cem.py. The gate defaults to the CLEAN range, not the
# selection range: judging a candidate on the seeds it was selected on asks the
# search to mark its own work.
HOLDOUT_OFFSET = 10_000
CLEAN_OFFSET = 20_000


def decide(cand, champ, by_opp, margin_sigmas=1.0, floor_tolerance=0.10):
    """The decision rule, as a pure function of summary statistics.

    Kept separate from episode running so it can be tested directly. Driving
    this through real gameplay makes the test slow and, worse, dependent on
    whatever the agent happens to do at a given episode length.

    Returns (checks, delta, combined_se) where checks is a list of
    (label, passed, detail).
    """
    # Standard error of the difference between two independent means.
    combined_se = (cand["stderr"] ** 2 + champ["stderr"] ** 2) ** 0.5
    delta = cand["mean_bank"] - champ["mean_bank"]
    losing = [name for name, b in by_opp.items() if b["win_rate"] < 0.5]

    checks = [
        ("no errored episodes",
         cand["errors"] == 0,
         f"{cand['errors']} episodes did not finish"),
        (f"mean beats champion by >{margin_sigmas:.1f} sigma",
         delta > margin_sigmas * combined_se,
         f"delta {delta:+,.0f} vs {margin_sigmas:.1f} sigma = {margin_sigmas * combined_se:,.0f}"),
        (f"floor within {floor_tolerance:.0%} of champion",
         cand["min_bank"] >= champ["min_bank"] * (1 - floor_tolerance),
         f"worst {cand['min_bank']:,.0f} vs champion {champ['min_bank']:,.0f}"),
        ("beats every opponent in the pool",
         not losing,
         f"losing record against {losing}"),
    ]
    return checks, delta, combined_se


def run_gate(candidate, champion, pool_spec, n_seeds, steps,
             margin_sigmas=1.0, floor_tolerance=0.10, offset=CLEAN_OFFSET):
    """Evaluate candidate and champion identically, then apply `decide`."""
    opponents, labels = resolve_pool(pool_spec)
    seeds = [offset + i for i in range(n_seeds)]

    cand_rows = evaluate(candidate, opponents, seeds, steps, labels=labels,
                         metrics=True)
    champ_rows = evaluate(champion, opponents, seeds, steps, labels=labels,
                          metrics=True)
    cand, champ = summarise(cand_rows), summarise(champ_rows)
    by_opp = per_opponent(cand_rows)

    checks, delta, combined_se = decide(
        cand, champ, by_opp, margin_sigmas, floor_tolerance)
    return (checks, cand, champ, by_opp, delta, combined_se,
            cand_rows, champ_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="agent/params.json")
    ap.add_argument("--champion", default=None,
                    help="params.json to beat; defaults to hand-set defaults")
    ap.add_argument("--opponents", default="all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--margin-sigmas", type=float, default=1.0)
    ap.add_argument("--floor-tolerance", type=float, default=0.10)
    ap.add_argument("--on-selection-seeds", action="store_true",
                    help="judge on the seeds used to select (biased; for "
                         "measuring that bias, not for promotion decisions)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    candidate = Params.from_json(args.candidate)
    champion = Params.from_json(args.champion) if args.champion else Params()

    if not args.wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    offset = HOLDOUT_OFFSET if args.on_selection_seeds else CLEAN_OFFSET
    checks, cand, champ, by_opp, delta, se, cand_rows, champ_rows = run_gate(
        candidate, champion, args.opponents, args.seeds, args.steps,
        args.margin_sigmas, args.floor_tolerance, offset=offset)
    passed = all(ok for _, ok, _ in checks)

    print(f"candidate : {args.candidate}")
    print(f"champion  : {args.champion or 'Params() defaults'}")
    kind = "selection (BIASED)" if args.on_selection_seeds else "clean"
    print(f"seeds     : {kind} {offset}..{offset + args.seeds - 1}, both seats")
    print()
    print(f"{'':<12} {'mean':>12} {'worst':>12} {'win':>8}")
    print(f"{'candidate':<12} {cand['mean_bank']:>12,.0f} {cand['min_bank']:>12,.0f} {cand['win_rate']:>7.1%}")
    print(f"{'champion':<12} {champ['mean_bank']:>12,.0f} {champ['min_bank']:>12,.0f} {champ['win_rate']:>7.1%}")
    print(f"{'delta':<12} {delta:>+12,.0f} (1 sigma = {se:,.0f})")
    print()
    # Reporting only -- deliberately NOT a check in `decide`. These explain
    # WHERE a delta came from; making them pass/fail would be hand-tuning the
    # objective, and a farm can legitimately earn more from fewer tiles.
    if "mean_productive_tile_day_frac" in cand:
        print(f"{'':<12} {'land used':>12} {'quadrants':>12} {'products':>9} {'unsold':>8}")
        for name, s in (("candidate", cand), ("champion", champ)):
            print(f"{name:<12} {s['mean_productive_tile_day_frac']:>11.1%} "
                  f"{s['mean_max_quadrants']:>12.2f} "
                  f"{s['mean_products_sold_distinct']:>9.1f} "
                  f"{s['mean_end_shed_units']:>8.0f}")
        cf_c = shop_order_confound(cand_rows)
        cf_h = shop_order_confound(champ_rows)
        if cf_c and cf_h:
            print(f"shop-unlock confound: candidate omega^2 {cf_c['omega_sq']:.2f}"
                  f" ({cf_c['n_groups']} first-shops, smallest group"
                  f" {cf_c['min_group']}),  champion omega^2 {cf_h['omega_sq']:.2f}")
            if cf_c["omega_sq"] > 0.25 and cf_c["min_group"] >= 3:
                print("  WARNING: a large share of the candidate's bank spread tracks"
                      " which shop\n  unlocked first. Land-fill changes shift that RNG"
                      " draw, so reproduce the gain\n  against opponents with different"
                      " land-fill profiles before believing it.")
            elif cf_c["min_group"] < 3:
                print("  (too few seeds per first-shop group to read this; "
                      "raise --seeds to use it)")
        print()
    print("per opponent (candidate):")
    for name, b in sorted(by_opp.items()):
        print(f"  {name:<22} bank {b['mean_bank']:>11,.0f}   win {b['win_rate']:>6.1%}")
    print()
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<42} {detail}")

    print()
    print("GATE PASSED" if passed else "GATE FAILED")

    if args.wandb:
        from dataclasses import asdict
        with wandb_setup.start("gate", config=asdict(candidate), tags=["gate"]) as run:
            run.summary.update({
                "passed": passed,
                "delta": delta,
                "combined_stderr": se,
                "candidate_mean_bank": cand["mean_bank"],
                "candidate_min_bank": cand["min_bank"],
                "champion_mean_bank": champ["mean_bank"],
            })

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
