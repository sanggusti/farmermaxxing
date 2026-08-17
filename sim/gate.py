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

from sim.arena import (evaluate, summarise, per_opponent, shop_order_confound,
                       opponent_mix, CENSUS_PRODUCTS as PRODUCTS)
from sim.opponents import resolve_pool, champion_params, champion_name
from params import Params
from obs import wandb_setup

# Must match search/cem.py. The gate defaults to the CLEAN range, not the
# selection range: judging a candidate on the seeds it was selected on asks the
# search to mark its own work.
HOLDOUT_OFFSET = 10_000
CLEAN_OFFSET = 20_000


def paired_stderr(cand_banks, champ_banks):
    """Standard error of the difference, using the pairing we already have.

    The candidate and the champion are run on the IDENTICAL list of
    (opponent, seed, seat) cells, so their per-cell banks are paired. The
    unpaired formula `sqrt(se_a^2 + se_b^2)` throws that away, and here it
    throws away most of the information: the opponent main effect is enormous
    (roughly 128k against `starter` down to 59k against a frozen champion) and
    identical for both agents, so it cancels exactly in a paired difference
    while the unpaired formula counts it as noise twice.

    Measured on a real gate run: 79.7% of the variance is between cells rather
    than between agents, making the paired test 2.2x tighter.

    Perverse consequence of the old formula, now fixed: adding an opponent to
    the pool raised `combined_se` (more between-opponent spread) even though it
    added information, so a broader and better evaluation looked statistically
    weaker.

    Returns None when the banks are not aligned, so the caller can fall back
    rather than silently reporting a wrong number.
    """
    if (not cand_banks or not champ_banks
            or len(cand_banks) != len(champ_banks) or len(cand_banks) < 2):
        return None
    diffs = [a - b for a, b in zip(cand_banks, champ_banks)]
    return statistics.stdev(diffs) / len(diffs) ** 0.5


def holdout_split(cand_rows, champ_rows, trained_on):
    """Paired banks restricted to opponents the candidate never trained against.

    docs/6 measured why this has to be a check and not a footnote. A CEM run
    trained on four of the six band tapes looked like the day's best candidate
    at +3,812 margin and +9.5% wins. Split by opponent it was **+10,188
    (3.96 sigma) on the four it trained against and -8,940 (-1.89 sigma) on the
    two held out**. The seed splits the project has run since docs/2 do not
    catch this at all -- a four-opponent training pool is small enough to
    memorise, and the gate scored the memorisation as generalisation.

    Returns (cand_banks, champ_banks) over held-out cells, positionally paired.
    """
    if not trained_on:
        # No declared training pool means no held-out subset. Returning every
        # cell here would make the fifth check a duplicate of the second and
        # report a spurious PASS, which is worse than not checking.
        return [], []
    trained = set(trained_on)
    pairs = [(c["bank"], h["bank"]) for c, h in zip(cand_rows, champ_rows)
             if c["opponent"] not in trained]
    return [c for c, _ in pairs], [h for _, h in pairs]


def decide(cand, champ, by_opp, margin_sigmas=1.0, floor_tolerance=0.10,
           holdout=None):
    """The decision rule, as a pure function of summary statistics.

    Kept separate from episode running so it can be tested directly. Driving
    this through real gameplay makes the test slow and, worse, dependent on
    whatever the agent happens to do at a given episode length.

    `holdout`, when given, is the (cand_banks, champ_banks) pair from
    `holdout_split` and adds a fifth check. It is deliberately a *no-regression*
    test rather than an improvement test: the held-out subset is small, so
    demanding a positive delta there would reject good candidates on noise,
    while demanding it not be a sigma worse catches memorisation.

    Returns (checks, delta, combined_se) where checks is a list of
    (label, passed, detail).
    """
    # Paired when the per-cell banks are available, which they are for any
    # candidate and champion evaluated through `run_gate`. The unpaired form
    # remains as a fallback for synthetic stats in tests.
    combined_se = paired_stderr(cand.get("banks"), champ.get("banks"))
    if combined_se is None:
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

    if holdout:
        h_cand, h_champ = holdout
        h_se = paired_stderr(h_cand, h_champ)
        if h_cand and h_se:
            h_delta = (statistics.mean(h_cand) - statistics.mean(h_champ))
            checks.append((
                "no regression on held-out opponents",
                h_delta > -margin_sigmas * h_se,
                f"held-out delta {h_delta:+,.0f} over {len(h_cand)} cells "
                f"(1 sigma = {h_se:,.0f})"))

    return checks, delta, combined_se


def run_gate(candidate, champion, pool_spec, n_seeds, steps,
             margin_sigmas=1.0, floor_tolerance=0.10, offset=CLEAN_OFFSET,
             trained_on=None):
    """Evaluate candidate and champion identically, then apply `decide`."""
    opponents, labels = resolve_pool(pool_spec)
    seeds = [offset + i for i in range(n_seeds)]

    cand_rows = evaluate(candidate, opponents, seeds, steps, labels=labels,
                         metrics=True)
    champ_rows = evaluate(champion, opponents, seeds, steps, labels=labels,
                          metrics=True)
    cand, champ = summarise(cand_rows), summarise(champ_rows)
    # Both were evaluated over the same opponents x seeds x seats in the same
    # order, so these two lists are positionally paired. `decide` uses that to
    # block out the between-cell variance, which is most of it.
    cand["banks"] = [r["bank"] for r in cand_rows]
    champ["banks"] = [r["bank"] for r in champ_rows]
    by_opp = per_opponent(cand_rows)

    checks, delta, combined_se = decide(
        cand, champ, by_opp, margin_sigmas, floor_tolerance,
        holdout=holdout_split(cand_rows, champ_rows, trained_on))
    return (checks, cand, champ, by_opp, delta, combined_se,
            cand_rows, champ_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="agent/params.json")
    ap.add_argument("--champion", default=None,
                    help="params.json to beat; defaults to the snapshot named "
                         "in the CHAMPION file at the repo root")
    ap.add_argument("--opponents", default="top",
                    help="pool spec (top|band|real|frozen|all|names). Defaults "
                         "to `top`, the prize band -- see sim.opponents")
    ap.add_argument("--trained-on", default=None,
                    help="comma-separated opponent labels the candidate was "
                         "SEARCHED against. The rest of the pool becomes a "
                         "held-out subset and gets its own check.")
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
    # NOT `Params()`. Falling back to dataclass defaults meant `make gate` with
    # no CHAMPION= compared a tuned candidate against the hand-written baseline
    # that banks 24,895, which every candidate passes trivially. The gate then
    # printed GATE PASSED and meant nothing by it.
    if args.champion:
        champion = Params.from_json(args.champion)
        champion_label = args.champion
    else:
        champion = champion_params()          # raises if CHAMPION is unusable
        champion_label = f"CHAMPION -> {champion_name()}"

    if not args.wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    offset = HOLDOUT_OFFSET if args.on_selection_seeds else CLEAN_OFFSET
    trained_on = ([s.strip() for s in args.trained_on.split(",") if s.strip()]
                  if args.trained_on else None)
    checks, cand, champ, by_opp, delta, se, cand_rows, champ_rows = run_gate(
        candidate, champion, args.opponents, args.seeds, args.steps,
        args.margin_sigmas, args.floor_tolerance, offset=offset,
        trained_on=trained_on)
    passed = all(ok for _, ok, _ in checks)

    print(f"candidate : {args.candidate}")
    print(f"champion  : {champion_label}")
    print(f"pool      : {args.opponents} -> {sorted(by_opp)}")
    if trained_on:
        held = sorted(set(by_opp) - set(trained_on))
        print(f"trained on: {sorted(trained_on)}")
        print(f"held out  : {held or 'NOTHING -- the pool is entirely trained-on'}")
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
        # The revenue mix, side by side with the opponents'. This is the readout
        # for the throughput gap: against the top of the ladder we sell ~850
        # units of 5 products where they sell ~4,000 of 9, and no aggregate
        # metric the project logged could show that.
        opp_mix = opponent_mix(cand_rows)
        print(f"\n{'':<12} {'units':>7}  " + "  ".join(f"{p[:4]:>5}" for p in PRODUCTS))
        for name, s in (("candidate", cand), ("champion", champ)):
            print(f"{name:<12} {s.get('mean_sell_units_total', 0):>7,.0f}  " +
                  "  ".join(f"{s.get(f'mean_sell_units_{p}', 0):>5,.0f}"
                            for p in PRODUCTS))
        if opp_mix:
            print(f"{'opponents':<12} {sum(opp_mix.values()):>7,.0f}  " +
                  "  ".join(f"{opp_mix.get(p, 0):>5,.0f}" for p in PRODUCTS))
        print()
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
    champ_by_opp = per_opponent(champ_rows)
    trained = set(trained_on or ())
    for name, b in sorted(by_opp.items()):
        mark = "T" if name in trained else (" " if not trained else "h")
        cm = b["mean_margin"] - champ_by_opp[name]["mean_margin"]
        print(f"  {mark} {name:<22} bank {b['mean_bank']:>11,.0f}   "
              f"win {b['win_rate']:>6.1%}   margin {b['mean_margin']:>+11,.0f}"
              f"   vs champ {cm:>+10,.0f}")
    if trained:
        print("  (T = trained against, h = held out)")
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
