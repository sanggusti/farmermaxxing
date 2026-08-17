"""Cross-entropy method over the agent's parameters.

Keep a Gaussian over the 60 searchable scalars (see params.SEARCH_SPACE;
36 float and 24 integer, because sell_floor_frac and late_target_mult each
expand into one dimension per key), sample a population, score each
against a fixed set of (opponent, seed, seat) cells, refit the Gaussian to the
top slice, repeat.

Why CEM and not RL: the objective is a ~40-dimensional scalar function we can
evaluate exactly against the real engine in about a second. CEM needs ~10^4
evaluations, which is minutes on Modal and carries no risk of the simulator
disagreeing with the one that scores the ladder.

Scoring uses mean bank, not win-rate. Farms are independent and the market
coupling is weak, so bank is nearly deterministic given a seed -- far less noisy
than a win/loss bit, which means many fewer episodes per candidate.

    python -m search.cem --generations 8 --population 24 --seeds 4        # local
    python -m search.cem --generations 8 --population 48 --seeds 6 --modal
"""

import argparse
import contextlib
import os
import random
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# agent/ holds flat modules (params, policy, ...) because that is how Kaggle
# unpacks a submission; put it on the path before importing them.
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

from params import Params, SEARCH_SPACE, flatten, unflatten   # noqa: E402
from obs import wandb_setup                                    # noqa: E402
from sim.arena import CENSUS_KEYS as ARENA_CENSUS_KEYS         # noqa: E402
from sim.opponents import resolve_pool                         # noqa: E402
from search.league import (build_cells, normalised_fitness,    # noqa: E402
                           worst_opponent)

# Searches write to their own run directory, never straight to the tracked
# agent/params.json. Writing into the working tree means a concurrent command
# can clobber a running search (this happened), `git add -A` commits whatever
# intermediate set is on disk (this also happened, in PR #15), two searches
# cannot run at once, and nothing records which run produced the file.
# Promotion into agent/params.json is a deliberate separate step.
RUNS_DIR = os.path.join(REPO, "runs")


def initial_distribution(base, spread=0.25):
    """Centre the Gaussian on `base`, with std at `spread` of each range.

    Cold start uses the wide default spread to explore. A warm start from an
    already-tuned set wants a narrower spread, so the search refines what it was
    given instead of wandering back out to where it began.
    """
    mean = flatten(base)
    std = {name: (hi - lo) * spread for name, (lo, hi, _) in SEARCH_SPACE.items()}
    return mean, std


def sample(mean, std, rng):
    return {name: rng.gauss(mean[name], std[name]) for name in SEARCH_SPACE}


def refit(elites, smoothing=0.3, floor=0.02, int_floor=0.4):
    """Refit the Gaussian to the elite set.

    `floor` keeps a little exploration alive in every dimension so a parameter
    that collapses early can still recover in a later generation.

    Two things here are not obvious, and both were silent bugs.

    **Elite values are clipped into the box before fitting.** `sample()` draws
    unbounded Gaussians and only `unflatten()` clips, so the stored elite
    vectors hold raw out-of-range values. Fitting to those lets the mean of a
    bound-optimal parameter march further outside the box every generation,
    with nothing to pull it back. Simulated over 30 generations on a parameter
    whose optimum sits at its lower bound, the fitted mean reached -124 against
    a bound of 0 and 100% of draws clipped to the same value -- a population of
    duplicates differing only by their noise draw, where any apparent
    improvement is max-over-noise. The champion has 22 of 45 parameters sitting
    on a bound, so this was affecting about half the search space.

    **Integer dimensions get a floor in integer units.** A draw only matters
    for an integer parameter if it changes the *rounded* value, and a floor
    expressed as a fraction of the range does not guarantee that. At
    `floor=0.02`, once the elites agreed the probability of ever changing
    `fertilize_enabled` (range 0-1), `hire_turns` (1-4) or `liquidate_days`
    (1-6) was 0.0000%, 0.0000% and 0.0001%. Three parameters were unsearchable,
    and those frozen discrete switches are plausibly where the basins live --
    which would explain why multi-restart works while in-run exploration does
    not. `int_floor` is in raw units, so 0.4 gives roughly a one-in-five chance
    of stepping to an adjacent integer.
    """
    mean, std = {}, {}
    for name, (lo, hi, kind) in SEARCH_SPACE.items():
        vals = [min(max(e[name], lo), hi) for e in elites]
        mean[name] = statistics.mean(vals)
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        floor_abs = (hi - lo) * floor
        if kind == "i":
            floor_abs = max(floor_abs, int_floor)
        std[name] = max(spread, floor_abs)
    return mean, std


def score_local(vectors, cells, steps, metrics=False):
    """Same contract as score_modal: one summary per candidate, over `cells`."""
    from sim.fastplay import fast_play
    from sim.harness import make_agent
    from search.modal_app import summarise_cells

    labels = [c[1] for c in cells]
    out = []
    for vec in vectors:
        params = unflatten(vec)
        rows = []
        for opp, _label, seed, seat in cells:
            me = make_agent(params)
            a, b = (me, opp) if seat == 0 else (opp, me)
            r = fast_play(a, b, seed=seed, steps=steps, metrics=metrics)
            row = {"bank": r["banks"][seat], "opp_bank": r["banks"][1 - seat],
                   "status": r["statuses"][seat]}
            if "metrics" in r:
                row.update(r["metrics"][seat])
            rows.append(row)
        out.append(summarise_cells(rows, labels))
    return out


def score_modal(vectors, cells, steps, metrics=False):
    from search.modal_app import score_population
    return score_population(vectors, cells, steps, metrics=metrics)


def selection_score(stats, key):
    """The champion-selection number: mean bank, or mean margin over the pool.

    `summarise_cells` reports margin per opponent but not in aggregate, so the
    pooled margin is averaged across opponents here. Equal weight per opponent
    rather than per episode, which matters when the pool is unbalanced.
    """
    if key == "mean_bank":
        return stats["mean_bank"]
    by_opp = stats.get("by_opponent") or {}
    if not by_opp:
        return stats["mean_bank"]
    return statistics.mean([b["mean_margin"] for b in by_opp.values()])


def modal_session():
    from search.modal_app import session
    return session()


HOLDOUT_OFFSET = 10_000   # selection seeds: used to pick the champion
CLEAN_OFFSET = 20_000     # reporting seeds: never used to optimise or select

# How far the champion's worst matchup may slip while its average improves.
# Not zero: at these sample sizes the worst-opponent margin carries real noise,
# and rejecting every downward wobble would freeze the search. Not generous
# either, because a collapsing matchup is a lost match on the ladder however
# good the average looks.
WORST_TOLERANCE = 0.05

# ...and a floor, in coins, because the quantity being guarded is a MARGIN and
# margins pass through zero. When the incumbent's worst margin is near zero -- a
# warm start from the reigning champion ties itself at exactly 0 -- a purely
# proportional tolerance collapses to nothing and the guard becomes infinitely
# strict. Observed: a v7 run rejected 16 of its first 17 generations and
# returned its own starting point.
#
# Sized from the noise it has to absorb. Per-cell bank standard deviation
# measures around 9,500 coins, and a per-opponent margin is averaged over
# holdout_seeds x 2 seats -- six episodes at the usual settings -- so one
# standard error is roughly 3,900. A floor near that lets ordinary sampling
# wobble through while still catching a genuine collapse.
WORST_TOLERANCE_FLOOR = 4000.0

# Three sets, because two is not enough. Train fits the parameters. Holdout
# picks the champion, once per generation, which makes it a selection set:
# repeatedly taking the max over a noisy 16-episode measurement biases it
# upward, and the bias grows with generation count. Clean seeds are touched
# exactly once, at the end, so the number they produce is not something the
# search was ever allowed to chase.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=4, help="train seeds")
    ap.add_argument("--holdout-seeds", type=int, default=6,
                    help="disjoint seeds used to select the champion")
    ap.add_argument("--clean-seeds", type=int, default=8,
                    help="seeds touched only once, for an unbiased final number")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--opponent", default="starter",
                    help="single training opponent (legacy; prefer --opponents)")
    ap.add_argument("--opponents", default=None,
                    help="pool spec for TRAINING, e.g. 'starter,v3-fixed,"
                         "v4-champion' or 'all'. Overrides --opponent. "
                         "Ranking standardises each cell across the population, "
                         "so a strong opponent is not drowned out by a weak one "
                         "producing bigger banks.")
    ap.add_argument("--fitness", choices=("bank", "margin"), default="bank",
                    help="What the population is RANKED on. `bank` is what "
                         "every search so far used. `margin` is my bank minus "
                         "the opponent's: both players trade into one market, "
                         "so a shared shock cancels, and its sign is the win "
                         "the ladder scores. Selection and reporting stay on "
                         "mean bank either way, so runs remain comparable.")
    ap.add_argument("--reference", default=None,
                    help="pool spec used for SELECTION and reporting. Held "
                         "fixed so holdout scores stay comparable across "
                         "generations and across runs. Defaults to --opponents.")
    ap.add_argument("--holdout-opponents", type=int, default=0,
                    help="withhold this many opponents from TRAINING while "
                         "keeping them in the reference pool. Seed splits do "
                         "not catch opponent memorisation: a run trained on 4 "
                         "of 6 band tapes scored +10,188 (3.96 sigma) on those "
                         "4 and -8,940 (-1.89 sigma) on the 2 held out, and "
                         "every instrument reported it as the day's best "
                         "candidate.")
    ap.add_argument("--modal", action="store_true", help="fan out on Modal")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--init-params", default=None,
                    help="warm-start from this params.json instead of defaults")
    ap.add_argument("--init-spread", type=float, default=None,
                    help="initial std as a fraction of each range "
                         "(default 0.25 cold, 0.10 warm)")
    args = ap.parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    rng = random.Random(args.rng_seed)
    train_seeds = list(range(args.seeds))
    holdout_seeds = [HOLDOUT_OFFSET + i for i in range(args.holdout_seeds)]
    clean_seeds = [CLEAN_OFFSET + i for i in range(args.clean_seeds)]

    # Training pool may vary; the reference pool must not. Selection and the
    # final clean number are both measured on the reference, so that a holdout
    # score from generation 2 and one from generation 29 mean the same thing.
    train_pool_spec = args.opponents or args.opponent
    ref_pool_spec = args.reference or train_pool_spec
    train_opps, train_labels = resolve_pool(train_pool_spec)
    ref_opps, ref_labels = resolve_pool(ref_pool_spec)

    # Opponent hold-out. The reference pool is resolved from the FULL spec above
    # and left intact, so the withheld opponents still score every generation --
    # they just never shape the parameters. That asymmetry is the whole point:
    # `vs/<label>/*` then splits cleanly into memorised and generalised.
    heldout_labels = []
    if args.holdout_opponents > 0:
        if args.holdout_opponents >= len(train_labels):
            ap.error(f"--holdout-opponents {args.holdout_opponents} leaves "
                     f"nothing to train on ({len(train_labels)} in the pool)")
        keep = list(range(len(train_labels)))
        drop = sorted(rng.sample(keep, args.holdout_opponents))
        heldout_labels = [train_labels[i] for i in drop]
        train_opps = [o for i, o in enumerate(train_opps) if i not in drop]
        train_labels = [l for i, l in enumerate(train_labels) if i not in drop]
        print(f"held out of training: {heldout_labels}")
        print(f"training on         : {train_labels}")

    train_cells = build_cells(train_opps, train_labels, train_seeds)
    holdout_cells = build_cells(ref_opps, ref_labels, holdout_seeds)
    clean_cells = build_cells(ref_opps, ref_labels, clean_seeds)
    if args.init_params:
        base = Params.from_json(args.init_params)
        spread = args.init_spread if args.init_spread is not None else 0.10
    else:
        base = Params()
        spread = args.init_spread if args.init_spread is not None else 0.25
    mean, std = initial_distribution(base, spread)
    score = score_modal if args.modal else score_local

    # Modal needs its app held open across the whole search; locally this is a
    # no-op so the loop below is identical either way.
    backend_session = modal_session() if args.modal else contextlib.nullcontext()
    n_elite = max(2, int(args.population * args.elite_frac))

    group = args.group or f"cem-g{args.generations}-p{args.population}"
    run_dir = os.path.join(RUNS_DIR, group)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_params.json")
    # Selection is on HOLDOUT, never on train. With 60 free parameters and a
    # handful of seeds, the train score is a fitting artefact; the ladder scores
    # us on episodes we have never seen.
    best_holdout, best_vec, best_train = float("-inf"), None, None
    best_worst = None

    with backend_session, wandb_setup.start("cem", group=group, tags=["cem"], config={
        "generations": args.generations, "population": args.population,
        "elite_frac": args.elite_frac, "train_seeds": args.seeds,
        "holdout_seeds": args.holdout_seeds,
        "opponent": args.opponent,
        "fitness": args.fitness,
        "train_opponents": train_labels, "reference_opponents": ref_labels,
        "heldout_opponents": heldout_labels,
        "train_cells": len(train_cells), "holdout_cells": len(holdout_cells),
        "backend": "modal" if args.modal else "local",
        "init_params": args.init_params or "defaults", "init_spread": spread,
    }) as run:

        for gen in range(args.generations):
            # Generation 0 always includes the current defaults, so a search can
            # never return something worse than what we started with.
            population = [sample(mean, std, rng) for _ in range(args.population)]
            if gen == 0:
                population[0] = flatten(base)

            stats = score(population, train_cells, args.steps)
            # Rank on per-cell z-scores, not raw mean bank. With a mixed pool a
            # raw mean is dominated by whichever opponent pays the most coins:
            # `starter` at ~140k outvotes `v3-fixed` at ~46k about three to one,
            # so the mixture would quietly collapse back into training against
            # `starter` -- the exact bias this change exists to remove.
            key = "margins" if args.fitness == "margin" else "banks"
            fitness = normalised_fitness([s[key] for s in stats])
            ranked = sorted(zip(fitness, stats, population), key=lambda t: -t[0])
            elites = [vec for _, _, vec in ranked[:n_elite]]

            # Re-score the elites on unseen seeds and pick the generation's
            # champion there. Only the elites, to keep the extra cost small.
            # Census only on the elite re-scoring: it is a fraction of the
            # episodes and these are the numbers that get reported, so the
            # sampling cost never lands on the hot path.
            hold_stats = score(elites, holdout_cells, args.steps, metrics=True)
            # Selection uses the SAME quantity the population was ranked on,
            # in raw units over the FIXED reference pool. Raw rather than
            # z-scored because z-scores are relative to one population and the
            # champion must be comparable across generations; fixed pool for
            # the same reason.
            #
            # Ranking on margin while selecting on bank is incoherent, and it
            # showed: v8's elites were margin-good, then the bank-best of them
            # was chosen, and the result won every matchup while its mean bank
            # stayed flat. Mean margin over a fixed pool is just as comparable
            # across generations as mean bank, so there is no reason to mix.
            sel_key = "mean_margin" if args.fitness == "margin" else "mean_bank"
            hold_ranked = sorted(zip(hold_stats, elites),
                                 key=lambda sp: -selection_score(sp[0], sel_key))
            champion_stats, champion_vec = hold_ranked[0]

            # Promote on the aggregate, but refuse a champion that buys its
            # average by collapsing against one opponent. The ladder scores
            # wins and losses, so trading a matchup away for coins elsewhere
            # loses matches even as the mean rises. This is the gate's
            # fourth check, moved into the loop where it can still steer.
            worst_label, worst_margin = worst_opponent(champion_stats)
            # Tolerance is absolute, scaled by the incumbent's magnitude. A
            # multiplicative `best * (1 - tol)` is wrong for a signed quantity:
            # at best_worst = -1,000 it sets the bar at -950, so a genuine
            # improvement to -980 would be rejected for being "below" it.
            tolerance = max(WORST_TOLERANCE * abs(best_worst or 0.0),
                            WORST_TOLERANCE_FLOOR)
            regressed = (best_worst is not None
                         and worst_margin < best_worst - tolerance)
            if selection_score(champion_stats, sel_key) > best_holdout and not regressed:
                best_holdout = selection_score(champion_stats, sel_key)
                best_worst = worst_margin if best_worst is None else max(best_worst, worst_margin)
                best_vec = champion_vec
                best_train = ranked[0][1]["mean_bank"]
                unflatten(best_vec).to_json(best_path)

            new_mean, new_std = refit(elites)
            mean = {k: (1 - 0.3) * new_mean[k] + 0.3 * mean[k] for k in mean}
            std = new_std

            train_best = ranked[0][1]["mean_bank"]
            row = {
                "gen": gen,
                "train_best_bank": train_best,
                "train_pop_mean_bank": statistics.mean([s["mean_bank"] for s in stats]),
                "train_elite_mean_bank": statistics.mean(
                    [s["mean_bank"] for _, s, _ in ranked[:n_elite]]),
                "holdout_best_bank": champion_stats["mean_bank"],
                "holdout_win_rate": champion_stats["win_rate"],
                "holdout_min_bank": champion_stats["min_bank"],
                # A widening gap is the overfitting signal to watch.
                "generalisation_gap": train_best - champion_stats["mean_bank"],
                # In the units the champion was SELECTED on: mean bank, or
                # mean margin over the reference pool. Not interchangeable.
                "best_holdout_overall": best_holdout,
                "selection_metric": sel_key,
            }
            # Land and breadth census for the generation champion. Diagnostics,
            # never fitness -- optimising a proxy for "using the farm" instead
            # of the bank is how you get a farm that looks busy and earns less.
            for key in ARENA_CENSUS_KEYS:
                if f"mean_{key}" in champion_stats:
                    row[f"holdout_{key}"] = champion_stats[f"mean_{key}"]
            # Per-opponent, every generation. The 141k-vs-46k spread was only
            # discovered at the end of a search; logged here it is visible while
            # the run is still going, which is the point of watching at all.
            for label, b in (champion_stats.get("by_opponent") or {}).items():
                row[f"vs/{label}/mean_bank"] = b["mean_bank"]
                row[f"vs/{label}/win_rate"] = b["win_rate"]
            row["worst_opponent_margin"] = worst_margin
            run.log(row)
            print(f"gen {gen:>2}  train {train_best:>11,.0f}  "
                  f"holdout {champion_stats['mean_bank']:>11,.0f}  "
                  f"gap {row['generalisation_gap']:>10,.0f}  "
                  f"win {champion_stats['win_rate']:.0%}  "
                  f"worst {worst_label} {worst_margin:>+10,.0f}"
                  + ("  [rejected: worst regressed]" if regressed else ""))

        # One evaluation on seeds the search never saw. The difference against
        # the selection score IS the selection bias, so measure it rather than
        # arguing about whether it exists.
        clean = None
        if best_vec is not None:
            clean = score([best_vec], clean_cells, args.steps, metrics=True)[0]
            run.summary["clean_bank"] = clean["mean_bank"]
            run.summary["clean_min_bank"] = clean["min_bank"]
            # Compare like with like. `best_holdout` is in the units the
            # champion was SELECTED on, which under margin fitness is a margin,
            # not a bank. Subtracting a bank from a margin produced a reported
            # bias of -134% of the clean score on the v9 run -- a number with
            # no meaning that still looked like a measurement.
            clean_sel = selection_score(clean, sel_key)
            run.summary["clean_selection_score"] = clean_sel
            run.summary["selection_bias"] = best_holdout - clean_sel
            for label, b in (clean.get("by_opponent") or {}).items():
                run.summary[f"clean_vs/{label}/mean_bank"] = b["mean_bank"]
                run.summary[f"clean_vs/{label}/win_rate"] = b["win_rate"]
            for key in ARENA_CENSUS_KEYS:
                if f"mean_{key}" in clean:
                    run.summary[f"clean_{key}"] = clean[f"mean_{key}"]

        run.summary["best_holdout_bank"] = best_holdout
        run.summary["best_train_bank"] = best_train
        if best_vec is not None:
            wandb_setup.log_params_artifact(
                run, best_path, metadata={"holdout_mean_bank": best_holdout})

    unit = "margin" if args.fitness == "margin" else "bank"
    print(f"\nselection holdout : {best_holdout:>12,.0f}  (mean {unit} over "
          f"the reference pool)")
    if clean is not None:
        bias = best_holdout - selection_score(clean, sel_key)
        print(f"clean (unbiased)  : {clean['mean_bank']:>12,.0f}  "
              f"worst {clean['min_bank']:,.0f}")
        print("clean per opponent:")
        held = set(heldout_labels)
        for label, b in sorted((clean.get("by_opponent") or {}).items()):
            mark = "h" if label in held else ("T" if held else " ")
            print(f"  {mark} {label:<22} bank {b['mean_bank']:>11,.0f}   "
                  f"win {b['win_rate']:>6.1%}   "
                  f"margin {b.get('mean_margin', float('nan')):>+11,.0f}")
        if held:
            by_opp = clean.get("by_opponent") or {}
            t = [b for l, b in by_opp.items() if l not in held]
            h = [b for l, b in by_opp.items() if l in held]
            if t and h:
                tm = statistics.mean([b["mean_margin"] for b in t])
                hm = statistics.mean([b["mean_margin"] for b in h])
                tw = statistics.mean([b["win_rate"] for b in t])
                hw = statistics.mean([b["win_rate"] for b in h])
                print(f"  (T = trained against, h = held out)")
                print(f"\ntrained-on margin : {tm:>+12,.0f}  win {tw:>6.1%}")
                print(f"held-out  margin  : {hm:>+12,.0f}  win {hw:>6.1%}")
                print(f"memorisation gap  : {tm - hm:>+12,.0f}")
                if tm - hm > 5_000:
                    print("  WARNING: the candidate is markedly better against the "
                          "opponents it\n  trained on than against the ones it did "
                          "not. That gap is the part of\n  the improvement that will "
                          "not appear on the ladder.")
        clean_sel = selection_score(clean, sel_key)
        pct = f"{bias / abs(clean_sel):+.1%}" if clean_sel else "n/a"
        print(f"selection bias    : {bias:>+12,.0f}  ({pct} of the clean mean "
              f"{unit}, which is what the champion was selected on)")
        print("\nQuote the clean number. The selection score is what the search")
        print("optimised toward and is biased upward by construction.")
    print(f"\n{best_path}")
    print(f"promote with:  make promote FROM={best_path}")


if __name__ == "__main__":
    main()
