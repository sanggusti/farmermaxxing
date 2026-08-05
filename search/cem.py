"""Cross-entropy method over the agent's parameters.

Keep a Gaussian over the 41 searchable scalars, sample a population, score each
by mean final bank, refit the Gaussian to the top slice, repeat.

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


def refit(elites, smoothing=0.3, floor=0.02):
    """Refit the Gaussian to the elite set.

    `floor` keeps a little exploration alive in every dimension so a parameter
    that collapses early can still recover in a later generation.
    """
    mean, std = {}, {}
    for name, (lo, hi, _) in SEARCH_SPACE.items():
        vals = [e[name] for e in elites]
        mean[name] = statistics.mean(vals)
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        std[name] = max(spread, (hi - lo) * floor)
    return mean, std


def score_local(vectors, seeds, opponent, steps):
    from sim.arena import evaluate, summarise
    out = []
    for vec in vectors:
        rows = evaluate(unflatten(vec), [opponent], seeds, steps)
        out.append(summarise(rows))
    return out


def score_modal(vectors, seeds, opponent, steps):
    from search.modal_app import score_population
    return score_population(vectors, seeds, opponent, steps)


def modal_session():
    from search.modal_app import session
    return session()


HOLDOUT_OFFSET = 10_000   # selection seeds: used to pick the champion
CLEAN_OFFSET = 20_000     # reporting seeds: never used to optimise or select

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
    ap.add_argument("--opponent", default="starter")
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
    # Selection is on HOLDOUT, never on train. With 41 free parameters and a
    # handful of seeds, the train score is a fitting artefact; the ladder scores
    # us on episodes we have never seen.
    best_holdout, best_vec, best_train = float("-inf"), None, None

    with backend_session, wandb_setup.start("cem", group=group, tags=["cem"], config={
        "generations": args.generations, "population": args.population,
        "elite_frac": args.elite_frac, "train_seeds": args.seeds,
        "holdout_seeds": args.holdout_seeds,
        "opponent": args.opponent, "backend": "modal" if args.modal else "local",
        "init_params": args.init_params or "defaults", "init_spread": spread,
    }) as run:

        for gen in range(args.generations):
            # Generation 0 always includes the current defaults, so a search can
            # never return something worse than what we started with.
            population = [sample(mean, std, rng) for _ in range(args.population)]
            if gen == 0:
                population[0] = flatten(base)

            stats = score(population, train_seeds, args.opponent, args.steps)
            ranked = sorted(zip(stats, population), key=lambda sp: -sp[0]["mean_bank"])
            elites = [vec for _, vec in ranked[:n_elite]]

            # Re-score the elites on unseen seeds and pick the generation's
            # champion there. Only the elites, to keep the extra cost small.
            hold_stats = score(elites, holdout_seeds, args.opponent, args.steps)
            hold_ranked = sorted(zip(hold_stats, elites),
                                 key=lambda sp: -sp[0]["mean_bank"])
            champion_stats, champion_vec = hold_ranked[0]

            if champion_stats["mean_bank"] > best_holdout:
                best_holdout = champion_stats["mean_bank"]
                best_vec = champion_vec
                best_train = ranked[0][0]["mean_bank"]
                unflatten(best_vec).to_json(best_path)

            new_mean, new_std = refit(elites)
            mean = {k: (1 - 0.3) * new_mean[k] + 0.3 * mean[k] for k in mean}
            std = new_std

            train_best = ranked[0][0]["mean_bank"]
            row = {
                "gen": gen,
                "train_best_bank": train_best,
                "train_pop_mean_bank": statistics.mean([s["mean_bank"] for s in stats]),
                "train_elite_mean_bank": statistics.mean(
                    [s["mean_bank"] for s, _ in ranked[:n_elite]]),
                "holdout_best_bank": champion_stats["mean_bank"],
                "holdout_win_rate": champion_stats["win_rate"],
                "holdout_min_bank": champion_stats["min_bank"],
                # A widening gap is the overfitting signal to watch.
                "generalisation_gap": train_best - champion_stats["mean_bank"],
                "best_holdout_overall": best_holdout,
            }
            run.log(row)
            print(f"gen {gen:>2}  train {train_best:>11,.0f}  "
                  f"holdout {champion_stats['mean_bank']:>11,.0f}  "
                  f"gap {row['generalisation_gap']:>10,.0f}  "
                  f"win {champion_stats['win_rate']:.0%}")

        # One evaluation on seeds the search never saw. The difference against
        # the selection score IS the selection bias, so measure it rather than
        # arguing about whether it exists.
        clean = None
        if best_vec is not None:
            clean = score([best_vec], clean_seeds, args.opponent, args.steps)[0]
            run.summary["clean_bank"] = clean["mean_bank"]
            run.summary["clean_min_bank"] = clean["min_bank"]
            run.summary["selection_bias"] = best_holdout - clean["mean_bank"]

        run.summary["best_holdout_bank"] = best_holdout
        run.summary["best_train_bank"] = best_train
        if best_vec is not None:
            wandb_setup.log_params_artifact(
                run, best_path, metadata={"holdout_mean_bank": best_holdout})

    print(f"\nselection holdout : {best_holdout:>12,.0f}")
    if clean is not None:
        bias = best_holdout - clean["mean_bank"]
        print(f"clean (unbiased)  : {clean['mean_bank']:>12,.0f}  "
              f"worst {clean['min_bank']:,.0f}")
        print(f"selection bias    : {bias:>+12,.0f}  "
              f"({bias / clean['mean_bank']:+.1%} of the clean score)")
        print("\nQuote the clean number. The selection score is what the search")
        print("optimised toward and is biased upward by construction.")
    print(f"\n{best_path}")
    print(f"promote with:  make promote FROM={best_path}")


if __name__ == "__main__":
    main()
