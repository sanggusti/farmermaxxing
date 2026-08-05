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

BEST_PATH = os.path.join(REPO, "agent", "params.json")


def initial_distribution(base):
    """Mean at the hand-set defaults; std at a quarter of each range."""
    mean = flatten(base)
    std = {name: (hi - lo) * 0.25 for name, (lo, hi, _) in SEARCH_SPACE.items()}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--modal", action="store_true", help="fan out on Modal")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    rng = random.Random(args.rng_seed)
    seeds = list(range(args.seeds))
    mean, std = initial_distribution(Params())
    score = score_modal if args.modal else score_local
    n_elite = max(2, int(args.population * args.elite_frac))

    group = args.group or f"cem-g{args.generations}-p{args.population}"
    best_overall, best_vec = float("-inf"), None

    with wandb_setup.start("cem", group=group, tags=["cem"], config={
        "generations": args.generations, "population": args.population,
        "elite_frac": args.elite_frac, "seeds": args.seeds,
        "opponent": args.opponent, "backend": "modal" if args.modal else "local",
    }) as run:

        for gen in range(args.generations):
            # Generation 0 always includes the current defaults, so a search can
            # never return something worse than what we started with.
            population = [sample(mean, std, rng) for _ in range(args.population)]
            if gen == 0:
                population[0] = flatten(Params())

            stats = score(population, seeds, args.opponent, args.steps)
            ranked = sorted(zip(stats, population), key=lambda sp: -sp[0]["mean_bank"])
            elites = [vec for _, vec in ranked[:n_elite]]

            top = ranked[0][0]
            if top["mean_bank"] > best_overall:
                best_overall, best_vec = top["mean_bank"], ranked[0][1]
                unflatten(best_vec).to_json(BEST_PATH)

            new_mean, new_std = refit(elites)
            mean = {k: (1 - 0.3) * new_mean[k] + 0.3 * mean[k] for k in mean}
            std = new_std

            row = {
                "gen": gen,
                "best_bank": top["mean_bank"],
                "best_win_rate": top["win_rate"],
                "elite_mean_bank": statistics.mean(
                    [s["mean_bank"] for s, _ in ranked[:n_elite]]),
                "pop_mean_bank": statistics.mean([s["mean_bank"] for s in stats]),
                "best_overall": best_overall,
            }
            run.log(row)
            print(f"gen {gen:>2}  best {row['best_bank']:>12,.0f}  "
                  f"elite {row['elite_mean_bank']:>12,.0f}  "
                  f"pop {row['pop_mean_bank']:>12,.0f}  "
                  f"win {row['best_win_rate']:.0%}")

        run.summary["best_bank"] = best_overall
        if best_vec is not None:
            wandb_setup.log_params_artifact(
                run, BEST_PATH, metadata={"mean_bank": best_overall})

    print(f"\nbest mean bank {best_overall:,.0f} -> {BEST_PATH}")


if __name__ == "__main__":
    main()
