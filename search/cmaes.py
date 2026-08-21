"""CMA-ES over the agent's parameters, at a small population.

Issue #70 rejected CMA-ES on budget grounds; issue #72 re-opened it after
noticing the argument assumed CEM's population 384. CMA-ES's own default is
lambda = 4 + floor(3 ln n) = 16 at n=60, so the same ~250k-episode budget buys
~600-1,000 generations of covariance adaptation instead of 40 -- and the
generation count, not the population, is what docs/5 measured us starved of
(#69). The covariance is the point: CEM keeps a diagonal Gaussian, and the
flat directions in this landscape are diagonals (shift all 11 `prio_*`
together), which a diagonal cannot move along and full covariance can.

Three mechanism decisions, all resolved against pycma 4.4.4's source rather
than the issue text (which asked for a unit cube):

- **Raw parameter scale, not a unit cube.** `integer_variables` calibrates
  its minimum-std floor (~0.2) and integer centering to a grid with spacing 1
  in the coordinates CMA-ES actually samples, and since 4.3.0 `ask()` returns
  integer coordinates already rounded. A unit cube breaks all of that. The
  unit-cube *intent* -- heterogeneous scales (0.05-1.6 vs 0-6000) must not
  distort the covariance -- is carried by `CMA_stds`, per-coordinate sigma
  multipliers that are not represented in C.
- **`bounds` + the default BoundTransform** replaces the silent clip in
  `unflatten` as the box mechanism: measured on this space, every `ask()`
  point comes back inside the box with integer dims integral, so `unflatten`'s
  clip/round is a no-op safety net rather than the optimiser's view of the box.
- **`popsize` is set explicitly.** With `integer_variables` non-empty and
  popsize unset, cma overrides it to 6 + 3(ln n + ln n_int) -- measured 27 at
  (60, 24) -- which would silently spend 69% more per generation than the
  budget arithmetic above assumes. Rule 7: the bug would not raise.

Everything around the optimiser is the CEM harness unchanged: the same cell
scoring (common random numbers within a generation), per-cell z-score fitness,
rotating train seeds, holdout selection with the worst-opponent guard, and the
shared finish_run clean-eval contract.

Configuration composes from configs/cmaes.yaml (issue #98):

    python -m search.cmaes generations=20 seeds=2                     # local
    python -m search.cmaes backend=modal generations=600 popsize=16 seeds=2
"""

import contextlib
import math
import os
import random
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

from params import Params, SEARCH_SPACE, flatten, unflatten   # noqa: E402
from obs import wandb_setup                                    # noqa: E402
from sim.opponents import resolve_pool                         # noqa: E402
from search.league import (build_cells, normalised_fitness,    # noqa: E402
                           worst_opponent)
from search.cem import (CLEAN_OFFSET, HOLDOUT_OFFSET, RUNS_DIR,  # noqa: E402
                        TRAIN_POOL, finish_run, modal_session,
                        score_local, score_modal, selection_score,
                        worst_tolerance)

# One fixed name order so list positions and dict keys never drift apart.
# cma works in numpy arrays; the rest of the repo works in dicts keyed by
# SEARCH_SPACE name. These two functions are the entire boundary.
NAMES = list(SEARCH_SPACE)
LOS = [float(SEARCH_SPACE[n][0]) for n in NAMES]
HIS = [float(SEARCH_SPACE[n][1]) for n in NAMES]
INT_IDX = [i for i, n in enumerate(NAMES) if SEARCH_SPACE[n][2] == "i"]


def x_to_vec(x):
    """cma point -> {search-space name: value}."""
    return {n: float(v) for n, v in zip(NAMES, x)}


def vec_to_x(vec):
    """{search-space name: value} -> cma point, in NAMES order."""
    return [float(vec[n]) for n in NAMES]


def default_popsize(n=None):
    """CMA-ES's own default, 4 + floor(3 ln n) = 16 at n=60 -- NOT cma's
    integer-variables override (see module docstring)."""
    return 4 + int(3 * math.log(n if n is not None else len(NAMES)))


def make_es(x0, spread, popsize, seed):
    """Construct the strategy with the three resolved mechanisms applied."""
    import cma
    return cma.CMAEvolutionStrategy(list(x0), 1.0, {
        "popsize": popsize,
        "bounds": [LOS, HIS],
        "integer_variables": list(INT_IDX),
        "CMA_stds": [spread * (hi - lo) for lo, hi in zip(LOS, HIS)],
        "seed": seed,
        "verbose": -1,
        "verb_log": 0,   # no outcmaes/ files; wandb is the record
    })


def main(cfg):
    """Run the search described by `cfg` (composed from configs/cmaes.yaml)."""
    from omegaconf import OmegaConf   # driver-side only, like hydra below

    if cfg.backend == "kaggle":
        raise SystemExit("error: backend=kaggle is CEM-only; there is no "
                         "cmaes kernel (search/kaggle_notebook/cem_kernel.py)")
    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"
    if cfg.train_pool >= HOLDOUT_OFFSET:
        raise SystemExit(f"error: train_pool {cfg.train_pool} overlaps with "
                         f"holdout seeds")
    if cfg.train_pool < cfg.seeds:
        raise SystemExit(f"error: train_pool {cfg.train_pool} < seeds "
                         f"{cfg.seeds}")
    if not cfg.rng_seed:
        raise SystemExit("error: rng_seed 0 would hand cma a clock-based "
                         "seed; use >= 1")

    popsize = cfg.popsize or default_popsize()
    rng = random.Random(cfg.rng_seed)
    holdout_seeds = [HOLDOUT_OFFSET + i for i in range(cfg.holdout_seeds)]
    clean_seeds = [CLEAN_OFFSET + i for i in range(cfg.clean_seeds)]

    # Pool resolution and opponent hold-out: same semantics as search.cem.
    train_pool_spec = cfg.opponents
    ref_pool_spec = cfg.reference or train_pool_spec
    train_opps, train_labels = resolve_pool(train_pool_spec)
    ref_opps, ref_labels = resolve_pool(ref_pool_spec)
    heldout_labels = []
    if cfg.holdout_opponents > 0:
        if cfg.holdout_opponents >= len(train_labels):
            raise SystemExit(f"error: holdout_opponents "
                             f"{cfg.holdout_opponents} leaves nothing to "
                             f"train on ({len(train_labels)} in the pool)")
        drop = sorted(rng.sample(range(len(train_labels)),
                                 cfg.holdout_opponents))
        heldout_labels = [train_labels[i] for i in drop]
        train_opps = [o for i, o in enumerate(train_opps) if i not in drop]
        train_labels = [l for i, l in enumerate(train_labels) if i not in drop]
        print(f"held out of training: {heldout_labels}")
        print(f"training on         : {train_labels}")

    holdout_cells = build_cells(ref_opps, ref_labels, holdout_seeds)
    clean_cells = build_cells(ref_opps, ref_labels, clean_seeds)
    if cfg.init_params:
        base = Params.from_json(cfg.init_params)
        spread = cfg.init_spread if cfg.init_spread is not None else 0.10
    else:
        base = Params()
        spread = cfg.init_spread if cfg.init_spread is not None else 0.25
    on_modal = cfg.backend == "modal"
    score = score_modal if on_modal else score_local
    backend_session = modal_session() if on_modal else contextlib.nullcontext()

    group = cfg.group or f"cmaes-g{cfg.generations}-p{popsize}"
    run_dir = os.path.join(RUNS_DIR, group)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_params.json")

    key = "margins" if cfg.fitness == "margin" else "banks"
    sel_key = "mean_margin" if cfg.fitness == "margin" else "mean_bank"

    es = make_es(vec_to_x(flatten(base)), spread, popsize, cfg.rng_seed)

    # The FULL composed config plus the derived values (same shape as cem).
    # selection_metric is a string, so it lives here and never in a row
    # (see the canonical key schema in obs/wandb_setup.py).
    with backend_session, wandb_setup.start("cmaes", group=group,
                                            tags=["cmaes"],
                                            step_metric="gen", config={
        **OmegaConf.to_container(cfg, resolve=True),
        "selection_metric": sel_key,
        "popsize": popsize,
        "train_opponents": train_labels, "reference_opponents": ref_labels,
        "heldout_opponents": heldout_labels,
        "train_cells_per_gen": len(train_opps) * cfg.seeds * 2,
        "train_episodes_total":
            popsize * len(train_opps) * cfg.seeds * 2 * cfg.generations,
        "init_params": cfg.init_params or "defaults", "init_spread": spread,
    }) as run:

        # The incumbent guarantee, stronger than CEM's population injection:
        # x0 is holdout-scored up front and becomes the number to beat, so
        # the run can never report a champion worse than its warm start.
        base_vec = flatten(base)
        base_stats = score([base_vec], holdout_cells, cfg.steps,
                           metrics=True)[0]
        best_holdout = selection_score(base_stats, sel_key)
        best_vec = base_vec
        best_train = None
        _, best_worst = worst_opponent(base_stats)
        unflatten(best_vec).to_json(best_path)
        print(f"warm start  holdout {base_stats['mean_bank']:>11,.0f}  "
              f"selection {best_holdout:>11,.0f}")

        for gen in range(cfg.generations):
            xs = es.ask()
            population = [x_to_vec(x) for x in xs]

            # Same rotation as search.cem (issue #68): fresh block per
            # generation, common random numbers within one.
            train_seeds = [(gen * cfg.seeds + i) % cfg.train_pool
                           for i in range(cfg.seeds)]
            train_cells = build_cells(train_opps, train_labels, train_seeds)

            stats = score(population, train_cells, cfg.steps)
            fitness = normalised_fitness([s[key] for s in stats])
            # cma minimises; CMA-ES is rank-based, so the per-cell z-scores
            # are a legal objective and keep the mixed-pool balance rationale
            # from search.league (a big-coin opponent must not outvote).
            es.tell(xs, [-f for f in fitness])

            train_rank = sorted(zip(fitness, stats, population),
                                key=lambda t: -t[0])
            train_best = train_rank[0][1]["mean_bank"]
            row = {
                "gen": gen,
                "train_best_bank": train_best,
                "train_pop_mean_bank":
                    statistics.mean([s["mean_bank"] for s in stats]),
                # The covariance-adaptation readout this experiment exists to
                # watch: if axis_ratio never leaves ~1, 600 generations of
                # full covariance bought nothing a diagonal did not have.
                "sigma": float(es.sigma),
                "axis_ratio": float(max(es.D) / min(es.D)),
                "best_holdout_bank": best_holdout,
            }

            # Periodic selection, not per-generation: at popsize 16 and 2
            # train seeds a generation is cheap, and holdout-scoring every
            # one of 600 generations would spend more on selecting than on
            # searching (and take 600 maxima over holdout noise -- the
            # selection bias grows with every max taken).
            is_last = gen == cfg.generations - 1
            if gen % cfg.holdout_every == 0 or is_last:
                cand_vec = train_rank[0][2]
                cand_stats = score([cand_vec], holdout_cells, cfg.steps,
                                   metrics=True)[0]
                cand_sel = selection_score(cand_stats, sel_key)
                worst_label, worst_margin = worst_opponent(cand_stats)
                regressed = worst_margin < best_worst - worst_tolerance(best_worst)
                if cand_sel > best_holdout and not regressed:
                    best_holdout = cand_sel
                    best_worst = max(best_worst, worst_margin)
                    best_vec = cand_vec
                    best_train = train_best
                    unflatten(best_vec).to_json(best_path)
                row.update({
                    "holdout_best_bank": cand_stats["mean_bank"],
                    "holdout_win_rate": cand_stats["win_rate"],
                    "holdout_min_bank": cand_stats["min_bank"],
                    "generalisation_gap": train_best - cand_stats["mean_bank"],
                    "worst_opponent_margin": worst_margin,
                })
                for label, b in (cand_stats.get("by_opponent") or {}).items():
                    row[f"vs/{label}/mean_bank"] = b["mean_bank"]
                    row[f"vs/{label}/win_rate"] = b["win_rate"]
                print(f"gen {gen:>3}  train {train_best:>11,.0f}  "
                      f"holdout {cand_stats['mean_bank']:>11,.0f}  "
                      f"sigma {es.sigma:.3f}  axis {row['axis_ratio']:.2f}  "
                      f"worst {worst_label} {worst_margin:>+10,.0f}"
                      + ("  [rejected: worst regressed]" if regressed else ""))
            run.log(row)

        run.summary["final_sigma"] = float(es.sigma)
        run.summary["final_axis_ratio"] = float(max(es.D) / min(es.D))
        finish_run(run, best_vec=best_vec, best_holdout=best_holdout,
                   best_train=best_train, fitness=cfg.fitness, score_fn=score,
                   clean_cells=clean_cells, steps=cfg.steps,
                   heldout_labels=heldout_labels, run_dir=run_dir, group=group,
                   best_path=best_path)


if __name__ == "__main__":
    # Deferred decoration: importing this module must never require hydra
    # (search.subspace imports NAMES/LOS/HIS from here) -- only running does.
    import hydra
    hydra.main(config_path=os.path.join(REPO, "configs"),
               config_name="cmaes", version_base=None)(main)()
