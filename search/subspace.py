"""Random-subspace quadratic-model search (Cartis & Roberts 2024, arXiv:2412.14431).

A research probe (issue #72, item 4), not a production replacement for CEM.
The target is the one structure neither CEM nor coordinate-wise anything can
exploit: cross-curvature. The flat directions in this landscape are diagonals
-- shifting all 11 `prio_*` together changes almost nothing, shifting one
alone changes a lot -- and a quadratic model fitted in a random 5-D subspace
keeps the cross terms WITHIN that subspace at a fraction of full-covariance
cost: (p+1)(p+2)/2 = 21 coefficients at p=5 against 1,830 for the full n=60
covariance.

Each iteration: draw a fresh random orthonormal 5-D basis, evaluate a
21-point unisolvent design around the anchor (centre, +-radius along each
axis, radius/sqrt(2) along each axis pair -- the classic quadratic
interpolation set), fit the quadratic by least squares to the per-cell
z-scored fitness, step to the model maximiser inside the trust region, and
accept or reject on the HOLDOUT cells with the same worst-opponent guard as
every other driver. Accept doubles the radius (cap 0.5), reject halves it
(floor 0.02), and the subspace is redrawn either way, so over iterations the
model sees ever-different 5-D shadows of the same 60-D landscape.

Numbers are in unit-cube coordinates (per-dim (v - lo) / (hi - lo)) purely
for geometry -- a radius must mean the same thing along `sell_floor_frac.EGG`
(range 1.55) and `land_buy_reserve` (range 6,000). There is no integer
machinery here to miscalibrate (unlike search/cmaes.py): integer dims ride as
floats and `unflatten` rounds them, accepted for v1 and stated here so the
limitation is a fact rather than a surprise.

numpy is deliberately used (QR, lstsq): it is already in the venv as cma's
dependency, and this module runs driver-side only -- workers and the Modal
image never import it.

Configuration composes from configs/subspace.yaml (issue #98):

    python -m search.subspace iterations=10 seeds=2                  # local
    python -m search.subspace backend=modal iterations=40 seeds=2
"""

import contextlib
import os
import random
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

from params import Params, flatten, unflatten                   # noqa: E402
from obs import wandb_setup                                      # noqa: E402
from sim.opponents import resolve_pool                           # noqa: E402
from search.league import (build_cells, normalised_fitness,      # noqa: E402
                           worst_opponent)
from search.cem import (CLEAN_OFFSET, HOLDOUT_OFFSET, RUNS_DIR,  # noqa: E402
                        TRAIN_POOL, finish_run, modal_session,
                        score_local, score_modal, selection_score,
                        worst_tolerance)
from search.cmaes import HIS, LOS, NAMES                         # noqa: E402

_LOS = np.array(LOS)
_SPAN = np.array(HIS) - _LOS


def to_unit(vec):
    """{name: value} -> unit-cube ndarray in NAMES order."""
    x = np.array([float(vec[n]) for n in NAMES])
    return (x - _LOS) / _SPAN


def from_unit(u):
    """Unit-cube ndarray -> {name: value}; unflatten clips and rounds later."""
    x = _LOS + np.clip(u, 0.0, 1.0) * _SPAN
    return {n: float(v) for n, v in zip(NAMES, x)}


def random_subspace(n, p, rng):
    """An n x p matrix with orthonormal columns, deterministic under `rng`."""
    g = np.array([[rng.gauss(0, 1) for _ in range(p)] for _ in range(n)])
    q, r = np.linalg.qr(g)
    # Fix the sign convention so the basis is a pure function of the draws.
    return q * np.sign(np.diag(r))


def design_points(p, radius):
    """The 21-point (at p=5) unisolvent quadratic design, in subspace coords.

    Centre + two points along each axis + one along each axis pair:
    1 + 2p + p(p-1)/2 = (p+1)(p+2)/2, exactly the number of coefficients of a
    full quadratic in p variables, so the noiseless fit is interpolation.
    """
    pts = [np.zeros(p)]
    for i in range(p):
        e = np.zeros(p)
        e[i] = radius
        pts += [e.copy(), -e]
    for i in range(p):
        for j in range(i + 1, p):
            e = np.zeros(p)
            e[i] = e[j] = radius / np.sqrt(2)
            pts.append(e)
    return pts


def quad_features(z):
    """[1, z_i ..., z_i z_j (i <= j) ...] -- (p+1)(p+2)/2 features."""
    p = len(z)
    feats = [1.0] + [float(v) for v in z]
    for i in range(p):
        for j in range(i, p):
            feats.append(float(z[i] * z[j]))
    return feats


def fit_quadratic(zs, ys):
    """Least-squares fit; returns (c, g, H) with model c + g.z + 0.5 z.H.z."""
    A = np.array([quad_features(z) for z in zs])
    coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    p = len(zs[0])
    c = coef[0]
    g = coef[1:1 + p]
    H = np.zeros((p, p))
    k = 1 + p
    for i in range(p):
        for j in range(i, p):
            # x_i x_j appears once in the features; split it symmetrically,
            # and remember the model term is 0.5 z.H.z, so H_ii = 2 * coef.
            if i == j:
                H[i, i] = 2.0 * coef[k]
            else:
                H[i, j] = H[j, i] = coef[k]
            k += 1
    return c, g, H


def model_step(g, H, radius):
    """The model maximiser within ||z|| <= radius.

    Interior stationary point if the Hessian says it is a maximum and it is
    inside the region; otherwise the gradient step to the boundary. (The
    exact trust-region subproblem has a closed form, but against per-cell
    noise this coarse step is well inside the model's own error bars.)
    """
    eigvals = np.linalg.eigvalsh(H)
    if eigvals.max() < 0:
        z = -np.linalg.solve(H, g)
        if np.linalg.norm(z) <= radius:
            return z
    gn = np.linalg.norm(g)
    if gn == 0:
        return np.zeros(len(g))
    return radius * g / gn


def main(cfg):
    """Run the search described by `cfg` (composed from configs/subspace.yaml)."""
    from omegaconf import OmegaConf   # driver-side only, like hydra below

    if cfg.backend == "kaggle":
        raise SystemExit("error: backend=kaggle is CEM-only; there is no "
                         "subspace kernel")
    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"
    if cfg.train_pool >= HOLDOUT_OFFSET:
        raise SystemExit(f"error: train_pool {cfg.train_pool} overlaps with "
                         f"holdout seeds")

    rng = random.Random(cfg.rng_seed)
    holdout_seeds = [HOLDOUT_OFFSET + i for i in range(cfg.holdout_seeds)]
    clean_seeds = [CLEAN_OFFSET + i for i in range(cfg.clean_seeds)]

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
    base = Params.from_json(cfg.init_params) if cfg.init_params else Params()
    on_modal = cfg.backend == "modal"
    score = score_modal if on_modal else score_local
    backend_session = modal_session() if on_modal else contextlib.nullcontext()

    group = cfg.group or f"subspace-i{cfg.iterations}-p{cfg.dim}"
    run_dir = os.path.join(RUNS_DIR, group)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_params.json")

    key = "margins" if cfg.fitness == "margin" else "banks"
    sel_key = "mean_margin" if cfg.fitness == "margin" else "mean_bank"
    n_design = (cfg.dim + 1) * (cfg.dim + 2) // 2 + cfg.extra_points

    # The FULL composed config plus the derived values (same shape as cem).
    with backend_session, wandb_setup.start("subspace", group=group,
                                            tags=["subspace"], config={
        **OmegaConf.to_container(cfg, resolve=True),
        "design_points": n_design,
        "train_opponents": train_labels, "reference_opponents": ref_labels,
        "heldout_opponents": heldout_labels,
        "train_episodes_total":
            n_design * len(train_opps) * cfg.seeds * 2 * cfg.iterations,
        "init_params": cfg.init_params or "defaults",
    }) as run:

        # Incumbent guarantee: the anchor is holdout-scored before anything
        # moves, so the run can never report worse than its warm start.
        anchor_u = np.clip(to_unit(flatten(base)), 0.0, 1.0)
        anchor_stats = score([from_unit(anchor_u)], holdout_cells, cfg.steps,
                             metrics=True)[0]
        best_holdout = selection_score(anchor_stats, sel_key)
        best_vec = from_unit(anchor_u)
        best_train = None
        _, best_worst = worst_opponent(anchor_stats)
        unflatten(best_vec).to_json(best_path)
        print(f"anchor  holdout {anchor_stats['mean_bank']:>11,.0f}  "
              f"selection {best_holdout:>11,.0f}")

        radius = cfg.radius
        for it in range(cfg.iterations):
            Q = random_subspace(len(NAMES), cfg.dim, rng)
            zs = design_points(cfg.dim, radius)
            zs += [np.array([rng.gauss(0, radius / 2) for _ in range(cfg.dim)])
                   for _ in range(cfg.extra_points)]
            us = [np.clip(anchor_u + Q @ z, 0.0, 1.0) for z in zs]
            population = [from_unit(u) for u in us]

            # Same rotation as search.cem (issue #68).
            train_seeds = [(it * cfg.seeds + i) % cfg.train_pool
                           for i in range(cfg.seeds)]
            train_cells = build_cells(train_opps, train_labels, train_seeds)
            stats = score(population, train_cells, cfg.steps)
            fitness = normalised_fitness([s[key] for s in stats])

            _, g, H = fit_quadratic(zs, fitness)
            z_star = model_step(g, H, radius)
            cand_u = np.clip(anchor_u + Q @ z_star, 0.0, 1.0)
            cand_vec = from_unit(cand_u)

            cand_stats = score([cand_vec], holdout_cells, cfg.steps,
                               metrics=True)[0]
            cand_sel = selection_score(cand_stats, sel_key)
            worst_label, worst_margin = worst_opponent(cand_stats)
            regressed = worst_margin < best_worst - worst_tolerance(best_worst)
            accepted = cand_sel > best_holdout and not regressed
            if accepted:
                best_holdout = cand_sel
                best_worst = max(best_worst, worst_margin)
                best_vec = cand_vec
                best_train = max(s["mean_bank"] for s in stats)
                anchor_u = cand_u
                unflatten(best_vec).to_json(best_path)
                radius = min(radius * 2.0, 0.5)
            else:
                radius = max(radius * 0.5, 0.02)

            row = {
                "iter": it,
                "train_best_bank": max(s["mean_bank"] for s in stats),
                "train_pop_mean_bank":
                    statistics.mean([s["mean_bank"] for s in stats]),
                "holdout_cand_bank": cand_stats["mean_bank"],
                "holdout_win_rate": cand_stats["win_rate"],
                "worst_opponent_margin": worst_margin,
                "accepted": int(accepted),
                "radius": radius,
                "step_norm": float(np.linalg.norm(z_star)),
                "best_holdout_overall": best_holdout,
                "selection_metric": sel_key,
            }
            for label, b in (cand_stats.get("by_opponent") or {}).items():
                row[f"vs/{label}/mean_bank"] = b["mean_bank"]
                row[f"vs/{label}/win_rate"] = b["win_rate"]
            run.log(row)
            print(f"iter {it:>3}  cand {cand_stats['mean_bank']:>11,.0f}  "
                  f"{'ACCEPT' if accepted else 'reject'}  "
                  f"radius {radius:.3f}  "
                  f"worst {worst_label} {worst_margin:>+10,.0f}")

        finish_run(run, best_vec=best_vec, best_holdout=best_holdout,
                   best_train=best_train, fitness=cfg.fitness, score_fn=score,
                   clean_cells=clean_cells, steps=cfg.steps,
                   heldout_labels=heldout_labels, run_dir=run_dir, group=group,
                   best_path=best_path)


if __name__ == "__main__":
    # Deferred decoration: importing this module must never require hydra --
    # only running it does.
    import hydra
    hydra.main(config_path=os.path.join(REPO, "configs"),
               config_name="subspace", version_base=None)(main)()
