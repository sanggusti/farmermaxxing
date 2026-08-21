"""Semantic parameter blocks, the elite bimodality diagnostic, block crossover.

Issue #70's one prescribed experiment before touching the optimiser: per
semantic parameter block, test the CEM elite pool for bimodality (k-means
k=1 vs k=2). The diagonal Gaussian's specific multi-basin failure is that a
unimodal fit to elites straddling two basins puts the mean IN THE VALLEY
between them while sigma inflates -- so the question "are the elites
bimodal?" decides whether any recombination machinery could ever pay.

Coordinate-wise SBX/BLX crossover was refuted 2026-08-19 (DEVELOPMENTS.md
"Crossover: answered, negative"): the flat directions are diagonals, and
crossover between two bound-pinned parents manufactures a behaviourally
identical child whose fresh noise score inflates population-best with zero
underlying change. `crossover_children` below is a DIFFERENT operator --
whole-block donation, never a midpoint -- which cannot produce a valley
point by construction. It is default-off (configs/cem.yaml crossover_frac:
0.0) and promotable only on `diag/*/bimodal` evidence from a diagnostic run.

Stdlib only: this module ships inside the Kaggle kernel tarball (like
search/league.py), and numpy is driver-side only by convention -- the
workers and the kernel never import it (see search/subspace.py).
"""

import math

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

from params import SEARCH_SPACE   # noqa: E402

# The comment/blank-line groups of agent/params.py SEARCH_SPACE, promoted to
# data. tests/test_blocks.py asserts these tuples exactly partition
# SEARCH_SPACE, so a new parameter added without a block assignment fails
# loudly instead of being silently invisible to the diagnostic and immune to
# crossover.
BLOCKS = {
    "labour": ("hands_early", "hands_mid", "hands_late", "hire_turns"),
    "land": ("land_buy_reserve", "land_buy_empty_max"),
    "targets": ("target_geese", "target_cows", "target_sheep",
                "target_wheat_tiles", "target_melon_tiles",
                "target_carrot_tiles", "target_tomato_tiles",
                "target_strawberry_tiles"),
    "economy": ("animal_cash_reserve", "seed_batch",
                "plant_cutoff_slack", "plant_crops_per_turn"),
    "season_mix": ("mix_switch_day", "crop_price_elasticity",
                   "late_target_mult.target_wheat_tiles",
                   "late_target_mult.target_carrot_tiles",
                   "late_target_mult.target_tomato_tiles",
                   "late_target_mult.target_strawberry_tiles",
                   "late_target_mult.target_melon_tiles",
                   "late_target_mult.target_geese",
                   "late_target_mult.target_cows",
                   "late_target_mult.target_sheep"),
    "market": ("wheat_buy_max_price", "rival_supply_urgency",
               "rival_supply_ref", "rival_lookahead_days",
               "wheat_reserve_days", "liquidate_days",
               "shed_pressure_at", "shed_pressure_dump",
               "sell_order_floor"),
    "sell_floor": ("sell_floor_frac.WHEAT", "sell_floor_frac.CARROT",
                   "sell_floor_frac.TOMATO", "sell_floor_frac.STRAWBERRY",
                   "sell_floor_frac.MELON", "sell_floor_frac.EGG",
                   "sell_floor_frac.MILK", "sell_floor_frac.WOOL",
                   "sell_floor_frac.FERTILIZER"),
    "prio": ("prio_feed", "prio_water", "prio_harvest_animal",
             "prio_harvest_plant", "prio_collect_fertilizer", "prio_care",
             "prio_place_animal", "prio_build", "prio_plant", "prio_dig",
             "prio_fertilize"),
    "fertilize": ("fertilize_enabled", "fertilize_min_stock"),
    "distance": ("distance_penalty",),
}

# Below this many elites the verdict is untrustworthy in BOTH directions,
# measured on synthetic unimodal draws (400 seeds per size, 4-dim block):
# at n=6 the test fires SPURIOUSLY on 0.8% of unimodal pools (max observed
# separation 21.5 -- six points split into 2+4 can look arbitrarily clean),
# still 0.2% at n=12, and 0/400 from n=16 up (max separation 5.4 at n=24,
# converging toward the 2.65 artefact). Power on genuinely bimodal data
# (clusters at 0.3/0.7 of range) is 100% from n=16 at every tested tightness,
# so gating costs nothing detectable. `block_bimodality` therefore refuses to
# declare bimodal below this floor, and drivers log diag/underpowered so a
# small-pool "unimodal" cannot read as evidence (rule 7: the production
# default pool is 6 -- configs/experiment/bimodal-tpu.yaml buys width with
# population, 192 x 0.25 = 48 elites/gen).
MIN_POOL = 24

# Decision thresholds. 10.0 is Kass & Raftery's "very strong evidence" BIC
# bar. 3.0 for separation sits above the known artefact that k-means at k=2
# splits a UNIMODAL Gaussian down its widest axis: the two halves of a
# standard normal have means +-0.798 and within-half std 0.603, so the
# artefactual separation converges to 2*0.798/0.603 ~ 2.65 as n grows. A
# threshold under that would fire on every healthy elite pool ever sampled.
DELTA_BIC_THRESHOLD = 10.0
SEPARATION_THRESHOLD = 3.0


def _standardise(elites, names):
    """Elite subvectors in the unit box, clipped in.

    Clipping first is mandatory: elites are stored RAW (`sample()` draws
    unbounded Gaussians; only `unflatten()` clips), so an unclipped value at
    -124 against a bound of 0 -- the measured refit pathology -- would
    manufacture spread no sampled agent ever exhibits.
    """
    pts = []
    for e in elites:
        row = []
        for name in names:
            lo, hi, _ = SEARCH_SPACE[name]
            v = min(max(e[name], lo), hi)
            row.append((v - lo) / (hi - lo))
        pts.append(row)
    return pts


def _sse(pts, centre):
    return sum(sum((x - c) ** 2 for x, c in zip(p, centre)) for p in pts)


def _mean_point(pts):
    d = len(pts[0])
    return [sum(p[i] for p in pts) / len(pts) for i in range(d)]


def _bic(sse, n, k, d, sizes):
    """BIC in x-means form (Pelleg & Moore 2000): spherical Gaussians with a
    pooled variance. Lower is better. The multinomial term sum(n_j ln(n_j/n))
    is load-bearing -- it costs ~n*ln(2) at a balanced split, which is what
    makes a plain unimodal Gaussian robustly prefer k=1 over the variance
    reduction any split buys."""
    var = max(sse / (d * (n - k)), 1e-12)
    ln_l = (sum(nj * math.log(nj / n) for nj in sizes)
            - (n * d / 2.0) * math.log(2.0 * math.pi * var)
            - d * (n - k) / 2.0)
    p = (k - 1) + k * d + 1
    return -2.0 * ln_l + p * math.log(n)


def block_bimodality(elites, names):
    """k=1 vs k=2 on one block of the elite pool. Deterministic, zero rng.

    Returns {"n", "delta_bic", "separation", "min_cluster", "bimodal"}.
    delta_bic = BIC(k=1) - BIC(k=2), positive favouring two clusters;
    separation is the gap of the projected cluster means on the centroid
    axis over the pooled within-cluster std. For bimodal to be True: both
    thresholds cleared, both clusters >= 2 points (a singleton is an
    outlier, not a basin), and n >= MIN_POOL (below it the thresholds
    fire spuriously on unimodal data -- see MIN_POOL's measurements).

    Determinism matters operationally: the diagnostic consumes no rng draws,
    so diagnostics=true can never perturb the search it is watching. Lloyd's
    initialisation is the maximally separated point pair (ties -> lowest
    index pair) instead of a random draw.
    """
    n = len(elites)
    d = len(names)
    out = {"n": n, "delta_bic": 0.0, "separation": 0.0,
           "min_cluster": 0, "bimodal": False}
    if n < 4:
        # k=2's pooled variance divides by n-2; below four points the test
        # is arithmetic, not statistics.
        return out

    pts = _standardise(elites, names)
    mu = _mean_point(pts)
    sse1 = _sse(pts, mu)
    if sse1 == 0.0:
        return out   # every elite identical: degenerate, trivially unimodal

    # Deterministic 2-means: seed with the farthest pair, then Lloyd.
    best_i, best_j, best_d2 = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            d2 = sum((a - b) ** 2 for a, b in zip(pts[i], pts[j]))
            if d2 > best_d2:
                best_i, best_j, best_d2 = i, j, d2
    c1, c2 = list(pts[best_i]), list(pts[best_j])
    assign = None
    for _ in range(100):
        new_assign = [
            0 if (sum((x - a) ** 2 for x, a in zip(p, c1))
                  <= sum((x - b) ** 2 for x, b in zip(p, c2))) else 1
            for p in pts]
        if new_assign == assign:
            break
        assign = new_assign
        g1 = [p for p, a in zip(pts, assign) if a == 0]
        g2 = [p for p, a in zip(pts, assign) if a == 1]
        if not g1 or not g2:
            return out   # collapsed to one cluster: unimodal
        c1, c2 = _mean_point(g1), _mean_point(g2)

    g1 = [p for p, a in zip(pts, assign) if a == 0]
    g2 = [p for p, a in zip(pts, assign) if a == 1]
    sse2 = _sse(g1, c1) + _sse(g2, c2)
    out["min_cluster"] = min(len(g1), len(g2))
    out["delta_bic"] = (_bic(sse1, n, 1, d, [n])
                        - _bic(sse2, n, 2, d, [len(g1), len(g2)]))

    # Separation on the centroid axis: project every point onto the line
    # between the two centres and compare the cluster-mean gap with the
    # pooled within-cluster spread of the projections.
    axis = [b - a for a, b in zip(c1, c2)]
    norm = math.sqrt(sum(x * x for x in axis))
    if norm == 0.0:
        return out
    u = [x / norm for x in axis]
    t1 = [sum(x * w for x, w in zip(p, u)) for p in g1]
    t2 = [sum(x * w for x, w in zip(p, u)) for p in g2]
    m1 = sum(t1) / len(t1)
    m2 = sum(t2) / len(t2)
    ss_within = (sum((t - m1) ** 2 for t in t1)
                 + sum((t - m2) ** 2 for t in t2))
    gap = abs(m2 - m1)
    s_w = math.sqrt(ss_within / (n - 2))
    out["separation"] = gap / s_w if s_w > 0 else (math.inf if gap > 0 else 0.0)

    out["bimodal"] = (n >= MIN_POOL
                      and out["delta_bic"] > DELTA_BIC_THRESHOLD
                      and out["separation"] > SEPARATION_THRESHOLD
                      and out["min_cluster"] >= 2)
    return out


def bimodality_report(elites, blocks=BLOCKS):
    """The diagnostic over every block: {block_name: block_bimodality(...)}."""
    return {bname: block_bimodality(elites, names)
            for bname, names in blocks.items()}


def crossover_children(elites, n_children, rng, blocks=BLOCKS):
    """`n_children` whole-block recombinants of elite pairs.

    Per child: two distinct parents drawn uniformly from the elite pool, then
    one coin per block (in BLOCKS insertion order) decides which parent
    donates that ENTIRE block. Never a midpoint -- a diagonal-Gaussian CEM
    already performs per-coordinate uniform crossover over the elites, and
    its multi-basin failure is precisely the mean landing between basins;
    whole-block donation cannot produce a between-basin point.

    Donated values are clipped into the box (elites are stored raw; the same
    -124-against-a-bound-of-0 lesson as `refit` and `_standardise`).

    Consumes ZERO rng draws when n_children == 0 -- that early return is the
    byte-identity claim of crossover_frac=0.0, pinned by tests/test_blocks.py.
    """
    if n_children <= 0:
        return []
    block_of = {name: bname for bname, names in blocks.items()
                for name in names}
    children = []
    for _ in range(n_children):
        pa, pb = rng.sample(elites, 2)
        donor_of = {bname: (pa if rng.random() < 0.5 else pb)
                    for bname in blocks}
        child = {}
        for name, (lo, hi, _kind) in SEARCH_SPACE.items():
            if name not in block_of:
                continue   # a partial `blocks` arg recombines only its names
            donor = donor_of[block_of[name]]
            child[name] = min(max(donor[name], lo), hi)
        children.append(child)
    return children
