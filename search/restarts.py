"""Restart arithmetic: when to stop, and what the winner's number really is.

Two closed-form corrections for multi-restart sweeps (issue #72, items 2-3).
Both apply retrospectively to work already done; neither costs an episode.

**Kleywegt's stopping rule.** If restarts are exchangeable -- no ordering
information, which is exactly the situation with archetype warm starts -- the
probability that an (M+1)-th restart beats the best of M is 1/(M+1). The
five-restart archetype sweep (docs/4, clean banks 137,684 / 136,974 / 132,152 /
129,383 / 120,656) therefore left a 1/6 ~ 16.7% chance that a sixth start
would have topped `diversified`. That is the whole calculation; there is no
tuning constant to argue about.

**The max-over-restarts inflation.** `best over R restarts` is a max over R
noisy estimates. With per-restart measurement error SEM, the reported best is
inflated by roughly E[max of R standard normals] x SEM above the true value of
the winning restart: at R=8, E[max] ~ 1.424, so SEM = sigma/4 inflates the
best by ~0.36 sigma from restart selection alone -- on top of any within-run
selection bias. Corollary: restart count must be held FIXED across compared
arms, or the arm with more restarts wins by arithmetic.

**Tibshirani & Tibshirani (AOAS 2009) correction.** Estimates that inflation
from the data itself, with no distributional assumption. Folds are the shared
clean cells (every restart's clean eval plays the same (opponent, seed, seat)
cells -- common random numbers, see search/league.py); arms are the restarts.
The bias of the selected max is estimated by how often, cell by cell, some
other arm beat the overall winner:

    bias      = mean over cells c of ( max_arm score[arm][c] - score[winner][c] )
    corrected = naive_max - bias

If one arm truly dominates, the per-cell winner is the overall winner
everywhere and the bias estimate is 0. If the arms are near-tied, the
estimate is CONSERVATIVE: the per-cell gap is driven by single-episode noise
(~9,500 coins per cell, per sim/gate.py) rather than a fold-mean's, so it
overshoots -- T&T note the same in the paper. Measured on five synthetic
tied arms (truth 100, sigma 10, 40 cells): naive 101.3, corrected 91.3. The
truth is bracketed: naive is biased up, corrected is biased down, so quote
the corrected number as the floor and the naive one as the ceiling.

    python -m search.restarts --scores 137684,136974,132152,129383,120656
    python -m search.restarts --cells runs/a/clean_scores.json,runs/b/clean_scores.json

The second form reads the `clean_scores.json` that every search driver writes
next to `best_params.json`, and refuses files whose cells are not aligned --
an unpaired comparison would reintroduce the between-cell variance the paired
design exists to cancel (79.7% of it, per sim/gate.py).
"""

import argparse
import json
import math
import statistics


def p_next_restart_improves(m):
    """P(restart M+1 beats the best of M exchangeable restarts) = 1/(M+1)."""
    if m < 1:
        raise ValueError(f"need at least one completed restart, got {m}")
    return 1.0 / (m + 1)


def expected_max_std_normals(m, *, lo=-8.0, hi=8.0, intervals=4000):
    """E[max of m iid standard normals], by Simpson integration.

    Integrand: x * m * phi(x) * Phi(x)^(m-1). The tails beyond |x| = 8 carry
    less than 1e-15 of the mass, so the truncation error is far below the
    Simpson error, which at 4000 intervals is itself ~1e-12. Reference values
    this must hit: m=1 -> 0, m=2 -> 1/sqrt(pi), m=3 -> 1.5/sqrt(pi),
    m=8 -> ~1.4236 (the issue's "~1.42").
    """
    if m < 1:
        raise ValueError(f"need at least one restart, got {m}")

    def integrand(x):
        phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        return x * m * phi * cdf ** (m - 1)

    h = (hi - lo) / intervals
    total = integrand(lo) + integrand(hi)
    for i in range(1, intervals):
        total += integrand(lo + i * h) * (4 if i % 2 else 2)
    return total * h / 3.0


def expected_max_inflation(m, sem):
    """Coins of upward bias in `best of m` from restart selection alone."""
    return expected_max_std_normals(m) * sem


def tt_corrected_max(per_arm_cell_scores):
    """Tibshirani & Tibshirani bias correction for a selected maximum.

    `per_arm_cell_scores` maps arm name -> list of per-cell scores, every arm
    over the SAME cells in the SAME order (the clean_scores.json contract).

    Returns {"winner", "naive", "bias", "corrected"} where bias >= 0 and
    corrected = naive - bias.
    """
    if not per_arm_cell_scores:
        raise ValueError("no arms given")
    lengths = {len(v) for v in per_arm_cell_scores.values()}
    if len(lengths) != 1 or lengths == {0}:
        raise ValueError(f"arms must share one non-empty cell list, "
                         f"got lengths { {k: len(v) for k, v in per_arm_cell_scores.items()} }")
    means = {arm: statistics.mean(v) for arm, v in per_arm_cell_scores.items()}
    winner = max(means, key=means.get)
    n_cells = lengths.pop()
    bias = statistics.mean(
        max(v[c] for v in per_arm_cell_scores.values())
        - per_arm_cell_scores[winner][c]
        for c in range(n_cells))
    return {"winner": winner, "naive": means[winner], "bias": bias,
            "corrected": means[winner] - bias}


def restart_report(means, *, sems=None, per_arm_cells=None, p_threshold=0.1):
    """Assemble the human-readable verdict. Returns a list of lines."""
    m = len(means)
    p_next = p_next_restart_improves(m)
    lines = [f"restarts completed          : {m}",
             f"P(next restart improves)    : 1/{m + 1} = {p_next:.3f}",
             (f"verdict                     : "
              + ("CONTINUE -- another restart is still worth a run"
                 if p_next > p_threshold else
                 "STOP -- another restart is unlikely to help")
              + f"  (threshold {p_threshold:.2f})")]
    if sems:
        sem = statistics.mean(sems)
        infl = expected_max_inflation(m, sem)
        lines += [f"E[max of {m} std normals]     : {expected_max_std_normals(m):.4f}",
                  f"max-over-restarts inflation : ~{infl:,.0f} "
                  f"(SEM {sem:,.0f}) -- already inside the naive best"]
    if per_arm_cells:
        tt = tt_corrected_max(per_arm_cells)
        lines += [f"naive best ({tt['winner']})".ljust(28)
                  + f": {tt['naive']:,.0f}",
                  f"T&T bias estimate           : {tt['bias']:,.0f}",
                  f"corrected best              : {tt['corrected']:,.0f}  "
                  f"<- quote this one (VALUE= in make submit)"]
    lines.append("hold restart count FIXED across compared arms: the arm "
                 "with more restarts wins by max-over-noise arithmetic alone.")
    return lines


def _load_cells(paths):
    """Read clean_scores.json files into aligned {group: [per-cell score]}."""
    arms, labels, metric = {}, None, None
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        if metric is None:
            metric = doc["selection_metric"]
        elif doc["selection_metric"] != metric:
            raise SystemExit(f"{path}: selection_metric {doc['selection_metric']!r} "
                             f"!= {metric!r}; arms selected on different "
                             f"quantities are not comparable")
        if labels is None:
            labels = doc["cell_labels"]
        elif doc["cell_labels"] != labels:
            raise SystemExit(f"{path}: cells are not aligned with the first "
                             f"file -- runs must share --reference and "
                             f"--clean-seeds for a paired comparison")
        key = "clean_margins" if metric == "mean_margin" else "clean_banks"
        arms[doc["group"]] = doc[key]
    return arms


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", default=None,
                    help="comma-separated per-restart clean scores")
    ap.add_argument("--cells", default=None,
                    help="comma-separated clean_scores.json paths, one per "
                         "restart; enables the T&T correction")
    ap.add_argument("--sem", type=float, default=None,
                    help="per-restart measurement SEM in coins (with --scores; "
                         "derived from the cells with --cells)")
    ap.add_argument("--p-threshold", type=float, default=0.1,
                    help="stop when P(next restart improves) drops below this")
    args = ap.parse_args()

    if bool(args.scores) == bool(args.cells):
        ap.error("give exactly one of --scores or --cells")

    if args.cells:
        arms = _load_cells([p.strip() for p in args.cells.split(",")])
        means = [statistics.mean(v) for v in arms.values()]
        sems = [statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                for v in arms.values()]
        for arm, mu, sem in sorted(zip(arms, means, sems), key=lambda t: -t[1]):
            print(f"  {arm:<28} {mu:>12,.0f}  (sem {sem:,.0f})")
        print()
        lines = restart_report(means, sems=sems, per_arm_cells=arms,
                               p_threshold=args.p_threshold)
    else:
        means = [float(s) for s in args.scores.split(",")]
        sems = [args.sem] * len(means) if args.sem else None
        lines = restart_report(means, sems=sems, p_threshold=args.p_threshold)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
