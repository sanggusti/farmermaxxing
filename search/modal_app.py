"""Fan CEM population scoring out across Modal containers.

One container scores one candidate (all its seeds and both seats), which keeps
each call at roughly seeds x 2 x 1.3s of work -- long enough that container
startup is amortised, short enough to stay well inside the timeout.

    modal run search/modal_app.py            # smoke test the image
    python -m search.cem --modal ...         # what actually uses it

Cost check: CPU is $0.0472/core-hr, so ~1.3s per episode makes 10,000 episodes
roughly $0.20. Compute is not the constraint here; wall-clock is.
"""

import contextlib
import os

try:
    import modal
except ImportError:
    modal = None  # summarise_cells and _CENSUS_KEYS work without it

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKERS = int(os.environ.get("FM_WORKERS", "8"))


def _run_episode(args):
    """One episode, one seat. Module-level so multiprocessing can pickle it."""
    import sys

    sys.path[:0] = ["/root", "/root/agent"]
    from params import unflatten
    from sim.fastplay import fast_play
    from sim.harness import make_agent

    # Tolerate the 5-tuple form: a container left warm from before `metrics`
    # existed will still be handed work by a newer driver.
    vec, seed, seat, opponent, steps = args[:5]
    metrics = args[5] if len(args) > 5 else False

    me = make_agent(unflatten(vec))
    a, b = (me, opponent) if seat == 0 else (opponent, me)
    r = fast_play(a, b, seed=seed, steps=steps, metrics=metrics)
    out = {"bank": r["banks"][seat],
           "opp_bank": r["banks"][1 - seat],
           "status": r["statuses"][seat]}
    if "metrics" in r:
        out.update(r["metrics"][seat])
    return out


# ---- Summary functions (work without modal) --------------------------------

def summarise_cells(rows, labels):
    """Per-candidate summary. `rows` and `labels` are aligned to the cell list.

    Defined here rather than imported from sim.arena because the driver runs
    outside the Modal image; sim.arena.summarise is the same arithmetic and
    tests/test_gate.py pins the two together.
    """
    import statistics

    banks = [r["bank"] for r in rows]
    wins = [1 if r["bank"] > r["opp_bank"]
            else (0 if r["bank"] < r["opp_bank"] else 0.5) for r in rows]

    by_opp = {}
    for label, r in zip(labels, rows):
        by_opp.setdefault(label, []).append(r)

    out = {
        "n": len(rows),
        "mean_bank": statistics.mean(banks),
        "median_bank": statistics.median(banks),
        "min_bank": min(banks),
        "stderr": (statistics.stdev(banks) / len(banks) ** 0.5
                   if len(banks) > 1 else 0.0),
        "win_rate": statistics.mean(wins),
        "errors": sum(1 for r in rows if r["status"] != "DONE"),
        # Positional, aligned to `cells`, so the caller can standardise each
        # cell across the population before ranking.
        "banks": banks,
        # Margin per cell. Both players trade into one market, so a shock that
        # lifts my bank lifts theirs too; the difference cancels it. That makes
        # margin the paired statistic WITHIN an episode, and its sign is the
        # win the ladder actually scores.
        "margins": [r["bank"] - r["opp_bank"] for r in rows],
        "by_opponent": {
            label: {
                "n": len(rs),
                "mean_bank": statistics.mean([r["bank"] for r in rs]),
                # What actually decides the match. Both banks rise together in
                # a shared market, so a high bank against a strong opponent can
                # still be a loss -- see search.league.worst_opponent.
                "mean_margin": statistics.mean(
                    [r["bank"] - r["opp_bank"] for r in rs]),
                "min_bank": min(r["bank"] for r in rs),
                "win_rate": statistics.mean(
                    [1 if r["bank"] > r["opp_bank"]
                     else (0 if r["bank"] < r["opp_bank"] else 0.5) for r in rs]),
            }
            for label, rs in by_opp.items()
        },
    }
    for key in _CENSUS_KEYS:
        vals = [r[key] for r in rows if key in r]
        if len(vals) == len(rows) and vals:
            out[f"mean_{key}"] = statistics.mean(vals)
    return out


# Kept in sync with sim.arena.CENSUS_KEYS by
# tests/test_league.py::test_census_key_lists_stay_in_sync; duplicated because
# this module must import cleanly without the sim package present.
#
# The per-product `sell_units_*` keys are what located the real gap on
# 2026-08-17: ~870 units of 5 products for us against ~4,170 of 9 for the top of
# the ladder, on the same board. Aggregates could not show it.
_CENSUS_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                    "EGG", "MILK", "WOOL", "FERTILIZER")
_CENSUS_KEYS = (
    "productive_tile_day_frac", "weed_tile_day_frac",
    "idle_structure_tile_day_frac", "mean_unlocked_tiles", "max_quadrants",
    "plant_actions_per_day", "products_sold_distinct", "sell_units_total",
    "end_shed_units",
) + tuple(f"sell_units_{p}" for p in _CENSUS_PRODUCTS)


# ---- Modal infrastructure (requires `modal` package) -----------------------
# Everything below this line requires `modal` and is only used by the
# `--modal` backend. The summary functions above are also used by the
# `--kaggle` backend, which does not have `modal` installed.

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("kaggle-environments==1.32.4")
        .env({"WANDB_MODE": "disabled"})   # workers just return numbers
        .add_local_dir(os.path.join(REPO, "agent"), remote_path="/root/agent")
        .add_local_dir(os.path.join(REPO, "sim"), remote_path="/root/sim")
        .add_local_dir(os.path.join(REPO, "obs"), remote_path="/root/obs")
    )

    app = modal.App("farmermaxxing", image=image)

    # A CEM generation is two synchronisation barriers (score the population,
    # then re-score the elites on holdout), so a 20-generation run pays fixed
    # per-barrier overhead 40 times. Measured at population 64: ~150s per
    # generation against ~13s of actual episode work per container.
    #
    # cpu=8 with an 8-process pool inside, not cpu=1 with more containers,
    # because the account cap is on containers and cannot be raised. Benchmarked
    # (search/bench_cpu.py), episodes/sec per container:
    #     cpu=1  0.87   1.00x
    #     cpu=4  2.96   3.39x
    #     cpu=8  5.09   5.82x   <- plateau
    #     cpu=16 5.11   5.84x   double the cost for nothing
    @app.cls(
        cpu=8.0,
        memory=16384,
        timeout=3600,
        max_containers=int(os.environ.get('FM_MAX_CONTAINERS', '250')),
        min_containers=int(os.environ.get('FM_MIN_CONTAINERS', '8')),
        scaledown_window=600,
        retries=2,
    )
    class Scorer:
        @modal.enter()
        def setup(self):
            """Runs once per container, not once per candidate."""
            import sys

            sys.path[:0] = ["/root", "/root/agent"]
            import kaggle_environments  # noqa: F401  warm the import

            from sim.arena import evaluate, summarise

            self._evaluate = evaluate
            self._summarise = summarise

        @modal.method()
        def episodes(self, batch):
            """Run a batch of episodes across this container's worker pool."""
            import multiprocessing as mp

            if len(batch) == 1:
                return [_run_episode(batch[0])]

            # fork so workers inherit the imports @modal.enter() already warmed
            ctx = mp.get_context("fork")
            with ctx.Pool(WORKERS) as pool:
                return pool.map(_run_episode, batch)

    @contextlib.contextmanager
    def session():
        """Hold the Modal app open for a whole search."""
        with app.run():
            yield

    def score_population(vectors, cells, steps, containers=100, metrics=False):
        """Score candidates on Modal. Requires an open `session()`."""
        args, owner = [], []
        for i, v in enumerate(vectors):
            for opp, _label, seed, seat in cells:
                args.append((v, seed, seat, opp, steps, metrics))
                owner.append(i)

        size = max(WORKERS, (len(args) + containers - 1) // containers)
        batches = [args[i:i + size] for i in range(0, len(args), size)]

        flat = []
        for chunk in Scorer().episodes.map(batches):
            flat.extend(chunk)

        results = [[] for _ in vectors]
        for idx, r in zip(owner, flat):
            results[idx].append(r)

        labels = [c[1] for c in cells]
        return [summarise_cells(rows, labels) for rows in results]

    @app.local_entrypoint()
    def main():
        """Smoke test: score the current defaults on two seeds."""
        import sys

        sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
        from params import Params, flatten

        print(Scorer().episodes.remote(
            [(flatten(Params()), 0, 0, "starter", 720)]))
