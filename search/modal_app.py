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

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("kaggle-environments==1.32.4")
    .env({"WANDB_MODE": "disabled"})   # workers just return numbers
    .add_local_dir(os.path.join(REPO, "agent"), remote_path="/root/agent")
    .add_local_dir(os.path.join(REPO, "sim"), remote_path="/root/sim")
    .add_local_dir(os.path.join(REPO, "obs"), remote_path="/root/obs")
)

app = modal.App("farmermaxxing", image=image)

WORKERS = int(os.environ.get("FM_WORKERS", "8"))


def _run_episode(args):
    """One episode, one seat. Module-level so multiprocessing can pickle it."""
    import sys

    sys.path[:0] = ["/root", "/root/agent"]
    from params import unflatten
    from sim.fastplay import fast_play
    from sim.harness import make_agent

    vec, seed, seat, opponent, steps = args
    me = make_agent(unflatten(vec))
    a, b = (me, opponent) if seat == 0 else (opponent, me)
    r = fast_play(a, b, seed=seed, steps=steps)
    return {"bank": r["banks"][seat],
            "opp_bank": r["banks"][1 - seat],
            "status": r["statuses"][seat]}


# A CEM generation is two synchronisation barriers (score the population, then
# re-score the elites on holdout), so a 20-generation run pays fixed per-barrier
# overhead 40 times. Measured at population 64: ~150s per generation against
# ~13s of actual episode work per container. Almost all of it was containers
# cold-starting for every barrier.
#
# Three changes, in order of how much they bought:
#   min_containers   keeps a warm pool alive so a barrier does not cold-start
#   scaledown_window holds those containers through the gap while the driver
#                    refits the Gaussian between generations
#   @modal.enter()   pays the kaggle_environments import once per container
#                    instead of once per input
# cpu=8 with an 8-process pool inside, not cpu=1 with more containers, because
# the account cap is on containers and cannot be raised. Benchmarked
# (search/bench_cpu.py), episodes/sec per container:
#     cpu=1  0.87   1.00x
#     cpu=4  2.96   3.39x
#     cpu=8  5.09   5.82x   <- plateau, ~8 usable cores on the host
#     cpu=16 5.11   5.84x   double the cost for nothing
# Processes rather than threads: the workload is pure Python and GIL-bound.
# This is a wall-clock win, not a cost win. cpu=8 bills 8x per container-second
# for 5.8x the work, so cost per episode rises ~38%. Wall clock is the
# constraint we cannot buy our way out of, so that trade is worth taking.
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
        """Run a batch of episodes across this container's worker pool.

        The batch is a list of (vec, seed, seat, opponent, steps). Batching
        rather than one episode per input keeps dispatch overhead down and gives
        the pool enough work to saturate every core.
        """
        import multiprocessing as mp

        if len(batch) == 1:
            return [_run_episode(batch[0])]

        # fork so workers inherit the imports @modal.enter() already warmed
        ctx = mp.get_context("fork")
        with ctx.Pool(WORKERS) as pool:
            return pool.map(_run_episode, batch)


@contextlib.contextmanager
def session():
    """Hold the Modal app open for a whole search.

    Opening `app.run()` per call re-uploads the mounts and re-creates the app
    every time, which measured at roughly 15 minutes per CEM generation against
    about 1 minute of real episode compute.
    """
    with app.run():
        yield


def score_population(vectors, seeds, opponent, steps, containers=100):
    """Score candidates on Modal. Requires an open `session()`.

    Fans out at episode granularity, batched so each container gets enough work
    to fill its worker pool, and reassembles per candidate locally so the
    summary statistics are identical to `sim.arena.summarise`.
    """
    import statistics

    args, owner = [], []
    for i, v in enumerate(vectors):
        for seed in seeds:
            for seat in (0, 1):
                args.append((v, seed, seat, opponent, steps))
                owner.append(i)

    # One batch per container, rounded up, so every container gets a share and
    # none sits idle waiting on a straggler batch.
    size = max(WORKERS, (len(args) + containers - 1) // containers)
    batches = [args[i:i + size] for i in range(0, len(args), size)]

    flat = []
    for chunk in Scorer().episodes.map(batches):
        flat.extend(chunk)

    results = [[] for _ in vectors]
    for idx, r in zip(owner, flat):
        results[idx].append(r)

    out = []
    for rows in results:
        banks = [r["bank"] for r in rows]
        wins = [1 if r["bank"] > r["opp_bank"]
                else (0 if r["bank"] < r["opp_bank"] else 0.5) for r in rows]
        out.append({
            "n": len(rows),
            "mean_bank": statistics.mean(banks),
            "median_bank": statistics.median(banks),
            "min_bank": min(banks),
            "stderr": (statistics.stdev(banks) / len(banks) ** 0.5
                       if len(banks) > 1 else 0.0),
            "win_rate": statistics.mean(wins),
            "errors": sum(1 for r in rows if r["status"] != "DONE"),
        })
    return out


@app.local_entrypoint()
def main():
    """Smoke test: score the current defaults on two seeds."""
    import sys

    sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
    from params import Params, flatten

    print(Scorer().episodes.remote(
        [(flatten(Params()), 0, 0, "starter", 720)]))
