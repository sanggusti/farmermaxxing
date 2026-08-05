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
@app.cls(
    cpu=1.0,
    memory=2048,
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
    def episode(self, vec, seed, seat, opponent, steps):
        """One episode, one seat. The unit of fan-out.

        Scoring a whole candidate in one container serialises its seeds: 12
        episodes at ~1.4s is ~17s, and with two barriers per generation that
        alone floors a generation at ~34s no matter how many containers exist.
        At episode granularity a generation costs about one episode of wall
        clock plus dispatch, because 256 candidates x 12 episodes fan out
        across the pool instead of queuing behind each other.
        """
        from params import unflatten
        from sim.fastplay import fast_play
        from sim.harness import make_agent

        me = make_agent(unflatten(vec))
        a, b = (me, opponent) if seat == 0 else (opponent, me)
        r = fast_play(a, b, seed=seed, steps=steps)
        return {"bank": r["banks"][seat],
                "opp_bank": r["banks"][1 - seat],
                "status": r["statuses"][seat]}


@contextlib.contextmanager
def session():
    """Hold the Modal app open for a whole search.

    Opening `app.run()` per call re-uploads the mounts and re-creates the app
    every time, which measured at roughly 15 minutes per CEM generation against
    about 1 minute of real episode compute.
    """
    with app.run():
        yield


def score_population(vectors, seeds, opponent, steps):
    """Score candidates on Modal. Requires an open `session()`.

    Fans out at episode granularity and reassembles per candidate locally, so
    the summary statistics are identical to `sim.arena.summarise`.
    """
    import statistics

    args, owner = [], []
    for i, v in enumerate(vectors):
        for seed in seeds:
            for seat in (0, 1):
                args.append((v, seed, seat, opponent, steps))
                owner.append(i)

    results = [[] for _ in vectors]
    for idx, r in zip(owner, Scorer().episode.starmap(args)):
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

    print(Scorer().episode.remote(flatten(Params()), 0, 0, "starter", 720))
