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


@app.function(cpu=1.0, memory=2048, timeout=1800, max_containers=200, retries=2)
def score_one(vec, seeds, opponent, steps):
    """Score a single candidate. Returns the same dict `arena.summarise` gives.

    Workers deliberately do no W&B logging -- they return stats and the driver
    logs one row per generation. Hundreds of containers each opening their own
    run would bury the signal, and a preempted container would leave a dangling
    "running" run behind.
    """
    import sys

    sys.path[:0] = ["/root", "/root/agent"]
    from params import unflatten
    from sim.arena import evaluate, summarise

    rows = evaluate(unflatten(vec), [opponent], seeds, steps)
    return summarise(rows)


@contextlib.contextmanager
def session():
    """Hold the Modal app open for a whole search.

    Opening `app.run()` per call re-uploads the mounts and re-creates the app
    every time, which measured at roughly 15 minutes per CEM generation against
    about 1 minute of real episode compute. The search opens this once and every
    scoring call reuses it.
    """
    with app.run():
        yield


def score_population(vectors, seeds, opponent, steps):
    """Score candidates on Modal. Requires an open `session()`."""
    args = [(v, seeds, opponent, steps) for v in vectors]
    return list(score_one.starmap(args))


@app.local_entrypoint()
def main():
    """Smoke test: score the current defaults on two seeds."""
    import sys

    sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
    from params import Params, flatten

    result = score_one.remote(flatten(Params()), [0, 1], "starter", 720)
    print(result)
