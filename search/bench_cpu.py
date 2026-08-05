"""Does more CPU per container beat more containers?

Modal caps this account at 100 concurrent containers and that cannot be raised.
But the cap is on *containers*, not cores. If one container with `cpu=N` can run
N episodes at once, throughput scales past the cap without needing more
containers.

It has to be processes rather than threads: the workload is pure Python and
GIL-bound, so threads would serialise and gain nothing.

    modal run search/bench_cpu.py

Reports episodes/second per container at several core counts, timed inside the
container so dispatch overhead does not pollute the comparison.
"""

import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("kaggle-environments==1.32.4")
    .env({"WANDB_MODE": "disabled"})
    .add_local_dir(os.path.join(REPO, "agent"), remote_path="/root/agent")
    .add_local_dir(os.path.join(REPO, "sim"), remote_path="/root/sim")
    .add_local_dir(os.path.join(REPO, "obs"), remote_path="/root/obs")
)

app = modal.App("farmermaxxing-bench", image=image)


def _one_episode(args):
    """Module-level so it is picklable by multiprocessing."""
    import sys

    sys.path[:0] = ["/root", "/root/agent"]
    from params import Params
    from sim.fastplay import fast_play
    from sim.harness import make_agent

    vec, seed, steps = args
    return fast_play(make_agent(Params.from_dict(vec)), "starter",
                     seed=seed, steps=steps)["banks"][0]


def _bench(vec, n_episodes, workers, steps):
    import multiprocessing as mp
    import sys
    import time

    sys.path[:0] = ["/root", "/root/agent"]
    args = [(vec, 30_000 + i, steps) for i in range(n_episodes)]

    t = time.perf_counter()
    if workers == 1:
        results = [_one_episode(a) for a in args]
    else:
        # fork, so each worker inherits the already-warm imports
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            results = pool.map(_one_episode, args)
    elapsed = time.perf_counter() - t

    return {
        "workers": workers,
        "episodes": n_episodes,
        "seconds": elapsed,
        "eps_per_sec": n_episodes / elapsed,
        "checksum": sum(results),   # must match across configurations
    }


@app.function(cpu=1.0, memory=2048, timeout=1800)
def bench_1(vec, n, steps):
    return _bench(vec, n, 1, steps)


@app.function(cpu=4.0, memory=8192, timeout=1800)
def bench_4(vec, n, steps):
    return _bench(vec, n, 4, steps)


@app.function(cpu=8.0, memory=16384, timeout=1800)
def bench_8(vec, n, steps):
    return _bench(vec, n, 8, steps)


@app.function(cpu=16.0, memory=32768, timeout=1800)
def bench_16(vec, n, steps):
    return _bench(vec, n, 16, steps)


@app.local_entrypoint()
def main(episodes: int = 32, steps: int = 720):
    import sys
    from dataclasses import asdict

    sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
    from params import Params

    path = os.path.join(REPO, "agent", "params.json")
    vec = asdict(Params.from_json(path) if os.path.exists(path) else Params())

    rows = [
        bench_1.remote(vec, episodes, steps),
        bench_4.remote(vec, episodes, steps),
        bench_8.remote(vec, episodes, steps),
        bench_16.remote(vec, episodes, steps),
    ]

    base = rows[0]["eps_per_sec"]
    print(f"\n{'cpu':>5} {'episodes':>9} {'seconds':>9} {'eps/sec':>9} {'speedup':>9}  identical")
    print("-" * 64)
    for r in rows:
        same = "yes" if r["checksum"] == rows[0]["checksum"] else "MISMATCH"
        print(f"{r['workers']:>5} {r['episodes']:>9} {r['seconds']:>9.1f} "
              f"{r['eps_per_sec']:>9.2f} {r['eps_per_sec']/base:>8.2f}x  {same}")

    best = max(rows, key=lambda r: r["eps_per_sec"])
    print(f"\nbest: cpu={best['workers']} at {best['eps_per_sec']:.2f} eps/sec per container")
    print(f"across 100 containers: {best['eps_per_sec'] * 100:.0f} eps/sec "
          f"against {base * 100:.0f} at cpu=1")
