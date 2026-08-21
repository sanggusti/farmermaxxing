"""Measure what a Kaggle TPU VM actually offers as a CPU box.

The CEM backend runs on Kaggle's CPU tier: 4 cores, 30 GB, unlimited quota
(search/kaggle_nb.py). Kaggle's only bigger free machine is the TPU VM, whose
HOST exposes ~96 vCPUs and ~330 GB RAM -- and fastplay is pure Python, so for
us a TPU VM is just a very wide CPU box. Whether it is *usable* is an open
question this probe answers with numbers instead of specs: how many cores the
sandbox really grants (cpu_count vs sched_getaffinity), how fast one episode
runs on its (slower) cores, and how multiprocessing scales at full width.
TPU quota is ~20 h/week against the CPU tier's unlimited, so measure before
migrating anything.

    make probe-tpu          # push, poll, download runs/tpu-probe/probe.json

Measures only. The real backend stays on the CPU tier; a migration would be
its own change, justified by this probe's numbers.
"""

import base64
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO]

from search.kaggle_nb import (_build_tarball_bytes, _kaggle_username,  # noqa: E402
                              _wandb_api_key, poll, push_kernel, RUNS_DIR)

PROBE_SLUG_SUFFIX = "farmermaxxing-tpu-probe"

# Concatenated around the base64 payload rather than .format()ed, so the
# kernel code below can use braces freely.
_HEAD = '''#!/usr/bin/env python3
"""Auto-generated TPU-VM core probe for Kaggle. Do not edit directly."""

import base64
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time

probe = {"cpu_count": os.cpu_count()}
try:
    probe["sched_affinity"] = len(os.sched_getaffinity(0))
except AttributeError:
    probe["sched_affinity"] = None
with open("/proc/meminfo") as f:
    mem = dict(line.split(":", 1) for line in f)
probe["mem_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1048576, 1)
probe["mem_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 1048576, 1)
print(json.dumps(probe, indent=2))

# Same engine pin as the CEM kernel and make check-engine.
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "kaggle-environments==1.32.4", "-q", "--disable-pip-version-check",
])

_PAYLOAD = """'''

_TAIL = '''"""

CODE = "/tmp/fm"
os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PAYLOAD)),
             mode="r:gz").extractall(CODE)
sys.path[:0] = [CODE, os.path.join(CODE, "agent")]
os.environ["WANDB_MODE"] = "disabled"

from params import Params
from sim.harness import make_agent
from sim.fastplay import fast_play


def episode(seed):
    """One defaults-vs-starter episode, the CEM workload's unit of cost."""
    me = make_agent(Params())
    r = fast_play(me, "starter", seed=seed, steps=720)
    return r["banks"][0]


# Single-process baseline: per-core speed (TPU-VM cores may be slower than
# the CPU tier's; the per-episode second is the number that matters).
N_SINGLE = 8
t0 = time.time()
for s in range(N_SINGLE):
    episode(s)
single = (time.time() - t0) / N_SINGLE
probe["sec_per_episode_single"] = round(single, 3)
print(f"single-process: {single:.2f}s/episode over {N_SINGLE}")

# Full-width scaling: two waves at cpu_count workers, fork context (same
# methodology as the CEM kernel's pool -- fork preserves the warm imports).
workers = os.cpu_count()
n_par = 2 * workers
t0 = time.time()
with mp.get_context("fork").Pool(workers) as pool:
    pool.map(episode, range(100, 100 + n_par))
wall = time.time() - t0
probe["workers"] = workers
probe["episodes_parallel"] = n_par
probe["eps_per_sec_parallel"] = round(n_par / wall, 3)
probe["parallel_speedup"] = round(n_par * single / wall, 2)
print(f"parallel: {n_par} episodes in {wall:.1f}s at {workers} workers "
      f"(speedup {probe['parallel_speedup']:.1f}x)")

with open("/kaggle/working/probe.json", "w") as f:
    json.dump(probe, f, indent=2)
print("probe.json written")
'''


def build_script():
    """The self-extracting probe script (payload ignores cem_config.json)."""
    tarball = _build_tarball_bytes({"probe": True})
    return _HEAD + base64.b64encode(tarball).decode() + _TAIL


def main():
    username = _kaggle_username()
    slug = f"{username}/{PROBE_SLUG_SUFFIX}"
    run_dir = os.path.join(RUNS_DIR, "tpu-probe")

    script = build_script()
    print(f"=== Kaggle TPU-VM probe: {slug} ===")
    print(f"  script: {len(script):,} chars")

    # enable_tpu is the whole point; the kaggle CLI reads it as a string bool
    # (get_bool in kaggle_api_extended), same convention as enable_gpu.
    push_kernel(username, script, slug=slug,
                extra_meta={"enable_tpu": "true"})
    status = poll(username, interval=30, timeout=1800, slug=slug)
    if status != "complete":
        print(f"probe did not complete: {status}")
        print(f"  check logs: kaggle kernels output {slug}")
        sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", run_dir],
                   check=True)
    probe_path = os.path.join(run_dir, "probe.json")
    if not os.path.exists(probe_path):
        print(f"probe.json missing from output: {os.listdir(run_dir)}")
        sys.exit(1)
    with open(probe_path) as f:
        probe = json.load(f)
    print(json.dumps(probe, indent=2))

    # One small online run driver-side; the numbers are already local, so no
    # offline-sync dance is needed.
    if _wandb_api_key():
        from obs import wandb_setup
        with wandb_setup.start("tpu-probe", group="tpu-probe",
                               tags=["tpu-probe"], config=probe) as run:
            run.summary.update(probe)
        print("logged to W&B (job_type tpu-probe)")


if __name__ == "__main__":
    main()
