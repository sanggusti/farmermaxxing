"""Push a CEM search to a Kaggle CPU notebook and download the result.

Replaces Modal as the cloud compute backend. Kaggle provides unlimited free CPU
quota (4 cores, 30 GB RAM, 12-hour sessions). A full CEM run (~7,200 episodes)
takes about 40 minutes — well within limits, and $0.

    python -m search.cem --kaggle --generations 10 --population 48 --seeds 6

The kernel script is generated dynamically with all code and configuration
embedded as a base64-encoded tarball. No separate dataset needed — everything
travels in one self-extracting script.

One-time setup is no longer required.
"""

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, "runs")
STAGING_DIR = os.path.join(REPO, ".kaggle-staging")

KERNEL_SLUG_SUFFIX = "farmermaxxing-cem-search"


def _kaggle_username():
    """Read the Kaggle username from the CLI config."""
    result = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True, text=True, check=True,
    )
    for line in result.stdout.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Could not determine Kaggle username from `kaggle config view`")


def _kernel_slug(username):
    return f"{username}/{KERNEL_SLUG_SUFFIX}"


# ---------------------------------------------------------------------------
# Packaging — build a self-extracting kernel script
# ---------------------------------------------------------------------------

def _pycache_filter(tarinfo):
    """Exclude __pycache__ and .pyc from the tarball."""
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc"):
        return None
    return tarinfo


def _build_tarball_bytes(config_dict):
    """Build an in-memory tarball of the code + config.

    Returns the gzipped bytes. The tarball contains agent/, sim/, obs/,
    selected search/ files, and cem_config.json.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirname in ("agent", "sim", "obs"):
            src = os.path.join(REPO, dirname)
            tar.add(src, arcname=dirname, filter=_pycache_filter)
        # Selected search files — only the ones the kernel imports
        for fname in ("cem.py", "league.py", "modal_app.py"):
            src = os.path.join(REPO, "search", fname)
            if os.path.exists(src):
                tar.add(src, arcname=f"search/{fname}",
                        filter=_pycache_filter)
        # search/__init__.py so `from search.xxx import` works
        init_data = b""
        info = tarfile.TarInfo(name="search/__init__.py")
        info.size = 0
        tar.addfile(info, io.BytesIO(init_data))

        # Embed cem_config.json in the tarball
        config_bytes = json.dumps(config_dict, indent=2).encode()
        info = tarfile.TarInfo(name="cem_config.json")
        info.size = len(config_bytes)
        tar.addfile(info, io.BytesIO(config_bytes))

    return buf.getvalue()


def _generate_kernel_script(tarball_b64):
    """Generate a self-extracting kernel script.

    The script embeds the tarball as base64, extracts it to /tmp/fm,
    then runs the CEM loop.
    """
    # Read the CEM kernel template
    template_path = os.path.join(
        os.path.dirname(__file__), "kaggle_notebook", "cem_kernel.py")
    with open(template_path) as f:
        template = f.read()

    # Build the self-extracting preamble
    preamble = f'''#!/usr/bin/env python3
"""Auto-generated CEM search script for Kaggle. Do not edit directly."""

import base64
import io
import json
import os
import random
import statistics
import subprocess
import sys
import tarfile
import time

# Install the pinned engine version
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "kaggle-environments==1.32.4", "-q", "--disable-pip-version-check",
])

# Extract embedded code
_PAYLOAD = """{tarball_b64}"""

CODE = "/tmp/fm"
os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PAYLOAD)), mode="r:gz").extractall(CODE)
sys.path[:0] = [CODE, os.path.join(CODE, "agent")]
os.environ["WANDB_MODE"] = "disabled"

print(f"code extracted to {{CODE}}")
print(f"agent files: {{os.listdir(os.path.join(CODE, 'agent'))}}")

# Load configuration from the tarball
with open(os.path.join(CODE, "cem_config.json")) as f:
    cfg = json.load(f)

'''

    # Now append the imports + CEM loop from the template. Skip everything
    # up to the "4. Imports" section.
    marker = "# 4. Imports (after sys.path is set)"
    idx = template.find(marker)
    if idx == -1:
        raise ValueError("Cannot find imports marker in cem_kernel.py template")
    # Get everything from imports onward
    rest = template[idx:]

    return preamble + rest


# ---------------------------------------------------------------------------
# Kernel push / poll / download
# ---------------------------------------------------------------------------

def push_kernel(username, script_content):
    """Write the generated kernel script and push it."""
    kernel_staging = os.path.join(STAGING_DIR, "kernel")
    os.makedirs(kernel_staging, exist_ok=True)

    # Write the generated script
    script_path = os.path.join(kernel_staging, "cem_kernel.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Write kernel metadata — no dataset sources needed
    meta = {
        "id": _kernel_slug(username),
        "title": "farmermaxxing cem search",
        "code_file": "cem_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(kernel_staging, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", kernel_staging],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        # 409 means a previous version is still running/error; retry after wait
        if "409" in stderr or "Conflict" in stderr or "409" in stdout:
            print("previous kernel version still active, waiting 30s and retrying...")
            time.sleep(30)
            subprocess.run(
                ["kaggle", "kernels", "push", "-p", kernel_staging],
                check=True,
            )
        else:
            print(f"kernel push failed:\n  stdout: {stdout}\n  stderr: {stderr}")
            raise subprocess.CalledProcessError(result.returncode,
                                                result.args, stdout, stderr)
    print(f"kernel pushed: {_kernel_slug(username)}")


def poll(username, interval=30, timeout=7200):
    """Poll kernel status until completion or timeout."""
    slug = _kernel_slug(username)
    deadline = time.time() + timeout
    last_status = None

    while time.time() < deadline:
        result = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True, text=True,
        )
        output = result.stdout.strip().lower()

        for terminal in ("complete", "error", "cancelacknowledged"):
            if terminal in output:
                elapsed = time.time() - (deadline - timeout)
                print(f"\nkernel {terminal} after {elapsed:.0f}s")
                return terminal

        if output != last_status:
            print(f"\nkernel status: {output}")
            last_status = output
        else:
            print(".", end="", flush=True)

        time.sleep(interval)

    print(f"\ntimeout after {timeout}s")
    return "timeout"


def download(username, run_dir):
    """Download kernel output to the run directory."""
    os.makedirs(run_dir, exist_ok=True)
    slug = _kernel_slug(username)

    subprocess.run(
        ["kaggle", "kernels", "output", slug, "-p", run_dir],
        check=True,
    )

    best_path = os.path.join(run_dir, "best_params.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            json.load(f)  # validate
        return best_path

    print("WARNING: best_params.json not found in kernel output")
    print(f"  files downloaded: {os.listdir(run_dir)}")
    return None


def log_results_to_wandb(run_dir, group, config):
    """Replay the Kaggle CEM results into W&B, after the fact."""
    sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
    from obs import wandb_setup

    results_path = os.path.join(run_dir, "results.json")
    gen_path = os.path.join(run_dir, "generations.jsonl")

    with open(results_path) as f:
        results = json.load(f)

    with wandb_setup.start("cem", group=group, tags=["cem", "kaggle"],
                           config={**config, "backend": "kaggle"}) as run:
        # Replay per-generation metrics
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                for line in f:
                    row = json.loads(line)
                    run.log(row)

        run.summary["best_holdout_bank"] = results.get("best_holdout")
        run.summary["best_train_bank"] = results.get("best_train")
        if results.get("clean_bank") is not None:
            run.summary["clean_bank"] = results["clean_bank"]
            run.summary["clean_min_bank"] = results.get("clean_min_bank")
            run.summary["clean_selection_score"] = results.get(
                "clean_selection_score")
            run.summary["selection_bias"] = results.get("selection_bias")

        best_path = os.path.join(run_dir, "best_params.json")
        if os.path.exists(best_path):
            wandb_setup.log_params_artifact(
                run, best_path,
                metadata={"holdout_mean_bank": results.get("best_holdout")})

    print(f"W&B run logged: {group}")


# ---------------------------------------------------------------------------
# Main entry point (called from cem.py --kaggle)
# ---------------------------------------------------------------------------

def run_cem_on_kaggle(args):
    """Package, push, poll, download — the full Kaggle CEM workflow."""
    username = _kaggle_username()

    config = {
        "generations": args.generations,
        "population": args.population,
        "elite_frac": args.elite_frac,
        "seeds": args.seeds,
        "train_pool": args.train_pool,
        "holdout_seeds": args.holdout_seeds,
        "clean_seeds": args.clean_seeds,
        "steps": args.steps,
        "opponents": args.opponents or args.opponent,
        "reference": args.reference,
        "fitness": args.fitness,
        "holdout_opponents": args.holdout_opponents,
        "rng_seed": args.rng_seed,
        "group": args.group or f"cem-kaggle-g{args.generations}-p{args.population}",
    }

    # Embed init params data if provided (paths don't transfer)
    if args.init_params:
        sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
        from params import Params
        p = Params.from_json(args.init_params)
        config["init_params_data"] = p.__dict__
        config["init_spread"] = args.init_spread

    group = config["group"]
    run_dir = os.path.join(RUNS_DIR, group)

    print(f"=== Kaggle CEM: {group} ===")
    print(f"  generations: {config['generations']}")
    print(f"  population:  {config['population']}")
    print(f"  seeds:       {config['seeds']}")
    print(f"  opponents:   {config['opponents']}")
    print()

    # Step 1: Build the self-extracting kernel script
    print("building kernel script with embedded code...")
    tarball_bytes = _build_tarball_bytes(config)
    tarball_b64 = base64.b64encode(tarball_bytes).decode()
    script = _generate_kernel_script(tarball_b64)
    print(f"  tarball: {len(tarball_bytes):,} bytes")
    print(f"  script:  {len(script):,} chars")

    # Step 2: Push
    os.makedirs(STAGING_DIR, exist_ok=True)
    print("pushing kernel...")
    push_kernel(username, script)

    # Step 3: Poll until done
    print(f"polling (timeout 2h, interval 30s)...")
    status = poll(username, interval=30, timeout=7200)

    if status != "complete":
        print(f"kernel did not complete: {status}")
        print(f"  check logs: kaggle kernels output {_kernel_slug(username)}")
        sys.exit(1)

    # Step 4: Download results
    print("downloading results...")
    best_path = download(username, run_dir)

    if best_path:
        print(f"\nbest_params.json: {best_path}")
        print(f"promote with:  make promote FROM={best_path}")

        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            print(f"\nholdout: {results.get('best_holdout', 0):,.0f}")
            if results.get("clean_bank") is not None:
                print(f"clean:   {results['clean_bank']:,.0f}")
            print(f"wall:    {results.get('wall_seconds', 0):.0f}s")

    # Step 5: Log to W&B (unless --no-wandb)
    if not args.no_wandb:
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            print("\nlogging to W&B...")
            log_results_to_wandb(run_dir, group, config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kaggle notebook CEM backend")
    ap.add_argument("--test-package", action="store_true",
                    help="build the script and print its size, but don't push")
    args = ap.parse_args()

    if args.test_package:
        config = {"generations": 1, "population": 4, "seeds": 1,
                  "opponents": "starter", "group": "test"}
        tb = _build_tarball_bytes(config)
        b64 = base64.b64encode(tb).decode()
        script = _generate_kernel_script(b64)
        print(f"tarball: {len(tb):,} bytes")
        print(f"base64:  {len(b64):,} chars")
        print(f"script:  {len(script):,} chars")
        print(f"script size: {len(script.encode()):,} bytes")
    else:
        ap.print_help()
