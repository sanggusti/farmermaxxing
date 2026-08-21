"""Push a CEM search to a Kaggle CPU notebook and download the result.

Replaces Modal as the cloud compute backend. Kaggle provides unlimited free CPU
quota (4 cores, 30 GB RAM, 12-hour sessions). A full CEM run (~7,200 episodes)
takes about 40 minutes — well within limits, and $0.

    python -m search.cem --kaggle --generations 10 --population 48 --seeds 6

Data flow:
  1. Package agent/, sim/, obs/, search/ into a tarball
  2. Upload as a Kaggle dataset version alongside cem_config.json
  3. Push a kernel script that extracts the tarball and runs the CEM loop
  4. Poll until complete
  5. Download best_params.json, results.json, generations.jsonl

One-time setup (run once before first use):

    python -m search.kaggle_nb --setup

This creates the private dataset `sanggusti/farmermaxxing-cem-code` on Kaggle.
"""

import argparse
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
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kaggle_notebook")

# These are fixed after one-time setup. The username is read from the Kaggle
# CLI config; the slugs follow Kaggle's naming convention.
DATASET_SLUG_SUFFIX = "farmermaxxing-cem-code"
KERNEL_SLUG_SUFFIX = "farmermaxxing-cem"


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


def _dataset_slug(username):
    return f"{username}/{DATASET_SLUG_SUFFIX}"


def _kernel_slug(username):
    return f"{username}/{KERNEL_SLUG_SUFFIX}"


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

def setup_dataset(username):
    """Create the private dataset on Kaggle (run once)."""
    os.makedirs(STAGING_DIR, exist_ok=True)

    meta = {
        "title": "farmermaxxing CEM code",
        "id": _dataset_slug(username),
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = os.path.join(STAGING_DIR, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Write a placeholder file so the dataset has something to upload
    placeholder = os.path.join(STAGING_DIR, "README.md")
    with open(placeholder, "w") as f:
        f.write("# farmermaxxing CEM code\n\nCode transport for Kaggle CEM runs.\n")

    subprocess.run(
        ["kaggle", "datasets", "create", "-p", STAGING_DIR],
        check=True,
    )
    print(f"dataset created: {_dataset_slug(username)}")
    print("you can now run: make search-kaggle")


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def _clean_staging():
    """Remove everything in the staging directory except dataset-metadata.json."""
    if not os.path.isdir(STAGING_DIR):
        os.makedirs(STAGING_DIR)
        return
    for name in os.listdir(STAGING_DIR):
        if name == "dataset-metadata.json":
            continue
        path = os.path.join(STAGING_DIR, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _pycache_filter(tarinfo):
    """Exclude __pycache__ and .pyc from the tarball."""
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc"):
        return None
    return tarinfo


def package_and_upload(config_dict, username):
    """Package code + config into the Kaggle dataset and upload a new version.

    The tarball contains agent/, sim/, obs/, and selected search/ files. The
    cem_config.json sits alongside the tarball (not inside it) so it survives
    Kaggle's --dir-mode skip without extraction.
    """
    _clean_staging()

    # Build the code tarball
    tarball_path = os.path.join(STAGING_DIR, "code.tar.gz")
    with tarfile.open(tarball_path, "w:gz") as tar:
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
        init_path = os.path.join(STAGING_DIR, "__search_init__.py")
        with open(init_path, "w") as f:
            f.write("")
        tar.add(init_path, arcname="search/__init__.py")
        os.remove(init_path)

    # Write the config
    config_path = os.path.join(STAGING_DIR, "cem_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Ensure dataset-metadata.json exists
    meta_path = os.path.join(STAGING_DIR, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python -m search.kaggle_nb --setup` first."
        )

    # Upload
    group = config_dict.get("group", "cem-kaggle")
    subprocess.run(
        ["kaggle", "datasets", "version", "-p", STAGING_DIR,
         "-m", f"CEM run {group}", "--dir-mode", "skip"],
        check=True,
    )
    print(f"dataset version uploaded: {_dataset_slug(username)}")


def _wait_for_dataset(username, timeout=120):
    """Poll until the dataset version is ready (processed)."""
    slug = _dataset_slug(username)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["kaggle", "datasets", "status", slug],
            capture_output=True, text=True,
        )
        output = result.stdout.strip().lower()
        if "ready" in output or "error" in output:
            return output
        time.sleep(5)
    return "timeout"


# ---------------------------------------------------------------------------
# Kernel push / poll / download
# ---------------------------------------------------------------------------

def push_kernel(username):
    """Write kernel-metadata.json and push the kernel script."""
    kernel_staging = os.path.join(STAGING_DIR, "kernel")
    os.makedirs(kernel_staging, exist_ok=True)

    # Copy the kernel script
    src = os.path.join(KERNEL_DIR, "cem_kernel.py")
    shutil.copy2(src, os.path.join(kernel_staging, "cem_kernel.py"))

    # Write kernel metadata
    meta = {
        "id": _kernel_slug(username),
        "title": "farmermaxxing CEM search",
        "code_file": "cem_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [_dataset_slug(username)],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(kernel_staging, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    subprocess.run(
        ["kaggle", "kernels", "push", "-p", kernel_staging],
        check=True,
    )
    print(f"kernel pushed: {_kernel_slug(username)}")


def poll(username, interval=30, timeout=7200):
    """Poll kernel status until completion or timeout.

    Returns the final status string: 'complete', 'error', 'cancelAcknowledged',
    or 'timeout'.
    """
    slug = _kernel_slug(username)
    deadline = time.time() + timeout
    last_status = None

    while time.time() < deadline:
        result = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True, text=True,
        )
        output = result.stdout.strip().lower()

        # Parse status from output like "has status 'running'"
        status = output
        for terminal in ("complete", "error", "cancelacknowledged"):
            if terminal in output:
                elapsed = time.time() - (deadline - timeout)
                print(f"\nkernel {terminal} after {elapsed:.0f}s")
                return terminal

        if status != last_status:
            print(f"\nkernel status: {status}")
            last_status = status
        else:
            print(".", end="", flush=True)

        time.sleep(interval)

    print(f"\ntimeout after {timeout}s")
    return "timeout"


def download(username, run_dir):
    """Download kernel output to the run directory.

    Returns the path to best_params.json, or None if not found.
    """
    os.makedirs(run_dir, exist_ok=True)
    slug = _kernel_slug(username)

    subprocess.run(
        ["kaggle", "kernels", "output", slug, "-p", run_dir],
        check=True,
    )

    best_path = os.path.join(run_dir, "best_params.json")
    if os.path.exists(best_path):
        # Validate it's parseable
        with open(best_path) as f:
            json.load(f)
        return best_path

    print("WARNING: best_params.json not found in kernel output")
    print(f"  check: kaggle kernels output {slug}")
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

        # Summary
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

    # Build the config dict from the argparse namespace
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

    # Step 1: Package and upload
    print("packaging code and config...")
    package_and_upload(config, username)

    # Step 2: Wait for dataset to be ready
    print("waiting for dataset processing...")
    ds_status = _wait_for_dataset(username)
    if "error" in str(ds_status):
        print(f"dataset processing failed: {ds_status}")
        sys.exit(1)

    # Step 3: Push the kernel
    print("pushing kernel...")
    push_kernel(username)

    # Step 4: Poll until done
    print(f"polling (timeout 2h, interval 30s)...")
    status = poll(username, interval=30, timeout=7200)

    if status != "complete":
        print(f"kernel did not complete: {status}")
        print(f"  check: kaggle kernels status {_kernel_slug(username)}")
        sys.exit(1)

    # Step 5: Download results
    print("downloading results...")
    best_path = download(username, run_dir)

    if best_path:
        print(f"\nbest_params.json: {best_path}")
        print(f"promote with:  make promote FROM={best_path}")

        # Show results summary
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            print(f"\nholdout: {results.get('best_holdout', 0):,.0f}")
            if results.get("clean_bank") is not None:
                print(f"clean:   {results['clean_bank']:,.0f}")
            print(f"wall:    {results.get('wall_seconds', 0):.0f}s")

    # Step 6: Log to W&B (unless --no-wandb)
    if not args.no_wandb:
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            print("\nlogging to W&B...")
            log_results_to_wandb(run_dir, group, config)


# ---------------------------------------------------------------------------
# CLI for setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kaggle notebook CEM setup")
    ap.add_argument("--setup", action="store_true",
                    help="one-time: create the private dataset on Kaggle")
    args = ap.parse_args()

    if args.setup:
        username = _kaggle_username()
        setup_dataset(username)
    else:
        ap.print_help()
