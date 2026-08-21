"""Push a CEM search to a Kaggle CPU notebook and download the result.

Replaces Modal as the cloud compute backend. Kaggle provides unlimited free CPU
quota (4 cores, 30 GB RAM, 12-hour sessions). A full CEM run (~7,200 episodes)
takes about 40 minutes — well within limits, and $0.

    python -m search.cem backend=kaggle generations=10 population=48 seeds=6
    python -m search.cem backend=kaggle +experiment=smoke

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


def _wandb_api_key():
    """Read WANDB_API_KEY from .env or environment."""
    # Check environment first
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    # Then .env
    env_path = os.path.join(REPO, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("WANDB_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
    return None


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
        for fname in ("cem.py", "league.py", "modal_app.py",
                      "kernel_config.py"):
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


def _generate_kernel_script(tarball_b64, wandb_api_key=None):
    """Generate a self-extracting kernel script.

    The script embeds the tarball as base64, extracts it to /tmp/fm,
    then runs the CEM loop.

    W&B runs in OFFLINE mode on Kaggle because: (1) wandb's _has_internet()
    was removed causing AttributeError on Kaggle (wandb/wandb#10967), and
    (2) Kaggle may disable internet for batch kernels. The offline run
    files are saved to /kaggle/working/wandb/ and synced locally after
    download by the orchestrator.
    """
    # Read the CEM kernel template
    template_path = os.path.join(
        os.path.dirname(__file__), "kaggle_notebook", "cem_kernel.py")
    with open(template_path) as f:
        template = f.read()

    # W&B runs offline on Kaggle — synced after download by the orchestrator.
    # Install wandb for offline logging; set key for later sync.
    if wandb_api_key:
        wandb_block = f'''
# Install wandb for offline tracking (live tracking broken on Kaggle:
# wandb/wandb#10967 _has_internet removed; Kaggle may block internet).
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "wandb", "-q", "--disable-pip-version-check",
])
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "{wandb_api_key}"
os.environ["WANDB_DISABLE_GIT"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DIR"] = "/kaggle/working"
'''
    else:
        wandb_block = '''
os.environ["WANDB_MODE"] = "disabled"
'''

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
{wandb_block}
# Extract embedded code
_PAYLOAD = """{tarball_b64}"""

CODE = "/tmp/fm"
os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_PAYLOAD)), mode="r:gz").extractall(CODE)
sys.path[:0] = [CODE, os.path.join(CODE, "agent")]

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

def push_kernel(username, script_content, slug=None, extra_meta=None):
    """Write the generated kernel script and push it.

    `slug`/`extra_meta` let another workload (the TPU probe) reuse this under
    its own kernel slot instead of clobbering a possibly-running CEM kernel
    (pushes to an active slug 409 and then overwrite it on retry).
    """
    slug = slug or _kernel_slug(username)
    kernel_staging = os.path.join(STAGING_DIR, slug.rsplit("/", 1)[-1])
    os.makedirs(kernel_staging, exist_ok=True)

    # Write the generated script
    script_path = os.path.join(kernel_staging, "cem_kernel.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Write kernel metadata — no dataset sources needed
    meta = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " "),
        "code_file": "cem_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        **(extra_meta or {}),
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
    print(f"kernel pushed: {slug}")


def poll(username, interval=30, timeout=7200, slug=None):
    """Poll kernel status until completion or timeout."""
    slug = slug or _kernel_slug(username)
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


def download(username, run_dir, slug=None):
    """Download kernel output to the run directory."""
    os.makedirs(run_dir, exist_ok=True)
    slug = slug or _kernel_slug(username)

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


def _sync_wandb_offline(run_dir, wandb_key):
    """Sync the offline W&B run created by the Kaggle kernel.

    The kernel logs to WANDB_MODE=offline with WANDB_DIR=/kaggle/working,
    producing a wandb/offline-run-*/ directory. After download, `wandb sync`
    pushes it to the server using the local API key.
    """
    import glob

    wandb_dir = os.path.join(run_dir, "wandb")
    if not os.path.isdir(wandb_dir):
        print(f"\nW&B: no wandb/ directory found in {run_dir}")
        print("  (kernel may not have reached wandb.init — check logs)")
        return

    # Find the offline run directory
    offline_runs = glob.glob(os.path.join(wandb_dir, "offline-run-*"))
    if not offline_runs:
        # Also check for run- directories (wandb sometimes uses this format)
        offline_runs = glob.glob(os.path.join(wandb_dir, "run-*"))
    if not offline_runs:
        print(f"\nW&B: no offline-run-* found in {wandb_dir}")
        print(f"  contents: {os.listdir(wandb_dir)}")
        return

    os.environ["WANDB_API_KEY"] = wandb_key
    for run_path in offline_runs:
        print(f"\nsyncing W&B offline run: {os.path.basename(run_path)}")
        result = subprocess.run(
            ["wandb", "sync", run_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  synced successfully")
            # Extract the run URL from output
            for line in result.stdout.splitlines():
                if "wandb.ai" in line or "https://" in line:
                    print(f"  {line.strip()}")
        else:
            print(f"  sync failed: {result.stderr.strip()}")
            # Fall back to replay
            print("  falling back to replay from generations.jsonl")


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
# Main entry point (called from cem.py when backend=kaggle)
# ---------------------------------------------------------------------------

def run_cem_on_kaggle(cfg):
    """Package, push, poll, download — the full Kaggle CEM workflow.

    `cfg` is the COMPOSED configs/cem.yaml (an omegaconf DictConfig), shipped
    whole. The predecessor of this function hand-copied 14 named args into a
    dict, so a flag it had not been taught (--ramp) was dropped WITHOUT ERROR;
    search/kernel_config.py now makes the kernel refuse any key mismatch
    instead, in its first minute.
    """
    from omegaconf import OmegaConf   # driver-side only; not in the kernel

    username = _kaggle_username()

    config = OmegaConf.to_container(cfg, resolve=True)
    config.pop("backend")   # consumed by the routing in search.cem.main
    config["group"] = (config["group"]
                       or f"cem-kaggle-g{cfg.generations}-p{cfg.population}")

    # Embed init params data if provided (paths don't transfer)
    init_params = config.pop("init_params")
    if init_params:
        sys.path[:0] = [REPO, os.path.join(REPO, "agent")]
        from params import Params
        p = Params.from_json(init_params)
        config["init_params_data"] = p.__dict__

    group = config["group"]
    run_dir = os.path.join(RUNS_DIR, group)

    print(f"=== Kaggle CEM: {group} ===")
    print(f"  generations: {config['generations']}")
    print(f"  population:  {config['population']}")
    print(f"  seeds:       {config['seeds']}")
    print(f"  opponents:   {config['opponents']}")
    print()

    # Step 1: Build the self-extracting kernel script
    wandb_key = _wandb_api_key() if cfg.wandb else None
    print("building kernel script with embedded code...")
    if wandb_key:
        print("  W&B: offline logging on Kaggle, sync after download")
    else:
        print("  W&B: disabled" + (" (no key found)" if cfg.wandb else " (wandb=false)"))
    tarball_bytes = _build_tarball_bytes(config)
    tarball_b64 = base64.b64encode(tarball_bytes).decode()
    script = _generate_kernel_script(tarball_b64, wandb_api_key=wandb_key)
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

    # Step 5: Sync W&B
    if cfg.wandb and wandb_key:
        # The kernel logged offline to wandb/ dir. Sync it now.
        _sync_wandb_offline(run_dir, wandb_key)
    elif cfg.wandb:
        # No API key — replay from generations.jsonl
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            print("\nreplaying results to W&B...")
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
        # Size the REAL composed config, not a stub: the config travels in
        # the tarball, so the size check should see what actually ships.
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=os.path.join(REPO, "configs"),
                                   version_base=None):
            config = OmegaConf.to_container(compose(config_name="cem"),
                                            resolve=True)
        tb = _build_tarball_bytes(config)
        b64 = base64.b64encode(tb).decode()
        script = _generate_kernel_script(b64)
        print(f"tarball: {len(tb):,} bytes")
        print(f"base64:  {len(b64):,} chars")
        print(f"script:  {len(script):,} chars")
        print(f"script size: {len(script.encode()):,} bytes")
    else:
        ap.print_help()
