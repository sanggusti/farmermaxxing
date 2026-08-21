#!/usr/bin/env python3
"""CEM search running inside a Kaggle CPU notebook.

Mirrors the generation loop in search.cem.main() but reads config from
cem_config.json (the COMPOSED configs/cem.yaml, shipped whole -- validated by
search.kernel_config so a key mismatch fails in the first minute) and writes
output to /kaggle/working/. The local orchestrator (search.kaggle_nb) pushes
this script, polls for completion, downloads the results, and syncs the
offline W&B run.

Only the part from the "# 4. Imports" marker down actually ships: the
orchestrator's _generate_kernel_script slices there and prepends its own
self-extracting preamble (base64 tarball -> /tmp/fm, sys.path, cfg =
json.load). Sections 1-3 below exist so this file also runs standalone
against a mounted dataset, the pre-base64 delivery path.
"""

import json
import os
import random
import statistics
import subprocess
import sys
import tarfile
import time

# ---------------------------------------------------------------------------
# 1. Install the pinned engine version
# ---------------------------------------------------------------------------
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "kaggle-environments==1.32.4", "-q", "--disable-pip-version-check",
])

# ---------------------------------------------------------------------------
# 2. Locate the code from the dataset
# ---------------------------------------------------------------------------
# Kaggle auto-extracts .tar.gz uploads. The extracted tree may live directly
# under the dataset mount, under a code/ subdirectory, or the tarball may
# still be present as-is. Try all three.
DATASET = "/kaggle/input/farmermaxxing-cem-code"

# Debug: show what's actually in the mount
print("dataset mount contents:")
for root, dirs, files in os.walk(DATASET):
    depth = root.replace(DATASET, "").count(os.sep)
    if depth <= 2:  # don't recurse too deep
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth < 2:
            for f in files[:10]:
                print(f"{indent}  {f}")

CODE = None
# Option 1: auto-extracted into code/ subdirectory
if os.path.isdir(os.path.join(DATASET, "code", "agent")):
    CODE = os.path.join(DATASET, "code")
# Option 2: extracted flat into the dataset root
elif os.path.isdir(os.path.join(DATASET, "agent")):
    CODE = DATASET
# Option 3: tarball still present, extract manually
elif os.path.isfile(os.path.join(DATASET, "code.tar.gz")):
    CODE = "/tmp/fm"
    tarfile.open(os.path.join(DATASET, "code.tar.gz")).extractall(CODE)
else:
    raise FileNotFoundError(
        f"Cannot find code in {DATASET}. Contents: "
        f"{os.listdir(DATASET)}"
    )

print(f"using CODE={CODE}")
sys.path[:0] = [CODE, os.path.join(CODE, "agent")]
os.environ["WANDB_MODE"] = "disabled"

# ---------------------------------------------------------------------------
# 3. Load configuration
# ---------------------------------------------------------------------------
with open(os.path.join(DATASET, "cem_config.json")) as f:
    cfg = json.load(f)

# ---------------------------------------------------------------------------
# 4. Imports (after sys.path is set)
# ---------------------------------------------------------------------------
from params import Params, SEARCH_SPACE, flatten, unflatten  # noqa: E402
from obs import wandb_setup                                   # noqa: E402
from sim.opponents import resolve_pool                        # noqa: E402
from sim.arena import CENSUS_KEYS as ARENA_CENSUS_KEYS        # noqa: E402
from search.league import (build_cells, normalised_fitness,   # noqa: E402
                           worst_opponent)
from search.cem import (                                      # noqa: E402
    initial_distribution, sample, refit, ramp_schedule,
    selection_score, worst_tolerance, HOLDOUT_OFFSET, CLEAN_OFFSET,
)
from search.kernel_config import resolve_cem_config            # noqa: E402
from search.modal_app import summarise_cells                   # noqa: E402

# ---------------------------------------------------------------------------
# 4b. Parallel score_local for Kaggle (4 cores available)
# ---------------------------------------------------------------------------
import multiprocessing as mp

def _run_one(args):
    """Score one (candidate, cell) pair. Picklable for multiprocessing."""
    vec, opp, seed, seat, steps, metrics = args
    from params import unflatten as _unflatten
    from sim.fastplay import fast_play
    from sim.harness import make_agent
    params = _unflatten(vec)
    me = make_agent(params)
    a, b = (me, opp) if seat == 0 else (opp, me)
    r = fast_play(a, b, seed=seed, steps=steps, metrics=metrics)
    row = {"bank": r["banks"][seat], "opp_bank": r["banks"][1 - seat],
           "status": r["statuses"][seat]}
    if "metrics" in r:
        row.update(r["metrics"][seat])
    return row

# Kaggle provides 4 cores; use them all.
_WORKERS = int(os.environ.get("FM_WORKERS", "4"))

def score_local(vectors, cells, steps, metrics=False):
    """Parallel version of search.cem.score_local for Kaggle's 4-core CPUs.

    ~51k episodes at 1.3s each would take ~18h single-threaded, exceeding
    Kaggle's 12h limit. With 4 workers: ~4.7h, safely within budget.
    """
    work = []
    owner = []
    for i, vec in enumerate(vectors):
        for opp, _label, seed, seat in cells:
            work.append((vec, opp, seed, seat, steps, metrics))
            owner.append(i)

    labels = [c[1] for c in cells]

    # fork preserves the warm imports
    ctx = mp.get_context("fork")
    with ctx.Pool(_WORKERS) as pool:
        flat = pool.map(_run_one, work)

    results = [[] for _ in vectors]
    for idx, r in zip(owner, flat):
        results[idx].append(r)
    return [summarise_cells(rows, labels) for rows in results]

# ---------------------------------------------------------------------------
# 5. Resolve configuration
# ---------------------------------------------------------------------------
OUTPUT = "/kaggle/working"

# The config is the full composed configs/cem.yaml, so every key is indexed
# directly: a missing key is a loud KeyError, never a shadow default that can
# drift from the yaml. resolve_cem_config already refused any key mismatch.
cfg = resolve_cem_config(cfg)

generations = cfg["generations"]
population = cfg["population"]
elite_frac = cfg["elite_frac"]
seeds = cfg["seeds"]
train_pool_size = cfg["train_pool"]
holdout_seeds_n = cfg["holdout_seeds"]
clean_seeds_n = cfg["clean_seeds"]
steps = cfg["steps"]
fitness_key = cfg["fitness"]
holdout_opponents_n = cfg["holdout_opponents"]
rng_seed = cfg["rng_seed"]
group = cfg["group"]

# Same ramp semantics as search.cem.main: the schedule sums to
# generations * seeds exactly, and at ramp=1.0 the cumulative starts make
# this bit-for-bit the legacy (gen * seeds + i) % pool rotation (pinned by
# test_constant_ramp_reproduces_legacy_formula). The old kernel silently ran
# the constant allocation whatever ramp was asked for -- the founding case
# for the resolve_cem_config unknown-key check above.
seeds_schedule = ramp_schedule(generations, seeds, cfg["ramp"])
seed_starts = [0]
for n in seeds_schedule:
    seed_starts.append(seed_starts[-1] + n)

rng = random.Random(rng_seed)
holdout_seeds = [HOLDOUT_OFFSET + i for i in range(holdout_seeds_n)]
clean_seeds = [CLEAN_OFFSET + i for i in range(clean_seeds_n)]

# Resolve opponent pools
train_pool_spec = cfg["opponents"]
ref_pool_spec = cfg["reference"] or train_pool_spec
train_opps, train_labels = resolve_pool(train_pool_spec)
ref_opps, ref_labels = resolve_pool(ref_pool_spec)

# Opponent hold-out
heldout_labels = []
if holdout_opponents_n > 0:
    keep = list(range(len(train_labels)))
    drop = sorted(rng.sample(keep, holdout_opponents_n))
    heldout_labels = [train_labels[i] for i in drop]
    train_opps = [o for i, o in enumerate(train_opps) if i not in drop]
    train_labels = [l for i, l in enumerate(train_labels) if i not in drop]
    print(f"held out of training: {heldout_labels}")
    print(f"training on         : {train_labels}")

holdout_cells = build_cells(ref_opps, ref_labels, holdout_seeds)
clean_cells = build_cells(ref_opps, ref_labels, clean_seeds)

# Warm start or cold start
init_params_data = cfg["init_params_data"]
if init_params_data:
    base = Params(**init_params_data)
    spread = cfg["init_spread"] if cfg["init_spread"] is not None else 0.10
else:
    base = Params()
    spread = cfg["init_spread"] if cfg["init_spread"] is not None else 0.25

mean, std = initial_distribution(base, spread)
n_elite = max(2, int(population * elite_frac))
sel_key = "mean_margin" if fitness_key == "margin" else "mean_bank"

# ---------------------------------------------------------------------------
# 6. CEM generation loop (mirrors cem.py lines 324-501)
# ---------------------------------------------------------------------------
best_holdout, best_vec, best_train = float("-inf"), None, None
best_worst = None
gen_log = open(os.path.join(OUTPUT, "generations.jsonl"), "w")
t0 = time.time()

# W&B (offline on Kaggle; the orchestrator syncs after download). When
# WANDB_MODE=disabled (no API key), wandb_setup.start() returns a _NullRun
# whose .log() is a no-op, so the code below never branches.
# The FULL shipped config plus the derived values, matching the local driver;
# init_params_data is a whole parameter set, summarised as a flag instead.
wandb_run = wandb_setup.start("cem", group=group, tags=["cem", "kaggle"], config={
    **{k: v for k, v in cfg.items() if k != "init_params_data"},
    "init_params": "inline" if init_params_data else "defaults",
    "init_spread": spread,
    "train_opponents": train_labels, "reference_opponents": ref_labels,
    "heldout_opponents": heldout_labels,
    "train_cells_per_gen": len(train_opps) * seeds * 2,
    "seeds_schedule": seeds_schedule,
    "train_episodes_total":
        population * len(train_opps) * 2 * sum(seeds_schedule),
    "backend": "kaggle",
})

for gen in range(generations):
    gt = time.time()
    population_vecs = [sample(mean, std, rng) for _ in range(population)]
    if gen == 0:
        population_vecs[0] = flatten(base)

    # Rotate training seeds; the count comes from the ramp schedule, the
    # cumulative start keeps blocks consecutive (same as search.cem.main).
    train_seeds = [(seed_starts[gen] + i) % train_pool_size
                   for i in range(seeds_schedule[gen])]
    train_cells = build_cells(train_opps, train_labels, train_seeds)

    stats = score_local(population_vecs, train_cells, steps)
    key = "margins" if fitness_key == "margin" else "banks"
    fitness = normalised_fitness([s[key] for s in stats])
    ranked = sorted(zip(fitness, stats, population_vecs), key=lambda t: -t[0])
    elites = [vec for _, _, vec in ranked[:n_elite]]

    # Re-score elites on holdout with census metrics
    hold_stats = score_local(elites, holdout_cells, steps, metrics=True)
    hold_ranked = sorted(zip(hold_stats, elites),
                         key=lambda sp: -selection_score(sp[0], sel_key))
    champion_stats, champion_vec = hold_ranked[0]

    # Worst-opponent guard, via the shared helper so the tolerance cannot
    # drift from the other drivers (cmaes and subspace already import it).
    worst_label, worst_margin = worst_opponent(champion_stats)
    regressed = (best_worst is not None
                 and worst_margin < best_worst - worst_tolerance(best_worst))
    if selection_score(champion_stats, sel_key) > best_holdout and not regressed:
        best_holdout = selection_score(champion_stats, sel_key)
        best_worst = (worst_margin if best_worst is None
                      else max(best_worst, worst_margin))
        best_vec = champion_vec
        best_train = ranked[0][1]["mean_bank"]
        unflatten(best_vec).to_json(os.path.join(OUTPUT, "best_params.json"))

    new_mean, new_std = refit(elites)
    mean = {k: (1 - 0.3) * new_mean[k] + 0.3 * mean[k] for k in mean}
    std = new_std

    train_best = ranked[0][1]["mean_bank"]
    row = {
        "gen": gen,
        "train_best_bank": train_best,
        "train_pop_mean_bank": statistics.mean(
            [s["mean_bank"] for s in stats]),
        "train_elite_mean_bank": statistics.mean(
            [s["mean_bank"] for _, s, _ in ranked[:n_elite]]),
        "holdout_best_bank": champion_stats["mean_bank"],
        "holdout_win_rate": champion_stats["win_rate"],
        "holdout_min_bank": champion_stats["min_bank"],
        "generalisation_gap": train_best - champion_stats["mean_bank"],
        "best_holdout_overall": best_holdout,
        "selection_metric": sel_key,
    }
    for ckey in ARENA_CENSUS_KEYS:
        if f"mean_{ckey}" in champion_stats:
            row[f"holdout_{ckey}"] = champion_stats[f"mean_{ckey}"]
    for label, b in (champion_stats.get("by_opponent") or {}).items():
        row[f"vs/{label}/mean_bank"] = b["mean_bank"]
        row[f"vs/{label}/win_rate"] = b["win_rate"]
    row["worst_opponent_margin"] = worst_margin
    # The ramp's audit trail, same as the local driver: cumulative episodes
    # must land on exactly population * opps * 2 * generations * seeds.
    row["train_seeds_this_gen"] = seeds_schedule[gen]
    row["episodes_this_gen"] = population * len(train_cells)
    row["cum_train_episodes"] = (
        population * len(train_opps) * 2 * seed_starts[gen + 1])
    row["wall_seconds"] = time.time() - gt

    wandb_run.log(row)
    gen_log.write(json.dumps(row) + "\n")
    gen_log.flush()
    print(f"gen {gen:>2}  train {train_best:>11,.0f}  "
          f"holdout {champion_stats['mean_bank']:>11,.0f}  "
          f"gap {row['generalisation_gap']:>10,.0f}  "
          f"win {champion_stats['win_rate']:.0%}  "
          f"worst {worst_label} {worst_margin:>+10,.0f}"
          f"  [{time.time() - gt:.0f}s]"
          + ("  [rejected: worst regressed]" if regressed else ""))

# ---------------------------------------------------------------------------
# 7. Clean evaluation
# ---------------------------------------------------------------------------
clean = None
if best_vec is not None:
    clean = score_local([best_vec], clean_cells, steps, metrics=True)[0]

# ---------------------------------------------------------------------------
# 8. Write results
# ---------------------------------------------------------------------------
results = {
    "best_holdout": best_holdout,
    "best_train": best_train,
    "selection_metric": sel_key,
    "generations_completed": generations,
    "group": group,
    "wall_seconds": time.time() - t0,
    "train_opponents": train_labels,
    "reference_opponents": ref_labels,
    "heldout_opponents": heldout_labels,
}
if clean is not None:
    results["clean_bank"] = clean["mean_bank"]
    results["clean_min_bank"] = clean["min_bank"]
    clean_sel = selection_score(clean, sel_key)
    results["clean_selection_score"] = clean_sel
    results["selection_bias"] = best_holdout - clean_sel
    results["clean_by_opponent"] = {
        label: {"mean_bank": b["mean_bank"], "win_rate": b["win_rate"],
                "mean_margin": b.get("mean_margin", 0)}
        for label, b in (clean.get("by_opponent") or {}).items()
    }

with open(os.path.join(OUTPUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

gen_log.close()

# W&B summary and artifact
wandb_run.summary["best_holdout_bank"] = results.get("best_holdout")
wandb_run.summary["best_train_bank"] = results.get("best_train")
if results.get("clean_bank") is not None:
    wandb_run.summary["clean_bank"] = results["clean_bank"]
    wandb_run.summary["clean_min_bank"] = results.get("clean_min_bank")
    wandb_run.summary["clean_selection_score"] = results.get("clean_selection_score")
    wandb_run.summary["selection_bias"] = results.get("selection_bias")
    for label, b in (results.get("clean_by_opponent") or {}).items():
        wandb_run.summary[f"clean_vs/{label}/mean_bank"] = b["mean_bank"]
        wandb_run.summary[f"clean_vs/{label}/win_rate"] = b["win_rate"]
best_path = os.path.join(OUTPUT, "best_params.json")
if os.path.exists(best_path):
    wandb_setup.log_params_artifact(
        wandb_run, best_path,
        metadata={"holdout_mean_bank": results.get("best_holdout")})
wandb_run.finish()

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
unit = "margin" if fitness_key == "margin" else "bank"
print(f"\nselection holdout : {best_holdout:>12,.0f}  (mean {unit})")
if clean is not None:
    bias = best_holdout - selection_score(clean, sel_key)
    print(f"clean (unbiased)  : {clean['mean_bank']:>12,.0f}  "
          f"worst {clean['min_bank']:,.0f}")
    print("clean per opponent:")
    held = set(heldout_labels)
    for label, b in sorted((clean.get("by_opponent") or {}).items()):
        mark = "h" if label in held else ("T" if held else " ")
        print(f"  {mark} {label:<22} bank {b['mean_bank']:>11,.0f}   "
              f"win {b['win_rate']:>6.1%}   "
              f"margin {b.get('mean_margin', float('nan')):>+11,.0f}")
    pct = f"{bias / abs(clean_sel):+.1%}" if clean_sel else "n/a"
    print(f"selection bias    : {bias:>+12,.0f}  ({pct})")
print(f"\ntotal wall clock  : {time.time() - t0:.0f}s")
print(f"\nbest_params.json written to {OUTPUT}")
