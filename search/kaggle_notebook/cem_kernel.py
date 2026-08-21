#!/usr/bin/env python3
"""CEM search running inside a Kaggle CPU notebook.

Mirrors the generation loop in search.cem.main() but reads config from a JSON
file (no argparse) and writes output to /kaggle/working/ (no W&B streaming).
The local orchestrator (search.kaggle_nb) pushes this script, polls for
completion, and downloads the results.

The code itself travels as a tarball in a Kaggle dataset mounted at
/kaggle/input/farmermaxxing-cem-code/. This keeps the notebook thin and the
code byte-identical to what runs locally.
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
# 2. Extract code from the dataset
# ---------------------------------------------------------------------------
DATASET = "/kaggle/input/farmermaxxing-cem-code"
CODE = "/tmp/fm"
tarfile.open(os.path.join(DATASET, "code.tar.gz")).extractall(CODE)
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
from sim.opponents import resolve_pool                        # noqa: E402
from sim.arena import CENSUS_KEYS as ARENA_CENSUS_KEYS        # noqa: E402
from search.league import (build_cells, normalised_fitness,   # noqa: E402
                           worst_opponent)
from search.cem import (                                      # noqa: E402
    initial_distribution, sample, refit, score_local,
    selection_score, HOLDOUT_OFFSET, CLEAN_OFFSET,
    WORST_TOLERANCE, WORST_TOLERANCE_FLOOR,
)

# ---------------------------------------------------------------------------
# 5. Resolve configuration
# ---------------------------------------------------------------------------
OUTPUT = "/kaggle/working"

generations = cfg["generations"]
population = cfg["population"]
elite_frac = cfg.get("elite_frac", 0.25)
seeds = cfg["seeds"]
train_pool_size = cfg.get("train_pool", 1000)
holdout_seeds_n = cfg.get("holdout_seeds", 6)
clean_seeds_n = cfg.get("clean_seeds", 8)
steps = cfg.get("steps", 720)
fitness_key = cfg.get("fitness", "bank")
holdout_opponents_n = cfg.get("holdout_opponents", 0)
rng_seed = cfg.get("rng_seed", 0)
group = cfg.get("group", "cem-kaggle")

rng = random.Random(rng_seed)
holdout_seeds = [HOLDOUT_OFFSET + i for i in range(holdout_seeds_n)]
clean_seeds = [CLEAN_OFFSET + i for i in range(clean_seeds_n)]

# Resolve opponent pools
train_pool_spec = cfg.get("opponents") or cfg.get("opponent", "starter")
ref_pool_spec = cfg.get("reference") or train_pool_spec
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
init_params_data = cfg.get("init_params_data")
if init_params_data:
    base = Params(**init_params_data)
    spread = cfg.get("init_spread") or 0.10
else:
    base = Params()
    spread = cfg.get("init_spread") or 0.25

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

for gen in range(generations):
    gt = time.time()
    population_vecs = [sample(mean, std, rng) for _ in range(population)]
    if gen == 0:
        population_vecs[0] = flatten(base)

    # Rotate training seeds
    train_seeds = [(gen * seeds + i) % train_pool_size for i in range(seeds)]
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

    # Worst-opponent guard
    worst_label, worst_margin = worst_opponent(champion_stats)
    tolerance = max(WORST_TOLERANCE * abs(best_worst or 0.0),
                    WORST_TOLERANCE_FLOOR)
    regressed = (best_worst is not None
                 and worst_margin < best_worst - tolerance)
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
    row["wall_seconds"] = time.time() - gt

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
