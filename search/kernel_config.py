"""The Kaggle kernel's config contract: every key accounted for, loudly.

The pre-Hydra backend rebuilt its config dict from args by hand, so a flag it
had not been taught was dropped WITHOUT ERROR -- a run asking for a ramp would
silently get the constant allocation, exactly the shape of bug rule 7 exists
for, and search/cem.py carried a special-case refusal of --ramp under --kaggle
to paper over it. Now the composed config travels whole and this module makes
the kernel refuse, in its first seconds, any key it does not know (a new yaml
key without kernel support) and any key it needs but did not get (a kernel
change without yaml support). One check replaces that guard for every future
key at once.

Ships inside the kernel tarball; stdlib only, importable both locally (for the
tests) and on Kaggle.
"""

# Every key of the composed configs/cem.yaml after run_cem_on_kaggle's two
# transforms: `backend` is consumed by the driver-side routing, and the
# `init_params` PATH becomes inlined `init_params_data` (paths don't transfer).
KNOWN_KEYS = frozenset({
    "generations", "population", "elite_frac", "seeds", "ramp",
    "train_pool", "holdout_seeds", "clean_seeds", "steps",
    "opponents", "reference", "fitness", "holdout_opponents",
    "rng_seed", "group", "wandb", "init_params_data", "init_spread",
})

# Present only on warm starts; everything else must always arrive.
_OPTIONAL = frozenset({"init_params_data"})


def resolve_cem_config(cfg):
    """Validate the shipped config dict; return it with optionals defaulted.

    Raises SystemExit naming the offending keys, so a mismatch between
    configs/cem.yaml and the kernel fails in the kernel's first minute
    instead of producing a complete run of the wrong experiment.
    """
    unknown = sorted(set(cfg) - KNOWN_KEYS)
    if unknown:
        raise SystemExit(
            f"error: cem_config.json carries keys this kernel does not "
            f"implement: {unknown}. A new configs/cem.yaml key needs kernel "
            f"support in search/kaggle_notebook/cem_kernel.py (and a "
            f"KNOWN_KEYS entry) before it can run on Kaggle.")
    missing = sorted(KNOWN_KEYS - _OPTIONAL - set(cfg))
    if missing:
        raise SystemExit(
            f"error: cem_config.json is missing keys this kernel requires: "
            f"{missing}. The config must be the full composed configs/cem.yaml "
            f"(see search/kaggle_nb.py run_cem_on_kaggle).")
    out = dict(cfg)
    out.setdefault("init_params_data", None)
    return out
