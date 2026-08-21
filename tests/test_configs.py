"""The configs/ tree composes, and typos fail loudly.

Deliberately NO assertions on default values (AGENTS.md: if a change would
make the agent better, no test should have to be re-recorded). What is pinned
is the mechanism that replaced argparse's unknown-flag error: OmegaConf struct
mode rejects an override key the config does not define, so a misspelled
`generatoins=2` errors instead of silently running the default -- the exact
failure shape rule 7 exists for.
"""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(REPO, "configs")

PRIMARIES = ("cem", "cmaes", "subspace", "gate", "arena")


@pytest.mark.parametrize("name", PRIMARIES)
def test_primary_composes(name):
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name=name)
    # Composes to a plain container: the transport into wandb config and the
    # Kaggle kernel is to_container + JSON, so nothing exotic may live here.
    OmegaConf.to_container(cfg, resolve=True)


def test_misspelled_override_is_rejected():
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="cem", overrides=["generatoins=2"])


@pytest.mark.parametrize("exp", ("smoke", "bimodal-tpu", "crossover-tpu"))
def test_experiment_composes_on_cem(exp):
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="cem", overrides=[f"+experiment={exp}"])
    # The experiment file re-roots over the primary (# @package _global_),
    # so its keys must all exist in the composed config -- and a CLI override
    # must still win over the experiment file.
    assert cfg.backend == "kaggle"
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="cem",
                      overrides=[f"+experiment={exp}", "backend=local"])
    assert cfg.backend == "local"


def test_diagnostic_experiments_size_the_elite_pool():
    """Both #70 experiments exist to read the bimodality diagnostic, and the
    diagnostic refuses to fire below MIN_POOL elites (measured spurious rate
    0.8% at n=6). An experiment file edited below that pool would complete
    cleanly and report 'unimodal' as pure absence of power -- rule 7's shape
    exactly."""
    from search.blocks import MIN_POOL
    for exp in ("bimodal-tpu", "crossover-tpu"):
        with initialize_config_dir(config_dir=CONFIGS, version_base=None):
            cfg = compose(config_name="cem", overrides=[f"+experiment={exp}"])
        assert cfg.diagnostics is True
        assert int(cfg.population * cfg.elite_frac) >= MIN_POOL, (
            f"{exp}: {int(cfg.population * cfg.elite_frac)} elites/gen is "
            f"below MIN_POOL={MIN_POOL}; the diagnostic cannot fire")


@pytest.mark.parametrize("name", ("cem", "cmaes", "subspace"))
def test_search_drivers_share_the_common_surface(name):
    """The shared keys exist on every driver, whatever their values.

    These are the keys the drivers' code reads unconditionally; a primary
    that drops one composes fine and then crashes mid-run.
    """
    shared = {"train_pool", "holdout_seeds", "clean_seeds", "steps",
              "opponents", "reference", "fitness", "holdout_opponents",
              "group", "wandb", "init_params", "init_spread", "backend",
              "seeds", "rng_seed"}
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name=name)
    assert shared <= set(cfg.keys())
