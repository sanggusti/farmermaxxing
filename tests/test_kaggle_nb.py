"""Tests for the Kaggle notebook CEM backend.

All tests run without Kaggle credentials — they verify packaging, config
serialisation, and result parsing, not the actual push/poll/download cycle.
"""

import base64
import io
import json
import os
import tarfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

from search.kaggle_nb import (  # noqa: E402
    _build_tarball_bytes,
    _generate_kernel_script,
    _pycache_filter,
)
from search.kernel_config import KNOWN_KEYS, resolve_cem_config  # noqa: E402

CONFIGS = os.path.join(REPO, "configs")


def composed_cem_config(*overrides):
    """The real composed configs/cem.yaml as a plain dict."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="cem", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


class TestPackaging:
    """Verify the code tarball is built correctly."""

    @pytest.fixture
    def tarball(self):
        config = {"generations": 2, "population": 8, "seeds": 2,
                  "opponents": "starter", "group": "test"}
        tb_bytes = _build_tarball_bytes(config)
        return tarfile.open(fileobj=io.BytesIO(tb_bytes), mode="r:gz")

    def test_tarball_contains_required_files(self, tarball):
        names = tarball.getnames()

        # Agent modules
        assert "agent/params.py" in names
        assert "agent/policy.py" in names
        assert "agent/main.py" in names
        assert "agent/market.py" in names
        assert "agent/rules.py" in names

        # Sim modules
        assert "sim/fastplay.py" in names
        assert "sim/harness.py" in names
        assert "sim/tape.py" in names
        assert "sim/opponents.py" in names
        assert "sim/arena.py" in names

        # Opponent and tape data
        tapes = [n for n in names if n.startswith("sim/tapes/") and n.endswith(".json")]
        opponents = [n for n in names if n.startswith("sim/opponents/") and n.endswith(".json")]
        assert len(tapes) > 0, "no tape files in tarball"
        assert len(opponents) > 0, "no opponent files in tarball"

        # Search helpers
        assert "search/cem.py" in names
        assert "search/league.py" in names
        assert "search/kernel_config.py" in names
        assert "search/blocks.py" in names
        assert "search/__init__.py" in names

        # Config
        assert "cem_config.json" in names

    def test_tarball_excludes_pycache(self, tarball):
        names = tarball.getnames()
        pycache = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        assert pycache == [], f"pycache files found in tarball: {pycache}"

    def test_config_embedded_in_tarball(self, tarball):
        config_file = tarball.extractfile("cem_config.json")
        config = json.load(config_file)
        assert config["generations"] == 2
        assert config["opponents"] == "starter"
        assert config["group"] == "test"


class TestScriptGeneration:
    """Verify the generated kernel script is valid."""

    @pytest.fixture
    def script(self):
        config = {"generations": 1, "population": 4, "seeds": 1,
                  "opponents": "starter", "group": "test"}
        tb = _build_tarball_bytes(config)
        b64 = base64.b64encode(tb).decode()
        return _generate_kernel_script(b64)

    def test_script_compiles(self, script):
        compile(script, "cem_kernel.py", "exec")

    def test_script_contains_payload(self, script):
        assert "_PAYLOAD" in script
        assert "base64.b64decode" in script

    def test_script_contains_cem_loop(self, script):
        assert "initial_distribution" in script
        assert "score_local" in script
        assert "best_params.json" in script

    def test_script_size_under_5mb(self, script):
        # Kaggle has a script size limit; our ~285KB should be well under
        assert len(script.encode()) < 5_000_000


class TestConfigSerialisation:
    """Verify config roundtrips correctly, and that the yaml surface and the
    kernel's KNOWN_KEYS cannot drift apart silently -- the predecessor of this
    transport hand-copied args into a dict and dropped an untaught flag
    (--ramp) WITHOUT ERROR."""

    @staticmethod
    def _apply_driver_transforms(config):
        """The key transforms run_cem_on_kaggle applies before shipping."""
        config = dict(config)
        config.pop("backend")
        config["group"] = config["group"] or "cem-test"
        if config.pop("init_params"):
            config["init_params_data"] = {}
        return config

    def test_basic_roundtrip(self):
        # The REAL composed experiment survives the JSON roundtrip inside the
        # tarball -- transport fidelity of what actually ships, not of a stub.
        config = self._apply_driver_transforms(
            composed_cem_config("+experiment=smoke"))
        tb = _build_tarball_bytes(config)
        with tarfile.open(fileobj=io.BytesIO(tb), mode="r:gz") as tar:
            loaded = json.load(tar.extractfile("cem_config.json"))
        assert loaded == config

    def test_kernel_rejects_unknown_config_keys(self):
        config = self._apply_driver_transforms(composed_cem_config())
        config["ramp_shape"] = 2
        with pytest.raises(SystemExit, match="ramp_shape"):
            resolve_cem_config(config)

    def test_kernel_rejects_missing_config_keys(self):
        config = self._apply_driver_transforms(composed_cem_config())
        del config["ramp"]
        with pytest.raises(SystemExit, match="ramp"):
            resolve_cem_config(config)

    def test_kernel_consumes_exactly_the_composed_keys(self):
        # configs/cem.yaml -> driver transforms -> kernel validation, with no
        # slack on either side: a new yaml key without a KNOWN_KEYS entry (and
        # kernel support) fails here, as does a KNOWN_KEYS entry no yaml
        # produces.
        config = self._apply_driver_transforms(composed_cem_config())
        resolved = resolve_cem_config(config)   # must not raise
        assert set(resolved) == KNOWN_KEYS

    def test_init_params_data_embedded(self):
        from params import Params
        p = Params()
        config = {
            "generations": 2, "population": 8, "seeds": 2,
            "opponents": "starter", "group": "test",
            "init_params_data": p.__dict__,
            "init_spread": 0.10,
        }
        tb = _build_tarball_bytes(config)
        with tarfile.open(fileobj=io.BytesIO(tb), mode="r:gz") as tar:
            loaded = json.load(tar.extractfile("cem_config.json"))
        assert "init_params_data" in loaded
        assert isinstance(loaded["init_params_data"], dict)
        assert "hands_early" in loaded["init_params_data"]


class TestResultParsing:
    """Verify the result download / parse logic."""

    def test_valid_results(self, tmp_path):
        from params import Params
        p = Params()
        p.to_json(str(tmp_path / "best_params.json"))

        results = {
            "best_holdout": 95000.0, "best_train": 98000.0,
            "clean_bank": 93000.0, "clean_min_bank": 85000.0,
            "generations_completed": 10, "group": "test",
            "wall_seconds": 2400.0,
        }
        with open(tmp_path / "results.json", "w") as f:
            json.dump(results, f)

        with open(tmp_path / "best_params.json") as f:
            assert isinstance(json.load(f), dict)
        with open(tmp_path / "results.json") as f:
            loaded = json.load(f)
        assert loaded["best_holdout"] == 95000.0

    def test_generations_jsonl(self, tmp_path):
        lines = [
            {"gen": 0, "train_best_bank": 90000, "holdout_best_bank": 88000},
            {"gen": 1, "train_best_bank": 92000, "holdout_best_bank": 90000},
        ]
        with open(tmp_path / "generations.jsonl", "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        with open(tmp_path / "generations.jsonl") as f:
            loaded = [json.loads(line) for line in f]
        assert len(loaded) == 2
        assert loaded[1]["gen"] == 1


class TestKernelScriptImports:
    """Verify the kernel script's imports match available modules."""

    def test_imported_helpers_exist(self):
        from search.cem import (  # noqa: F401
            initial_distribution, sample, refit, ramp_schedule, score_local,
            selection_score, worst_tolerance, HOLDOUT_OFFSET, CLEAN_OFFSET,
        )
        from search.kernel_config import resolve_cem_config  # noqa: F401
        from search.league import (  # noqa: F401
            build_cells, normalised_fitness, worst_opponent,
        )
        from sim.arena import CENSUS_KEYS  # noqa: F401
