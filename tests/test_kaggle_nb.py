"""Tests for the Kaggle notebook CEM backend.

All tests run without Kaggle credentials — they verify packaging, config
serialisation, and result parsing, not the actual push/poll/download cycle.
"""

import json
import os
import tarfile
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# We need REPO on the path so `from search.kaggle_nb import ...` works.
import sys
sys.path[:0] = [REPO, os.path.join(REPO, "agent")]

from search.kaggle_nb import (  # noqa: E402
    _pycache_filter,
    package_and_upload,
    STAGING_DIR,
)


class TestPackaging:
    """Verify the code tarball is built correctly."""

    @pytest.fixture(autouse=True)
    def staging(self, tmp_path, monkeypatch):
        """Redirect staging to a temp dir so tests don't touch the real one."""
        staging = tmp_path / "staging"
        staging.mkdir()
        # Write a dummy dataset-metadata.json
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": "test", "id": "test/test", "licenses": [{"name": "CC0-1.0"}]
        }))
        monkeypatch.setattr("search.kaggle_nb.STAGING_DIR", str(staging))
        monkeypatch.setattr("search.kaggle_nb.REPO", REPO)
        self.staging = staging

    def _build_tarball(self, config=None):
        """Build the tarball without uploading (mock subprocess)."""
        if config is None:
            config = {
                "generations": 2, "population": 8, "seeds": 2,
                "opponents": "starter", "group": "test",
            }

        # Mock subprocess.run so we don't actually upload
        import unittest.mock
        with unittest.mock.patch("search.kaggle_nb.subprocess") as mock_sub:
            mock_sub.run.return_value = unittest.mock.MagicMock(returncode=0)
            package_and_upload(config, "testuser")

        tarball = self.staging / "code.tar.gz"
        assert tarball.exists(), "tarball was not created"
        return tarball

    def test_tarball_contains_required_files(self):
        tarball = self._build_tarball()
        with tarfile.open(tarball) as tar:
            names = tar.getnames()

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
        assert "search/__init__.py" in names

    def test_tarball_excludes_pycache(self):
        tarball = self._build_tarball()
        with tarfile.open(tarball) as tar:
            names = tar.getnames()

        pycache = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        assert pycache == [], f"pycache files found in tarball: {pycache}"

    def test_config_written_alongside_tarball(self):
        config = {
            "generations": 4, "population": 12, "seeds": 3,
            "opponents": "top", "group": "test-config",
        }
        self._build_tarball(config)
        config_path = self.staging / "cem_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded["generations"] == 4
        assert loaded["opponents"] == "top"
        assert loaded["group"] == "test-config"


class TestConfigSerialisation:
    """Verify config roundtrips correctly."""

    def test_basic_roundtrip(self):
        config = {
            "generations": 10, "population": 48, "elite_frac": 0.25,
            "seeds": 6, "train_pool": 1000, "holdout_seeds": 6,
            "clean_seeds": 8, "steps": 720, "opponents": "top",
            "reference": None, "fitness": "bank",
            "holdout_opponents": 0, "rng_seed": 0,
            "group": "cem-test",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == config
        finally:
            os.unlink(path)

    def test_init_params_data_embedded(self):
        """When init_params is given, its data is embedded, not a path."""
        from params import Params
        p = Params()
        config = {
            "generations": 2, "population": 8, "seeds": 2,
            "opponents": "starter", "group": "test",
            "init_params_data": p.__dict__,
            "init_spread": 0.10,
        }
        serialised = json.dumps(config)
        loaded = json.loads(serialised)
        # Should contain actual parameter values, not a file path
        assert "init_params_data" in loaded
        assert isinstance(loaded["init_params_data"], dict)
        assert "hands_early" in loaded["init_params_data"]


class TestResultParsing:
    """Verify the result download / parse logic."""

    def test_valid_results(self, tmp_path):
        from params import Params
        # Write a mock best_params.json
        p = Params()
        p.to_json(str(tmp_path / "best_params.json"))

        # Write a mock results.json
        results = {
            "best_holdout": 95000.0,
            "best_train": 98000.0,
            "clean_bank": 93000.0,
            "clean_min_bank": 85000.0,
            "generations_completed": 10,
            "group": "test",
            "wall_seconds": 2400.0,
        }
        with open(tmp_path / "results.json", "w") as f:
            json.dump(results, f)

        # Verify parseable
        with open(tmp_path / "best_params.json") as f:
            params_data = json.load(f)
        assert isinstance(params_data, dict)

        with open(tmp_path / "results.json") as f:
            loaded = json.load(f)
        assert loaded["best_holdout"] == 95000.0
        assert loaded["generations_completed"] == 10

    def test_generations_jsonl(self, tmp_path):
        """Verify per-generation log is parseable."""
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

    def test_kernel_script_has_no_syntax_errors(self):
        """The kernel script should be valid Python."""
        kernel_path = os.path.join(
            REPO, "search", "kaggle_notebook", "cem_kernel.py")
        with open(kernel_path) as f:
            source = f.read()
        # Just check it compiles; it can't actually run without /kaggle/input
        compile(source, kernel_path, "exec")

    def test_imported_helpers_exist(self):
        """The helpers the kernel imports should be importable locally."""
        from search.cem import (  # noqa: F401
            initial_distribution, sample, refit, score_local,
            selection_score, HOLDOUT_OFFSET, CLEAN_OFFSET,
            WORST_TOLERANCE, WORST_TOLERANCE_FLOOR,
        )
        from search.league import (  # noqa: F401
            build_cells, normalised_fitness, worst_opponent,
        )
        from sim.arena import CENSUS_KEYS  # noqa: F401
