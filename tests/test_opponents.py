"""The opponent pool: resolution, freezing, and refusal to overwrite."""

import os

import pytest

from params import Params
from sim import opponents


def test_builtins_resolve_to_names():
    resolved, labels = opponents.resolve_pool("starter,pass")
    assert resolved == ["starter", "pass"]
    assert labels == ["starter", "pass"]


def test_unknown_opponent_is_rejected():
    with pytest.raises(ValueError, match="unknown opponent"):
        opponents.resolve_pool("no-such-agent")


def test_all_includes_every_builtin():
    resolved, labels = opponents.resolve_pool("all")
    for name in opponents.BUILTIN:
        assert name in labels


def test_freeze_roundtrip_and_no_clobber(tmp_path, monkeypatch):
    monkeypatch.setattr(opponents, "POOL_DIR", str(tmp_path))
    p = Params(target_geese=3)

    opponents.freeze(p, "snap-a", notes="test")
    assert opponents.frozen_names() == ["snap-a"]
    assert opponents.load("snap-a").target_geese == 3

    # A snapshot is a permanent record of what we promoted; silently replacing
    # one would invalidate every comparison made against it.
    with pytest.raises(FileExistsError):
        opponents.freeze(p, "snap-a")


def test_frozen_snapshot_resolves_to_params(tmp_path, monkeypatch):
    monkeypatch.setattr(opponents, "POOL_DIR", str(tmp_path))
    opponents.freeze(Params(target_geese=7), "snap-b")

    resolved, labels = opponents.resolve_pool("starter,snap-b")
    assert labels == ["starter", "snap-b"]
    assert resolved[0] == "starter"
    assert isinstance(resolved[1], Params)
    assert resolved[1].target_geese == 7


def test_notes_do_not_break_loading(tmp_path, monkeypatch):
    """`_notes` is metadata; Params.from_dict must ignore unknown keys."""
    monkeypatch.setattr(opponents, "POOL_DIR", str(tmp_path))
    opponents.freeze(Params(), "snap-c", notes="why we promoted this")
    opponents.load("snap-c")


def test_cli_runs_in_a_clean_interpreter(tmp_path):
    """`make freeze` must work outside a pytest session.

    The other tests here import sim.opponents after conftest.py has already put
    agent/ on sys.path, so they cannot catch a missing path setup in the module
    itself. This runs the CLI the way a user does, in a fresh interpreter.
    """
    import subprocess
    import sys as _sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = tmp_path / "src.json"
    Params().to_json(str(params))

    proc = subprocess.run(
        [_sys.executable, "-m", "sim.opponents", "--name", "cli-smoke",
         "--params", str(params)],
        cwd=repo, capture_output=True, text=True,
        env={**os.environ, "FARMERMAXXING_POOL_DIR": str(tmp_path)},
    )
    # Clean up whatever the CLI wrote into the real pool.
    written = os.path.join(repo, "sim", "opponents", "cli-smoke.json")
    if os.path.exists(written):
        os.remove(written)

    assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"
