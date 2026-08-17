"""What must be true of the thing we actually ship.

`tests/test_flat_layout.py` checks a hand-assembled copy of the agent. This file
checks `submission.tar.gz` itself, plus the three regimes nothing else covered:
seat 1, a strong opponent, and a policy that raises.

The gap this closes: a submission failure costs a slot and roughly a day of
rating convergence to notice, and the project has 5 submissions a day against a
2026-09-23 entry deadline. Every check here is cheap relative to that.
"""

import io
import os
import re
import runpy
import subprocess
import sys
import tarfile

import pytest

from kaggle_environments import make

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(REPO, "submission.tar.gz")
MAIN = os.path.join(REPO, "agent", "main.py")


def makefile_agent_files():
    """The file list the Makefile actually tars, parsed from the Makefile.

    Parsed rather than duplicated. The list previously existed in two places --
    `Makefile:7` and `tests/test_flat_layout.py` -- with nothing comparing them,
    so adding a module and updating only the test would ship a tarball missing
    that module with the whole suite green.
    """
    with open(os.path.join(REPO, "Makefile")) as fh:
        m = re.search(r"^AGENT_FILES\s*:?=\s*(.+)$", fh.read(), re.M)
    assert m, "Makefile no longer defines AGENT_FILES"
    return m.group(1).split()


def test_makefile_lists_every_agent_module():
    """Every .py in agent/ is in the shipped file list, and vice versa.

    This is the check that makes the duplication impossible to get wrong: a new
    module in agent/ fails here until it is added to the Makefile.
    """
    on_disk = sorted(f for f in os.listdir(os.path.join(REPO, "agent"))
                     if f.endswith(".py"))
    assert sorted(makefile_agent_files()) == on_disk, (
        "Makefile AGENT_FILES and agent/*.py disagree; the tarball would ship "
        "an incomplete or stale set of modules"
    )


def test_flat_layout_list_matches_the_makefile():
    from tests import test_flat_layout
    assert sorted(test_flat_layout.AGENT_FILES) == sorted(makefile_agent_files())


# --------------------------------------------------------------- the tarball


@pytest.fixture(scope="module")
def unpacked_bundle(tmp_path_factory):
    """`submission.tar.gz` extracted, or skip if it has not been built."""
    if not os.path.exists(BUNDLE):
        pytest.skip("no submission.tar.gz; run `make bundle` first")
    dest = tmp_path_factory.mktemp("bundle")
    with tarfile.open(BUNDLE) as tf:
        names = tf.getnames()
        tf.extractall(dest, filter="data")
    return dest, names


@pytest.mark.bundle
def test_bundle_has_main_at_the_archive_root(unpacked_bundle):
    """Kaggle unpacks the archive flat and looks for main.py at its root.

    The Makefile comment has always said this MUST hold and nothing checked it.
    A `tar -czf ... agent/` instead of `tar -czf ... -C agent` produces an
    archive that looks fine in `tar -tzf` and fails its validation episode.
    """
    _, names = unpacked_bundle
    assert "main.py" in names, f"main.py is not at the archive root: {names}"
    assert not any("/" in n.strip("./") for n in names), (
        f"archive contains nested paths, so Kaggle would not find main.py: {names}"
    )


@pytest.mark.bundle
def test_bundle_contains_every_agent_module(unpacked_bundle):
    _, names = unpacked_bundle
    missing = set(makefile_agent_files()) - set(names)
    assert not missing, f"tarball is missing {sorted(missing)}"


@pytest.mark.bundle
def test_bundle_ships_the_promoted_params(unpacked_bundle):
    """The params in the tarball must be the ones in agent/params.json.

    A search writes to `runs/<group>/best_params.json` and `make promote` copies
    it in; nothing verified that the copy happened before the tar. Shipping the
    previous champion's parameters under a new version's submission message is
    silent and would corrupt the ledger's whole point.
    """
    dest, names = unpacked_bundle
    if "params.json" not in names:
        pytest.skip("bundle carries no params.json (agent falls back to defaults)")
    sys.path.insert(0, REPO)
    from sim.ledger import params_sha256
    assert (params_sha256(os.path.join(dest, "params.json"))
            == params_sha256(os.path.join(REPO, "agent", "params.json"))), (
        "submission.tar.gz carries different parameters than agent/params.json; "
        "rebuild with `make bundle`"
    )


@pytest.mark.bundle
def test_bundle_runs_a_full_episode_from_its_own_directory(unpacked_bundle):
    """Run the extracted tarball, from a cwd that is not the repo.

    `tests/conftest.py` puts `agent/` on sys.path for every test, which hides
    exactly the import failure this is looking for. So this runs in a subprocess
    with a clean interpreter and a cwd elsewhere, the way Kaggle does.
    """
    dest, _ = unpacked_bundle
    script = (
        "import os, sys\n"
        "from kaggle_environments import make\n"
        f"main = {os.path.join(str(dest), 'main.py')!r}\n"
        "env = make('kaggriculture', configuration={'episodeSteps': 720, "
        "'seed': 11}, debug=True)\n"
        "env.run([main, 'starter'])\n"
        "final = env.steps[-1]\n"
        "assert final[0].status == 'DONE', final[0].status\n"
        "print('BANK', final[0].reward)\n"
    )
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    env.pop("FM_STRICT", None)
    out = subprocess.run([sys.executable, "-c", script], cwd=os.path.expanduser("~"),
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "BANK" in out.stdout


# ------------------------------------------------------------- both seats


def test_the_two_seats_play_the_same_game():
    """Seat 1 was never exercised, and a swapped farm index would be invisible.

    `Policy._scan` and `Policy._rival_supply` index `obs["farms"][0]` and `[1]`,
    and every existing test puts us in seat 0. Reading the OPPONENT's farm as our
    own would still finish DONE with a plausible-looking bank, so status is not
    the check -- symmetry is. The engine quotes both players against the same
    pre-commit inventory, so against a fixed opponent the two seats differ only
    through the weed RNG, which is drawn in seat order. A few percent apart is
    expected; a factor of two is a swapped index.
    """
    sys.path.insert(0, REPO)
    from sim.fastplay import fast_play
    from params import Params

    me = Params.from_json(os.path.join(REPO, "agent", "params.json"))
    banks = {}
    for seat in (0, 1):
        a, b = (me, "starter") if seat == 0 else ("starter", me)
        r = fast_play(a, b, seed=20_007, steps=720)
        assert r["statuses"][seat] == "DONE", (
            f"seat {seat} ended {r['statuses'][seat]}")
        banks[seat] = r["banks"][seat]

    assert min(banks.values()) > 20_000, f"a seat did essentially nothing: {banks}"
    lo, hi = sorted(banks.values())
    assert hi / lo < 1.5, (
        f"the seats disagree by {hi / lo:.2f}x ({banks}); the only asymmetry in "
        "this engine is the weed draw order, so this looks like a farm-index bug")


# -------------------------------------------------- the strong-opponent regime


@pytest.mark.slow
@pytest.mark.parametrize("tape", ["band-vishnu", "meta-a"])
def test_full_episode_against_a_real_ladder_agent(tape):
    """A full season against a recorded ladder opponent, both seats, no errors.

    This is the regime where an unguarded code path fires. Against a strong
    opponent the market collapses -- milk 160 -> 7, melon 250 -> 31 -- so price
    ratios, sell floors and rival-supply arithmetic all see values they never
    see against `starter`, which farms one carrot tile and never trades.
    """
    sys.path.insert(0, REPO)
    from sim.fastplay import fast_play
    from sim.tape import load as load_tape
    from params import Params

    me = Params.from_json(os.path.join(REPO, "agent", "params.json"))
    for seat in (0, 1):
        opp = load_tape(tape)
        a, b = (me, opp) if seat == 0 else (opp, me)
        r = fast_play(a, b, seed=20_007, steps=720)
        assert r["statuses"][seat] == "DONE", (
            f"seat {seat} vs {tape} ended {r['statuses'][seat]}")
        assert r["banks"][seat] > 3000.0


# ------------------------------------------------------------ crash-safety


def test_a_raising_policy_degrades_to_a_pass_not_an_error():
    """The whole point of the try/except in agent.main.

    Asserted by construction rather than by breaking the real policy: build the
    same fallback the module uses and confirm the engine accepts an all-empty
    turn as a legal no-op, so a swallowed exception costs one turn instead of
    the match.
    """
    def broken(obs):
        try:
            raise KeyError("engine renamed a field")
        except Exception:
            return {"farmer": [], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 96, "seed": 1},
               debug=True)
    env.run([broken, "pass"])
    final = env.steps[-1]
    assert final[0].status == "DONE", (
        "an all-empty action is not a legal no-op, so the fallback in "
        "agent/main.py would itself forfeit the match")


def test_strict_mode_still_raises():
    """FM_STRICT must keep local failures loud. conftest.py sets it for tests."""
    ns = runpy.run_path(MAIN)
    assert os.environ.get("FM_STRICT"), "conftest should set FM_STRICT"
    with pytest.raises(Exception):
        ns["agent"]({})          # nothing like an observation


# --------------------------------------------------------------- stdout


def test_the_agent_prints_nothing():
    """kaggle_environments captures per-step stdout into the replay.

    A stray print costs replay size on every one of 720 turns and, on some
    harness versions, trips validation. Cheap to guard, annoying to find.
    """
    ns = runpy.run_path(MAIN)
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 1})
    env.reset(2)
    obs = env.steps[0][0].observation

    buf = io.StringIO()
    stdout, sys.stdout = sys.stdout, buf
    try:
        ns["agent"](obs)
    finally:
        sys.stdout = stdout
    assert buf.getvalue() == "", f"agent printed {buf.getvalue()!r}"
