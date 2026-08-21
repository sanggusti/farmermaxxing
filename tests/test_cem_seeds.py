"""Training seeds must rotate across generations but stay fixed within one,
and the ramp must reshape the spend without changing it.

Seed-set overfitting (issue #68): holding the same seed block for 30 generations
presses 11,520 evaluations against one fixed empirical objective. Rotating per
generation removes the artefact while preserving common-random-number comparisons
within each generation.

The geometric ramp (issue #72) varies the block SIZE per generation. Its
contract is budget-neutrality: `sum(ramp_schedule(G, s, r)) == G * s` exactly,
whatever r, so a ramped run and the constant run it is judged against cost the
same episodes -- otherwise "the ramp helped" would be confounded with "the
ramp spent more".
"""

import pytest

from search.cem import HOLDOUT_OFFSET, TRAIN_POOL, ramp_schedule


def _train_seeds(gen, seeds_per_gen, pool, ramp=1.0, generations=50):
    """Mirror the rotation in cem.main(): cumulative block starts over the
    ramp schedule. At ramp=1 this is the legacy (gen * seeds + i) % pool."""
    schedule = ramp_schedule(generations, seeds_per_gen, ramp)
    start = sum(schedule[:gen])
    return [(start + i) % pool for i in range(schedule[gen])]


@pytest.mark.parametrize("ramp", [1.0, 8.0])
def test_train_seeds_rotate_across_generations(ramp):
    """Adjacent generations must get different seed blocks."""
    seen = set()
    for gen in range(10):
        block = tuple(_train_seeds(gen, 4, TRAIN_POOL, ramp))
        assert block not in seen, f"gen {gen} reused block {block}"
        seen.add(block)


@pytest.mark.parametrize("ramp", [1.0, 8.0])
def test_train_seeds_stay_within_pool(ramp):
    """Every rotated seed must be in [0, pool)."""
    for gen in range(50):
        for s in _train_seeds(gen, 6, TRAIN_POOL, ramp):
            assert 0 <= s < TRAIN_POOL


def test_train_seeds_do_not_overlap_holdout():
    """The train pool must sit entirely below the holdout offset."""
    assert TRAIN_POOL < HOLDOUT_OFFSET


@pytest.mark.parametrize("ramp", [1.0, 8.0])
def test_no_duplicates_within_a_generation(ramp):
    """A single generation's seed block must not contain repeats."""
    for gen in range(50):
        block = _train_seeds(gen, 6, TRAIN_POOL, ramp)
        assert len(block) == len(set(block)), f"gen {gen} has duplicate seeds"


def test_wrap_around_produces_valid_seeds():
    """When the cumulative start exceeds the pool, modular wrap stays in
    bounds."""
    pool = 40  # small pool to force wrapping
    for gen in range(25):
        block = _train_seeds(gen, 4, pool, ramp=8.0, generations=25)
        assert all(0 <= s < pool for s in block)


def test_constant_ramp_reproduces_legacy_formula():
    """--ramp 1 must be bit-for-bit the pre-#72 rotation, so every recorded
    run's seed stream stays reproducible."""
    for gen in range(50):
        legacy = [(gen * 4 + i) % TRAIN_POOL for i in range(4)]
        assert _train_seeds(gen, 4, TRAIN_POOL, ramp=1.0) == legacy


@pytest.mark.parametrize("generations,seeds,ramp",
                         [(40, 4, 8.0), (40, 4, 12.0), (10, 6, 2.0),
                          (8, 3, 8.0), (1, 4, 8.0), (40, 1, 8.0)])
def test_ramp_schedule_is_budget_neutral_exactly(generations, seeds, ramp):
    schedule = ramp_schedule(generations, seeds, ramp)
    assert sum(schedule) == generations * seeds


@pytest.mark.parametrize("generations,seeds,ramp",
                         [(40, 4, 8.0), (10, 6, 2.0), (8, 3, 8.0)])
def test_ramp_schedule_is_monotone_and_positive(generations, seeds, ramp):
    schedule = ramp_schedule(generations, seeds, ramp)
    assert all(n >= 1 for n in schedule)
    assert all(b >= a for a, b in zip(schedule, schedule[1:]))


def test_ramp_schedule_identity_at_one():
    assert ramp_schedule(40, 4, 1.0) == [4] * 40


def test_ramp_schedule_actually_ramps():
    """The endpoints must reflect the requested ratio, not collapse back to
    constant: at ramp=8 the last generation gets several times the first's
    episodes -- that is the resolution shift the issue asked for."""
    schedule = ramp_schedule(40, 4, 8.0)
    assert schedule[-1] >= 4 * schedule[0]


def test_ramp_schedule_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ramp_schedule(0, 4, 2.0)
    with pytest.raises(ValueError):
        ramp_schedule(10, 0, 2.0)
    with pytest.raises(ValueError):
        ramp_schedule(10, 4, 0.0)
