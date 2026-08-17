"""Training seeds must rotate across generations but stay fixed within one.

Seed-set overfitting (issue #68): holding the same seed block for 30 generations
presses 11,520 evaluations against one fixed empirical objective. Rotating per
generation removes the artefact while preserving common-random-number comparisons
within each generation.
"""

import pytest

from search.cem import HOLDOUT_OFFSET, TRAIN_POOL


def _train_seeds(gen, seeds_per_gen, pool):
    """Mirror the rotation formula in cem.main()."""
    return [(gen * seeds_per_gen + i) % pool for i in range(seeds_per_gen)]


def test_train_seeds_rotate_across_generations():
    """Adjacent generations must get different seed blocks."""
    seeds_per_gen = 4
    seen = set()
    for gen in range(10):
        block = tuple(_train_seeds(gen, seeds_per_gen, TRAIN_POOL))
        assert block not in seen, f"gen {gen} reused block {block}"
        seen.add(block)


def test_train_seeds_stay_within_pool():
    """Every rotated seed must be in [0, pool)."""
    seeds_per_gen = 6
    for gen in range(50):
        for s in _train_seeds(gen, seeds_per_gen, TRAIN_POOL):
            assert 0 <= s < TRAIN_POOL


def test_train_seeds_do_not_overlap_holdout():
    """The train pool must sit entirely below the holdout offset."""
    assert TRAIN_POOL < HOLDOUT_OFFSET


def test_no_duplicates_within_a_generation():
    """A single generation's seed block must not contain repeats."""
    seeds_per_gen = 6
    for gen in range(50):
        block = _train_seeds(gen, seeds_per_gen, TRAIN_POOL)
        assert len(block) == len(set(block)), f"gen {gen} has duplicate seeds"


def test_wrap_around_produces_valid_seeds():
    """When gen * seeds exceeds the pool, modular wrap stays in bounds."""
    seeds_per_gen = 4
    pool = 10  # small pool to force wrapping
    for gen in range(25):
        block = _train_seeds(gen, seeds_per_gen, pool)
        assert all(0 <= s < pool for s in block)
