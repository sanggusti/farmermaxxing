"""The bimodality diagnostic and block crossover must be trustworthy instruments.

Every optimiser bug this project has found produced complete runs with
plausible numbers (rule 7). The tests here pin the statistics of the new
instruments on synthetic data with known answers -- a diagnostic that can only
say "unimodal" is indistinguishable from one measuring nothing, and a
crossover flag that perturbs the rng at 0.0 silently changes every default
search that follows.
"""

import math
import random

from params import SEARCH_SPACE, Params, flatten
from search.blocks import (BLOCKS, MIN_POOL, bimodality_report,
                           block_bimodality, crossover_children)
from search.cem import sample, initial_distribution

D4_NAMES = BLOCKS["labour"]   # a real 4-dim block, so bounds are the real ones


def _unit_gauss_elites(rng, names, centres, sigma_frac, n):
    """Elites whose `names` coordinates are Gaussian around `centres`
    (fractions of each range), everything else at the Params() default."""
    base = flatten(Params())
    elites = []
    for _ in range(n):
        e = dict(base)
        for name in names:
            lo, hi, _ = SEARCH_SPACE[name]
            c = centres[rng.randrange(len(centres))]
            e[name] = lo + (hi - lo) * rng.gauss(c, sigma_frac)
        elites.append(e)
    return elites


def test_blocks_exactly_partition_search_space():
    """A SEARCH_SPACE parameter without a block is invisible to the
    diagnostic and immune to crossover, silently. Coverage and disjointness
    make that a loud test failure instead."""
    seen = [name for names in BLOCKS.values() for name in names]
    assert len(seen) == len(set(seen)), "a parameter appears in two blocks"
    assert set(seen) == set(SEARCH_SPACE), (
        "BLOCKS and SEARCH_SPACE disagree: "
        f"unblocked={sorted(set(SEARCH_SPACE) - set(seen))} "
        f"phantom={sorted(set(seen) - set(SEARCH_SPACE))}")


def test_unimodal_sample_is_not_flagged():
    """The instrument must be able to say no. k-means at k=2 splits ANY
    cloud, and on a unimodal Gaussian the artefactual separation converges
    to ~2.65 -- the thresholds exist to clear exactly that."""
    rng = random.Random(0)
    elites = _unit_gauss_elites(rng, D4_NAMES, centres=[0.5],
                                sigma_frac=0.10, n=128)
    r = block_bimodality(elites, D4_NAMES)
    assert r["bimodal"] is False
    assert r["separation"] < 3.0


def test_bimodal_sample_is_flagged():
    """Two tight clusters at 0.25 and 0.75 of range are the textbook picture
    of the failure the diagnostic hunts: a unimodal fit would put the mean in
    the valley between them."""
    rng = random.Random(1)
    elites = _unit_gauss_elites(rng, D4_NAMES, centres=[0.25],
                                sigma_frac=0.03, n=32)
    elites += _unit_gauss_elites(rng, D4_NAMES, centres=[0.75],
                                 sigma_frac=0.03, n=32)
    r = block_bimodality(elites, D4_NAMES)
    assert r["bimodal"] is True
    assert r["delta_bic"] > 50
    assert r["separation"] > 5


def test_diagnostic_is_deterministic():
    """Zero rng draws is the guarantee that diagnostics=true cannot perturb
    the search it watches; determinism is its observable consequence."""
    rng = random.Random(2)
    elites = _unit_gauss_elites(rng, D4_NAMES, centres=[0.3, 0.7],
                                sigma_frac=0.05, n=48)
    assert block_bimodality(elites, D4_NAMES) == \
        block_bimodality(elites, D4_NAMES)


def test_small_pool_cannot_fire_spuriously():
    """Measured before the MIN_POOL gate existed: at n=6, 0.8% of UNIMODAL
    pools cleared both thresholds (max separation 21.5 -- six points split
    2+4 can look arbitrarily clean), still 0.2% at n=12, clean from n=16.
    The production default pool is 6 (population 24 x elite_frac 0.25), so
    without the gate a default-sized run would average a phantom basin
    every ~12 generation-blocks. The gate must hold on exactly the draws
    that used to fire."""
    assert MIN_POOL > 12
    fired_thresholds = 0
    for seed in range(50):
        rng = random.Random(seed)
        elites = _unit_gauss_elites(rng, D4_NAMES, centres=[0.5],
                                    sigma_frac=0.10, n=6)
        r = block_bimodality(elites, D4_NAMES)
        assert r["bimodal"] is False
        if (r["delta_bic"] > 10.0 and r["separation"] > 3.0
                and r["min_cluster"] >= 2):
            fired_thresholds += 1
    # The raw thresholds DO fire at n=6 -- that is the measurement MIN_POOL
    # encodes. If this stops holding, the gate may no longer be needed.
    assert fired_thresholds > 0


def test_large_unimodal_pool_never_fires():
    """At and above MIN_POOL the artefactual k=2 split of a unimodal cloud
    stays under the thresholds (0/400 measured at n=24; max separation 5.45
    is above 3.0 only when delta_bic fails). The instrument can say no."""
    for seed in range(100):
        rng = random.Random(seed)
        elites = _unit_gauss_elites(rng, D4_NAMES, centres=[0.5],
                                    sigma_frac=0.10, n=MIN_POOL)
        assert block_bimodality(elites, D4_NAMES)["bimodal"] is False


def test_one_dim_block_works():
    """The `distance` block is a single dimension; the projection and BIC
    arithmetic must survive d=1 on both shapes."""
    names = BLOCKS["distance"]
    rng = random.Random(3)
    uni = _unit_gauss_elites(rng, names, centres=[0.5], sigma_frac=0.08, n=64)
    assert block_bimodality(uni, names)["bimodal"] is False
    bi = _unit_gauss_elites(rng, names, centres=[0.2], sigma_frac=0.02, n=32)
    bi += _unit_gauss_elites(rng, names, centres=[0.8], sigma_frac=0.02, n=32)
    assert block_bimodality(bi, names)["bimodal"] is True


def test_identical_elites_do_not_crash():
    """All-identical elites are the far end of convergence, not an error."""
    elites = [dict(flatten(Params())) for _ in range(24)]
    r = block_bimodality(elites, D4_NAMES)
    assert r == {"n": 24, "delta_bic": 0.0, "separation": 0.0,
                 "min_cluster": 0, "bimodal": False}


def test_report_covers_every_block():
    rng = random.Random(4)
    mean, std = initial_distribution(Params())
    elites = [sample(mean, std, rng) for _ in range(24)]
    report = bimodality_report(elites)
    assert set(report) == set(BLOCKS)


def test_crossover_children_zero_is_empty_and_stateless():
    """crossover_frac=0.0 must be byte-identical production CEM. Zero
    children => zero rng draws, or the flag's mere existence changes every
    default search's sampling sequence."""
    rng = random.Random(5)
    state = rng.getstate()
    assert crossover_children([flatten(Params())] * 4, 0, rng) == []
    assert rng.getstate() == state


def test_crossover_off_consumes_no_rng():
    """The driver's population construction at crossover_frac=0 (Gaussian
    draws first, crossover after an early return) must reproduce the plain
    comprehension draw-for-draw."""
    mean, std = initial_distribution(Params())
    rng_a, rng_b = random.Random(0), random.Random(0)
    pop_a = [sample(mean, std, rng_a) for _ in range(8)]
    pop_b = [sample(mean, std, rng_b) for _ in range(8 - 0)]
    pop_b += crossover_children([flatten(Params())] * 4, 0, rng_b)
    assert pop_a == pop_b
    assert rng_a.getstate() == rng_b.getstate()


def test_crossover_children_take_whole_blocks():
    """The operator's entire justification is donation over interpolation:
    every child block equals one parent's block elementwise (post-clip),
    never a blend -- a midpoint IS the valley point the diagonal Gaussian
    already produces."""
    rng = random.Random(6)
    mean, std = initial_distribution(Params())
    parents = [sample(mean, std, rng) for _ in range(6)]

    def clipped(vec, name):
        lo, hi, _ = SEARCH_SPACE[name]
        return min(max(vec[name], lo), hi)

    for child in crossover_children(parents, 16, rng):
        assert set(child) == set(SEARCH_SPACE)
        for bname, names in BLOCKS.items():
            matches = [
                p for p in parents
                if all(math.isclose(child[n], clipped(p, n)) for n in names)]
            assert matches, f"block {bname} matches no parent -- a blend"


def test_crossover_children_are_inside_the_box():
    """Parents are raw (unclipped) sample() draws; children must not inherit
    the out-of-range values refit's docstring measured (-124 against a bound
    of 0)."""
    rng = random.Random(7)
    mean, std = initial_distribution(Params(), spread=2.0)   # wildly raw
    parents = [sample(mean, std, rng) for _ in range(6)]
    for child in crossover_children(parents, 8, rng):
        for name, (lo, hi, _) in SEARCH_SPACE.items():
            assert lo <= child[name] <= hi
