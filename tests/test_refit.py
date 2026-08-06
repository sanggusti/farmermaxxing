"""CEM's refit must keep every dimension actually searchable.

Both cases below were live defects. Neither showed up as an error -- the search
ran, logged rising numbers, and returned a champion. They showed up as a search
that could not move on roughly half its dimensions while reporting progress.
"""

import math
import statistics
import sys

from params import SEARCH_SPACE, Params, flatten
from search.cem import refit, sample


def _p_round_changes(std):
    """Probability a draw centred on an integer rounds to a different one."""
    return math.erfc(0.5 / (std * math.sqrt(2)))


def test_integer_dimensions_stay_searchable_when_elites_agree():
    """A floor as a fraction of range does not keep integers alive.

    At floor=0.02, `fertilize_enabled` (range 0-1) had std 0.02, so the chance
    of ever sampling the other value once the elites agreed was 0.0000%. Three
    parameters were frozen outright. The variance floor exists precisely so
    that "a parameter that collapses early can still recover"; it has to be
    expressed in units where that means something.
    """
    elites = [dict(flatten(Params())) for _ in range(96)]
    _, std = refit(elites)

    for name, (lo, hi, kind) in SEARCH_SPACE.items():
        if kind != "i":
            continue
        p = _p_round_changes(std[name])
        assert p > 0.05, (
            f"{name} (range {lo}-{hi}) has std {std[name]:.3f}, giving only a "
            f"{p:.4%} chance of changing value -- effectively unsearchable"
        )


def test_refit_does_not_let_the_mean_leave_the_box():
    """Selection toward a bound must not walk the mean out of the range.

    `sample()` draws unbounded and only `unflatten()` clips, so fitting to raw
    elite values let a bound-optimal parameter's mean drift outside the box
    forever. Measured before the fix: mean -124 against a bound of 0 by
    generation 30, with 100% of draws clipping to the same value -- a
    population of duplicates whose only difference is the noise draw.
    """
    import random

    name = "prio_plant"
    lo, hi, _ = SEARCH_SPACE[name]
    rng = random.Random(0)

    mean = dict(flatten(Params()))
    mean[name] = float(lo)
    std = {k: (h - l) * 0.25 for k, (l, h, _) in SEARCH_SPACE.items()}

    for _ in range(30):
        pop = [sample(mean, std, rng) for _ in range(128)]
        # Maximal pressure toward the lower bound: the worst case for drift.
        elites = sorted(pop, key=lambda v: v[name])[:32]
        m, s = refit(elites)
        mean[name] = 0.7 * m[name] + 0.3 * mean[name]
        std[name] = s[name]

    assert lo <= mean[name] <= hi, (
        f"fitted mean {mean[name]:.2f} escaped the box [{lo}, {hi}]"
    )


def test_refit_still_tracks_the_elites_when_they_are_inside_the_box():
    """The clipping must not distort an ordinary, interior fit."""
    name = "distance_penalty"
    lo, hi, _ = SEARCH_SPACE[name]
    target = (lo + hi) / 2

    elites = []
    for i in range(64):
        v = dict(flatten(Params()))
        v[name] = target + (i - 32) * 0.01
        elites.append(v)

    mean, _ = refit(elites)
    assert mean[name] == statistics.mean([e[name] for e in elites])


def test_worst_opponent_tolerance_does_not_collapse_at_zero():
    """The guarded quantity is a margin, and margins pass through zero.

    A warm start from the reigning champion ties itself at exactly 0 against
    that champion. With a purely proportional tolerance the bar becomes
    `0 - 0.05 * 0 == 0`, so any candidate that loses to it by a single coin is
    rejected. A v7 run rejected 16 of its first 17 generations and returned its
    own starting point.
    """
    from search.cem import WORST_TOLERANCE, WORST_TOLERANCE_FLOOR

    def bar(best_worst):
        return best_worst - max(WORST_TOLERANCE * abs(best_worst),
                                WORST_TOLERANCE_FLOOR)

    # At a tie, ordinary sampling noise must still get through.
    assert bar(0.0) < -1000, "a near-zero incumbent must not freeze the search"

    # A genuine collapse is still caught, at any incumbent level.
    for best in (0.0, -20_000.0, 50_000.0):
        assert bar(best) > best - 40_000

    # The bar must move monotonically with the incumbent, including through
    # zero -- the multiplicative form inverted for negative values.
    assert bar(-20_000) < bar(0.0) < bar(50_000)
