"""The cma contracts search/cmaes.py relies on, pinned.

Every one of these is a behaviour the driver assumes rather than enforces, and
this repo's bugs do not raise -- they produce complete runs with plausible
numbers (rule 7). If a cma upgrade changes any of these, the search would
silently optimise something else; these tests make it loud instead.
"""

import pytest

from params import SEARCH_SPACE, Params, flatten
from search.cmaes import (HIS, INT_IDX, LOS, NAMES, default_popsize, make_es,
                          vec_to_x, x_to_vec)


def test_name_order_and_int_indices_match_search_space():
    assert NAMES == list(SEARCH_SPACE)
    for i, name in enumerate(NAMES):
        lo, hi, kind = SEARCH_SPACE[name]
        assert LOS[i] == lo and HIS[i] == hi
        assert (i in INT_IDX) == (kind == "i")


def test_round_trip_is_exact_for_all_dims():
    vec = flatten(Params())
    assert x_to_vec(vec_to_x(vec)) == {n: float(v) for n, v in vec.items()}


def test_default_popsize_is_cmaes_own_default():
    """4 + floor(3 ln 60) = 16. The point of the whole experiment: the same
    ~250k-episode budget at lambda 16 buys ~600 generations of covariance
    adaptation instead of CEM's 40 at population 384+."""
    assert default_popsize(60) == 16


def test_popsize_is_not_silently_overridden():
    """With integer_variables non-empty and popsize unset, cma overrides
    popsize to 6 + 3(ln n + ln n_int) -- measured 27 at (60, 24), a silent
    69% budget increase per generation. make_es must pin it explicitly."""
    es = make_es(vec_to_x(flatten(Params())), 0.10, 16, seed=1)
    assert es.popsize == 16


def test_ask_returns_in_bounds_integral_points():
    """The driver trusts ask(): points inside the box (BoundTransform, not
    the silent clip) with integer coordinates already rounded (cma >= 4.3
    integer handling). unflatten's clip/round must be a no-op on them."""
    es = make_es(vec_to_x(flatten(Params())), 0.10, 16, seed=1)
    for x in es.ask():
        for i, (v, lo, hi) in enumerate(zip(x, LOS, HIS)):
            assert lo <= v <= hi, f"{NAMES[i]}={v} outside [{lo}, {hi}]"
        for i in INT_IDX:
            assert float(x[i]).is_integer(), f"{NAMES[i]}={x[i]} not integral"


def test_x0_on_bounds_is_accepted():
    """22 of the champion's parameters sit exactly on a bound; a warm start
    must not be rejected or shifted for it."""
    vec = flatten(Params())
    vec["prio_feed"] = float(SEARCH_SPACE["prio_feed"][0])       # lower bound
    vec["sell_order_floor"] = float(SEARCH_SPACE["sell_order_floor"][1])  # upper
    es = make_es(vec_to_x(vec), 0.10, 16, seed=1)
    assert es.ask()  # constructs and samples without raising


def test_tell_accepts_negated_fitness_and_adapts():
    """cma minimises, the repo maximises: the driver negates. After a tell,
    the strategy state must have consumed the ranking (countiter advances)."""
    es = make_es(vec_to_x(flatten(Params())), 0.10, 16, seed=1)
    xs = es.ask()
    fitness = [float(i) for i in range(len(xs))]   # already-negated z-scores
    es.tell(xs, fitness)
    assert es.countiter == 1
    assert float(es.sigma) > 0
