"""The geometry search/subspace.py stands on, pinned.

The driver's claim is "a quadratic model in a random 5-D subspace sees the
cross-curvature". That claim is only as good as: the basis actually being
orthonormal (otherwise the trust region is an ellipsoid of unknown shape),
the design actually being unisolvent (otherwise the fit is underdetermined
and lstsq silently picks one of many quadratics), and the fit actually
recovering a known quadratic when there is no noise. None of these would
raise if broken -- they would produce plausible steps in wrong directions.
"""

import random

import numpy as np
import pytest

from params import Params, flatten
from search.subspace import (design_points, fit_quadratic, from_unit,
                             model_step, quad_features, random_subspace,
                             to_unit)


def test_subspace_is_orthonormal_and_deterministic():
    Q = random_subspace(60, 5, random.Random(3))
    assert Q.shape == (60, 5)
    assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-10)
    Q2 = random_subspace(60, 5, random.Random(3))
    assert np.array_equal(Q, Q2)


def test_design_has_exactly_the_quadratic_coefficient_count():
    pts = design_points(5, 0.1)
    assert len(pts) == 21 == (5 + 1) * (5 + 2) // 2
    assert all(np.linalg.norm(z) <= 0.1 + 1e-12 for z in pts)


def test_design_is_unisolvent():
    """The 21-point design must determine all 21 quadratic coefficients:
    the feature matrix is square and non-singular."""
    A = np.array([quad_features(z) for z in design_points(5, 0.1)])
    assert A.shape == (21, 21)
    assert np.linalg.matrix_rank(A) == 21


def test_fit_recovers_a_known_quadratic_exactly():
    """Noiseless interpolation: plant a quadratic, fit it back."""
    p = 5
    rng = np.random.default_rng(0)
    g_true = rng.normal(size=p)
    M = rng.normal(size=(p, p))
    H_true = M + M.T   # symmetric, generic cross terms
    c_true = 3.0

    def f(z):
        return c_true + g_true @ z + 0.5 * z @ H_true @ z

    zs = design_points(p, 0.1)
    c, g, H = fit_quadratic(zs, [f(z) for z in zs])
    assert c == pytest.approx(c_true, abs=1e-8)
    assert np.allclose(g, g_true, atol=1e-8)
    assert np.allclose(H, H_true, atol=1e-6)


def test_model_step_takes_the_interior_maximum_when_there_is_one():
    """Concave model with its peak inside the region -> step lands on it."""
    H = -np.eye(5)              # peak at z = g
    g = np.full(5, 0.01)
    z = model_step(g, H, radius=0.5)
    assert np.allclose(z, g, atol=1e-10)


def test_model_step_goes_to_the_boundary_otherwise():
    """Convex (or far-peaked) model -> gradient step of exactly `radius`."""
    H = np.eye(5)               # no interior maximum
    g = np.array([1.0, 0, 0, 0, 0])
    z = model_step(g, H, radius=0.1)
    assert np.linalg.norm(z) == pytest.approx(0.1)
    assert z[0] > 0             # along the gradient, not against it


def test_unit_cube_round_trip():
    vec = flatten(Params())
    back = from_unit(to_unit(vec))
    for name, v in vec.items():
        assert back[name] == pytest.approx(float(v), abs=1e-9)
