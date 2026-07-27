"""Tests for canonical determinants."""

import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac, reduced, canonical


def test_elementary_symmetric():
    """Elementary symmetric polynomials for a simple case."""
    lam = jnp.array([1.0, 2.0, 3.0], dtype=jnp.complex128)
    coeffs = canonical.elementary_symmetric(lam)
    # (z+1)(z+2)(z+3) = z^3 + 6 z^2 + 11 z + 6
    expected = jnp.array([1.0, 6.0, 11.0, 6.0], dtype=jnp.complex128)
    assert jnp.allclose(coeffs, expected)


def test_canonical_sum_rule():
    """Sum of canonical determinants equals det K[U, mu=0]."""
    key = random.PRNGKey(20)
    L, Lt = 4, 6
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    detK = reduced.reduced_determinant(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    dets = canonical.canonical_determinants(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)

    assert jnp.allclose(jnp.sum(dets), detK, atol=1e-10)


def test_canonical_reflection_symmetry():
    """det_k^* = det_{-k}."""
    key = random.PRNGKey(21)
    L, Lt = 5, 6
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    dets = canonical.canonical_determinants(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    for k in range(-L, L + 1):
        assert jnp.allclose(dets[k + L], jnp.conj(dets[-k + L]), atol=1e-10)


def test_canonical_zero_sector_real():
    """det_0 is real."""
    key = random.PRNGKey(22)
    L, Lt = 4, 6
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    dets = canonical.canonical_determinants(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    det0 = dets[L]
    assert abs(det0.imag) < 1e-10


def test_canonical_single_call():
    """canonical_determinant(k) matches canonical_determinants()[k+L]."""
    key = random.PRNGKey(23)
    L, Lt = 4, 6
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    dets = canonical.canonical_determinants(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    for k in [-L, 0, L]:
        det_k = canonical.canonical_determinant(theta, k, mu=0.0, kappa=kappa, boundary_phase=-1.0)
        assert jnp.allclose(det_k, dets[k + L])
