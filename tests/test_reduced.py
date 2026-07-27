"""Tests for the temporal determinant factorization."""

import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac, reduced


def test_reduced_determinant_antiperiodic():
    """Reduced determinant matches full determinant for anti-periodic BC."""
    key = random.PRNGKey(10)
    L, Lt = 6, 8
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    detK_full = jnp.linalg.det(K)
    detK_red = reduced.reduced_determinant(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)

    assert jnp.allclose(detK_full, detK_red, atol=1e-10)


def test_reduced_determinant_periodic_even():
    """Reduced determinant matches full determinant for periodic BC, even Lt."""
    key = random.PRNGKey(11)
    L, Lt = 6, 8
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=1.0)
    detK_full = jnp.linalg.det(K)
    detK_red = reduced.reduced_determinant(theta, mu=0.0, kappa=kappa, boundary_phase=1.0)

    assert jnp.allclose(detK_full, detK_red, atol=1e-10)


def test_reduced_determinant_periodic_odd():
    """Reduced determinant matches full determinant for periodic BC, odd Lt."""
    key = random.PRNGKey(12)
    L, Lt = 4, 5
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)

    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=1.0)
    detK_full = jnp.linalg.det(K)
    detK_red = reduced.reduced_determinant(theta, mu=0.0, kappa=kappa, boundary_phase=1.0)

    assert jnp.allclose(detK_full, detK_red, atol=1e-10)


def test_reduced_determinant_with_mass():
    """Reduced determinant matches full determinant for non-zero mass."""
    key = random.PRNGKey(13)
    L, Lt = 4, 6
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.5
    kappa = dirac.kappa_from_mass(mass)

    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    detK_full = jnp.linalg.det(K)
    detK_red = reduced.reduced_determinant(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)

    assert jnp.allclose(detK_full, detK_red, atol=1e-10)


def test_reduced_determinant_with_mu():
    """Reduced determinant matches full determinant with chemical potential."""
    key = random.PRNGKey(14)
    L, Lt = 4, 6
    theta = lattice.make_gauge_field(L, Lt, key)
    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)
    mu = 0.3

    K = dirac.wilson_dirac(theta, mu=mu, kappa=kappa, boundary_phase=-1.0)
    detK_full = jnp.linalg.det(K)
    detK_red = reduced.reduced_determinant(theta, mu=mu, kappa=kappa, boundary_phase=-1.0)

    assert jnp.allclose(detK_full, detK_red, atol=1e-10)
