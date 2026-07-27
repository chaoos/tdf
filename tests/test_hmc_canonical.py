"""Tests for TDF-based canonical HMC."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from tdf import lattice, dirac, hmc
from tdf.hmc_canonical import action_canonical, run_hmc_canonical


@pytest.mark.parametrize("L,Lt", [(4, 4), (6, 6)])
def test_action_canonical_real_and_grad_finite(L, Lt):
    key = random.PRNGKey(7)
    theta = random.normal(key, (2, Lt, L))
    kappa = dirac.kappa_from_mass(0.0)
    n = L // 2

    S = action_canonical(theta, beta=3.0, n=n, mu=0.0, kappa=kappa)
    assert jnp.isfinite(S)
    assert jnp.isreal(S)

    g = jax.grad(lambda t: action_canonical(t, beta=3.0, n=n, mu=0.0, kappa=kappa))(theta)
    assert g.shape == theta.shape
    assert jnp.all(jnp.isfinite(g))


def test_canonical_action_charge_conjugation_symmetric():
    """At mu=0 sectors n and -n have identical actions (charge conjugation)."""
    L, Lt = 4, 4
    key = random.PRNGKey(3)
    theta = random.normal(key, (2, Lt, L))
    kappa = dirac.kappa_from_mass(0.0)
    beta = 3.0

    for n in range(1, L + 1):
        S_n = action_canonical(theta, beta, n, 0.0, kappa)
        S_minus_n = action_canonical(theta, beta, -n, 0.0, kappa)
        assert jnp.allclose(S_n, S_minus_n, atol=1e-10)


def test_hmc_canonical_step_runs():
    L, Lt = 4, 4
    key = random.PRNGKey(1)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    history, configs = run_hmc_canonical(
        key, theta, beta=3.0, n=L // 2, mu=0.0, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1, dt=0.05, n_steps=3,
    )
    assert configs.shape == (2, 2, Lt, L)
    assert "plaquette" in history
    assert "accept" in history
    assert jnp.all(jnp.isfinite(history["plaquette"]))
