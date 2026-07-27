"""Tests for HMC samplers."""

import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac, hmc


def test_action_standard_is_real():
    key = random.PRNGKey(30)
    theta = lattice.make_gauge_field(4, 4, key)
    kappa = dirac.kappa_from_mass(0.0)
    S = hmc.action_standard(theta, beta=3.0, mu=0.0, kappa=kappa)
    assert jnp.isreal(S)
    assert jnp.isfinite(S)


def test_action_standard_matches_tdf_action():
    key = random.PRNGKey(31)
    theta = lattice.make_gauge_field(4, 4, key)
    kappa = dirac.kappa_from_mass(0.0)
    S_full = hmc.action_standard(theta, beta=3.0, mu=0.0, kappa=kappa)
    S_tdf = hmc.action_standard_tdf(theta, beta=3.0, mu=0.0, kappa=kappa)
    assert jnp.allclose(S_full, S_tdf, atol=1e-10)


def test_force_has_correct_shape():
    key = random.PRNGKey(32)
    theta = lattice.make_gauge_field(4, 4, key)
    kappa = dirac.kappa_from_mass(0.0)

    def action_fn(t):
        return hmc.action_standard(t, beta=3.0, mu=0.0, kappa=kappa)

    force_fn = hmc.make_force(action_fn)
    F = force_fn(theta)
    assert F.shape == theta.shape
    assert jnp.isreal(F).all()
    assert jnp.isfinite(F).all()


def test_hmc_step_runs():
    key = random.PRNGKey(33)
    key_field, key_hmc = random.split(key)
    theta = lattice.make_gauge_field(4, 4, key_field)
    kappa = dirac.kappa_from_mass(0.0)

    def action_fn(t):
        return hmc.action_standard(t, beta=3.0, mu=0.0, kappa=kappa)

    force_fn = hmc.make_force(action_fn)
    theta_new, accepted = hmc.hmc_step(key_hmc, theta, action_fn, force_fn, dt=0.1, n_steps=5)
    assert theta_new.shape == theta.shape
    assert accepted.dtype == jnp.bool_


def test_hmc_step_detailed_balance_quenched():
    """Quenched HMC should have near-perfect acceptance for a small step."""
    key = random.PRNGKey(34)
    key_field, key_hmc = random.split(key)
    theta = lattice.make_gauge_field(4, 4, key_field)
    kappa = dirac.kappa_from_mass(0.0)

    def action_fn(t):
        return lattice.gauge_action(t, beta=3.0)

    force_fn = hmc.make_force(action_fn)
    _, accepted = hmc.hmc_step(key_hmc, theta, action_fn, force_fn, dt=0.01, n_steps=5)
    assert accepted


def test_run_hmc_runs():
    key = random.PRNGKey(35)
    key_field, key_run = random.split(key)
    theta = lattice.make_gauge_field(4, 4, key_field)
    kappa = dirac.kappa_from_mass(0.0)

    def action_fn(t):
        return hmc.action_standard(t, beta=3.0, mu=0.0, kappa=kappa)

    force_fn = hmc.make_force(action_fn)
    history, configs = hmc.run_hmc(
        key_run, theta, action_fn, force_fn,
        n_therm=5, n_measure=3, n_skip=2, dt=0.1, n_steps=5
    )
    assert configs.shape == (3, 2, 4, 4)
    assert len(history["plaquette"]) == 3
    assert len(history["accept"]) == 3
