"""Tests for the pseudofermion estimators and HMC drivers."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from tdf import dirac, lattice
from tdf.hmc_pseudofermion import (
    run_hmc_pseudofermion_standard,
    run_hmc_pseudofermion_tdf,
)
from tdf.pseudofermion import (
    cg_solve,
    estimate_pseudofermion_action_distribution,
    pseudofermion_action_standard,
    pseudofermion_action_tdf,
    pseudofermion_force_standard,
    pseudofermion_force_tdf,
    refresh_pseudofermion,
)


def test_cg_solve_reaches_tolerance():
    """CG solver should reach the requested relative residual."""
    key = random.PRNGKey(0)
    N = 8
    A_base = random.normal(key, (N, N))
    A = A_base @ A_base.T + jnp.eye(N)  # positive definite
    b = random.normal(random.PRNGKey(1), (N,))

    A_fn = lambda x: A @ x
    x, it, relres = cg_solve(A_fn, b, tol=1e-9, maxiter=N)

    assert relres < 1e-9
    assert jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b) < 1e-8


def test_cg_solve_complex():
    """CG solver should handle complex positive-definite systems."""
    N = 6
    key = random.PRNGKey(2)
    real = random.normal(key, (N, N))
    imag = random.normal(random.PRNGKey(3), (N, N))
    M = real + 1j * imag
    A = M @ jnp.conj(M.T) + jnp.eye(N)
    b = random.normal(random.PRNGKey(4), (N,)) + 1j * random.normal(random.PRNGKey(5), (N,))

    A_fn = lambda x: A @ x
    x, it, relres = cg_solve(A_fn, b, tol=1e-9, maxiter=N)

    assert relres < 1e-9
    assert jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b) < 1e-8


def test_pseudofermion_force_shape():
    """Forces have the same shape as the gauge field."""
    L, Lt = 4, 4
    key = random.PRNGKey(0)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    phi = refresh_pseudofermion(random.PRNGKey(1), (2 * L * Lt,))
    force_std = pseudofermion_force_standard(theta, phi, kappa, tol=1e-6)
    assert force_std(theta).shape == theta.shape

    psi = refresh_pseudofermion(random.PRNGKey(2), (2 * L,))
    force_tdf = pseudofermion_force_tdf(theta, psi, kappa, tol=1e-6)
    assert force_tdf(theta).shape == theta.shape


@pytest.mark.parametrize("algorithm", ["standard", "tdf"])
def test_pseudofermion_action_distribution(algorithm):
    """Pseudofermion action is finite and its width measures the estimator noise."""
    L, Lt = 4, 4
    key = random.PRNGKey(7)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    result = estimate_pseudofermion_action_distribution(
        theta, kappa, n_samples=50, tol=1e-9, algorithm=algorithm,
        key=random.PRNGKey(8)
    )
    assert jnp.isfinite(result["mean_Spf"])
    assert jnp.isfinite(result["std_Spf"])
    assert result["std_Spf"] >= 0.0


@pytest.mark.parametrize("L,Lt", [(4, 4)])
def test_hmc_pseudofermion_standard_runs(L, Lt):
    """A short standard pseudofermion HMC run executes."""
    key = random.PRNGKey(1)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    history, configs = run_hmc_pseudofermion_standard(
        key, theta, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1, dt=0.05, n_steps=3, tol=1e-6,
    )
    assert configs.shape == (2, 2, Lt, L)
    assert jnp.all(jnp.isfinite(history["plaquette"]))


@pytest.mark.parametrize("L,Lt", [(4, 4)])
def test_hmc_pseudofermion_tdf_runs(L, Lt):
    """A short TDF pseudofermion HMC run executes."""
    key = random.PRNGKey(2)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    history, configs = run_hmc_pseudofermion_tdf(
        key, theta, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1, dt=0.05, n_steps=3, tol=1e-6,
    )
    assert configs.shape == (2, 2, Lt, L)
    assert jnp.all(jnp.isfinite(history["plaquette"]))
