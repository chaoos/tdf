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
    pseudofermion_action_tdf_block_cyclic_tm,
    pseudofermion_action_tdf_block_diagonal,
    pseudofermion_action_tdf_bulk_product,
    pseudofermion_action_tdf_stochastic_bulk,
    pseudofermion_force_standard,
    pseudofermion_force_tdf,
    pseudofermion_force_tdf_block_cyclic_tm,
    pseudofermion_force_tdf_block_diagonal,
    pseudofermion_force_tdf_bulk_product,
    pseudofermion_force_tdf_stochastic_bulk,
    refresh_pseudofermion,
    refresh_pseudofermion_standard,
    refresh_pseudofermion_tdf,
    refresh_pseudofermion_tdf_block_cyclic_tm,
    refresh_pseudofermion_tdf_block_diagonal,
    refresh_pseudofermion_tdf_bulk_product,
    refresh_pseudofermion_tdf_stochastic_bulk,
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


def test_refresh_pseudofermion_standard_covariance():
    """Standard HMC refresh produces pseudofermions with Cov(phi) = K K^dagger."""
    L, Lt = 2, 2
    key = random.PRNGKey(20)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    target = K @ jnp.conj(K.T)

    n_samples = 2000
    keys = random.split(random.PRNGKey(21), n_samples)
    samples = jax.vmap(lambda k: refresh_pseudofermion_standard(k, theta, kappa))(keys)
    # Cov_{j,k} = E[phi_j phi_k^*] = (samples^T @ conj(samples)) / n.
    sample_cov = (samples.T @ jnp.conj(samples)) / n_samples

    rel_diff = jnp.linalg.norm(sample_cov - target) / jnp.linalg.norm(target)
    assert rel_diff < 0.2


def test_refresh_pseudofermion_tdf_covariance():
    """TDF HMC refresh produces pseudofermions with Cov(psi) = M M^dagger."""
    L, Lt = 2, 2
    key = random.PRNGKey(22)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    from tdf.reduced import transfer_matrices
    _, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T
    target = M @ jnp.conj(M.T)

    n_samples = 2000
    keys = random.split(random.PRNGKey(23), n_samples)
    samples = jax.vmap(lambda k: refresh_pseudofermion_tdf(k, theta, kappa))(keys)
    sample_cov = (samples.T @ jnp.conj(samples)) / n_samples

    rel_diff = jnp.linalg.norm(sample_cov - target) / jnp.linalg.norm(target)
    assert rel_diff < 0.2


def test_refresh_pseudofermion_tdf_stochastic_bulk_covariance():
    """Stochastic-bulk refresh has correct per-slice and transfer covariances."""
    L, Lt = 2, 2
    key = random.PRNGKey(24)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    from tdf.reduced import transfer_matrices
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    n_samples = 2000
    keys = random.split(random.PRNGKey(25), n_samples)
    samples = jax.vmap(lambda k: refresh_pseudofermion_tdf_stochastic_bulk(k, theta, kappa))(keys)
    phi_bulk_samples = samples[0]  # (n_samples, Lt, 2L)
    psi_samples = samples[1]       # (n_samples, 2L)

    # Check each time-slice bulk pseudofermion.
    for t in range(Lt):
        target_t = R[t] @ jnp.conj(R[t].T)
        sample_cov = (phi_bulk_samples[:, t, :].T @ jnp.conj(phi_bulk_samples[:, t, :])) / n_samples
        rel_diff = jnp.linalg.norm(sample_cov - target_t) / jnp.linalg.norm(target_t)
        assert rel_diff < 0.25

    # Check transfer-matrix pseudofermion.
    target_m = M @ jnp.conj(M.T)
    sample_cov = (psi_samples.T @ jnp.conj(psi_samples)) / n_samples
    rel_diff = jnp.linalg.norm(sample_cov - target_m) / jnp.linalg.norm(target_m)
    assert rel_diff < 0.25


def test_hmc_pseudofermion_tdf_stochastic_bulk_runs():
    """A short stochastic-bulk TDF HMC run executes and produces finite data."""
    L, Lt = 4, 4
    key = random.PRNGKey(26)
    key_field, key_run = random.split(key)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)
    kappa = dirac.kappa_from_mass(0.0)
    from tdf.hmc_pseudofermion import run_hmc_pseudofermion_tdf_stochastic_bulk

    history, configs = run_hmc_pseudofermion_tdf_stochastic_bulk(
        key_run, theta0, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1,
        dt=0.05, n_steps=3, tol=1e-6,
    )
    assert jnp.all(jnp.isfinite(history["plaquette"]))


@pytest.mark.parametrize("order", ["natural", "reverse", "balanced"])
def test_bulk_product_determinant_identity(order):
    """det(M_bulk) equals the exact bulk factor for all multiplication orders."""
    L, Lt = 4, 4
    key = random.PRNGKey(27)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    from tdf.pseudofermion import _build_bulk_product
    from tdf.reduced import transfer_matrices
    R, _, _, bulk = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                      boundary_phase=-1.0)
    M_bulk = _build_bulk_product(R, order=order)
    product_det = jnp.linalg.det(M_bulk)
    rel_diff = jnp.abs(product_det - bulk) / jnp.abs(bulk)
    assert rel_diff < 1e-10


@pytest.mark.parametrize("order", ["natural", "reverse", "balanced"])
def test_refresh_pseudofermion_tdf_bulk_product_covariance(order):
    """Bulk-product refresh produces Cov(phi_bulk) = M_bulk M_bulk^dagger."""
    L, Lt = 2, 2
    key = random.PRNGKey(28)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    from tdf.pseudofermion import _build_bulk_product
    from tdf.reduced import transfer_matrices
    R, _, _, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    M_bulk = _build_bulk_product(R, order=order)
    target = M_bulk @ jnp.conj(M_bulk.T)

    n_samples = 2000
    keys = random.split(random.PRNGKey(29), n_samples)
    samples = jax.vmap(
        lambda k: refresh_pseudofermion_tdf_bulk_product(k, theta, kappa, order=order)[0]
    )(keys)
    sample_cov = (samples.T @ jnp.conj(samples)) / n_samples

    rel_diff = jnp.linalg.norm(sample_cov - target) / jnp.linalg.norm(target)
    assert rel_diff < 0.25


@pytest.mark.parametrize("order", ["natural", "reverse", "balanced"])
def test_hmc_pseudofermion_tdf_bulk_product_runs(order):
    """A short bulk-product TDF HMC run executes for each ordering."""
    L, Lt = 4, 4
    key = random.PRNGKey(30)
    key_field, key_run = random.split(key)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)
    kappa = dirac.kappa_from_mass(0.0)
    from tdf.hmc_pseudofermion import run_hmc_pseudofermion_tdf_bulk_product

    history, configs = run_hmc_pseudofermion_tdf_bulk_product(
        key_run, theta0, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1,
        dt=0.05, n_steps=3, tol=1e-6, order=order,
    )
    assert jnp.all(jnp.isfinite(history["plaquette"]))


def test_block_diagonal_action_matches_slice_sum():
    """Block-diagonal action equals the per-slice stochastic-bulk action.

    Both formulations represent the same bulk factor and the same
    transfer-matrix factor, so their total actions must agree for the same
    pseudofermion fields.
    """
    L, Lt = 2, 2
    key = random.PRNGKey(31)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    phi_bulk, psi = refresh_pseudofermion_tdf_block_diagonal(key, theta, kappa)
    phi_blocks = phi_bulk.reshape(Lt, 2 * L)

    action_slice = pseudofermion_action_tdf_stochastic_bulk(
        theta, phi_blocks, psi, kappa, tol=1e-9
    )
    action_bd = pseudofermion_action_tdf_block_diagonal(
        theta, phi_bulk, psi, kappa, tol=1e-9
    )

    assert jnp.isfinite(action_slice)
    assert jnp.isfinite(action_bd)
    # CG convergence can introduce small differences.
    assert jnp.abs(action_bd - action_slice) < 1e-4 * jnp.maximum(
        jnp.abs(action_slice), 1.0
    )


def test_refresh_pseudofermion_tdf_block_diagonal_covariance():
    """Block-diagonal refresh has block-wise Cov(phi_t) = R_t R_t^dagger."""
    L, Lt = 2, 2
    key = random.PRNGKey(32)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    from tdf.reduced import transfer_matrices
    R, _, _, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    d = R.shape[1]

    n_samples = 2000
    keys = random.split(random.PRNGKey(33), n_samples)
    samples = jax.vmap(
        lambda k: refresh_pseudofermion_tdf_block_diagonal(k, theta, kappa)[0]
    )(keys)
    sample_blocks = samples.reshape(n_samples, Lt, d)

    for t in range(Lt):
        target_t = R[t] @ jnp.conj(R[t].T)
        sample_cov = (sample_blocks[:, t, :].T @ jnp.conj(sample_blocks[:, t, :])) / n_samples
        rel_diff = jnp.linalg.norm(sample_cov - target_t) / jnp.linalg.norm(target_t)
        assert rel_diff < 0.25


def test_hmc_pseudofermion_tdf_block_diagonal_runs():
    """A short block-diagonal TDF HMC run executes and produces finite data."""
    L, Lt = 4, 4
    key = random.PRNGKey(34)
    key_field, key_run = random.split(key)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)
    kappa = dirac.kappa_from_mass(0.0)
    from tdf.hmc_pseudofermion import run_hmc_pseudofermion_tdf_block_diagonal

    history, configs = run_hmc_pseudofermion_tdf_block_diagonal(
        key_run, theta0, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1,
        dt=0.05, n_steps=3, tol=1e-6,
    )
    assert jnp.all(jnp.isfinite(history["plaquette"]))


def test_block_cyclic_tm_determinant_identity():
    """Block-cyclic determinant equals the transfer-matrix determinant.

    The block-cyclic matrix B satisfies det B = det M, and therefore
    det(B B^dagger) = |det M|^2.
    """
    from tdf.reduced import transfer_matrices

    L, Lt = 4, 4
    key = random.PRNGKey(35)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)
    R, S, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    # Build explicit B for comparison.
    d = T.shape[0]
    B = jnp.zeros((Lt * d, Lt * d), dtype=T.dtype)
    for t in range(Lt):
        B = B.at[t * d:(t + 1) * d, t * d:(t + 1) * d].set(I)
        T_t = jnp.linalg.solve(R[t], S[t])
        j = ((t + 1) % Lt) * d
        B = B.at[t * d:(t + 1) * d, j:j + d].set(-sign * T_t)

    det_M = jnp.linalg.det(M)
    det_B = jnp.linalg.det(B)
    det_BBdagger = jnp.linalg.det(B @ jnp.conj(B.T))

    assert jnp.abs(det_B - det_M) < 1e-6 * jnp.maximum(jnp.abs(det_M), 1.0)
    assert jnp.abs(det_BBdagger - jnp.abs(det_M) ** 2) < 1e-4 * jnp.maximum(
        jnp.abs(det_M) ** 2, 1.0
    )


def test_block_cyclic_tm_matvec_matches_explicit():
    """The implicit block-cyclic matvec matches the explicit B B^dagger matvec."""
    from tdf.pseudofermion import _make_block_cyclic_tm_matvec
    from tdf.reduced import transfer_matrices

    L, Lt = 4, 4
    key = random.PRNGKey(36)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)
    R, S, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    d = T.shape[0]

    # Build explicit B.
    I = jnp.eye(d, dtype=T.dtype)
    B = jnp.zeros((Lt * d, Lt * d), dtype=T.dtype)
    for t in range(Lt):
        B = B.at[t * d:(t + 1) * d, t * d:(t + 1) * d].set(I)
        T_t = jnp.linalg.solve(R[t], S[t])
        j = ((t + 1) % Lt) * d
        B = B.at[t * d:(t + 1) * d, j:j + d].set(-sign * T_t)

    A_fn = _make_block_cyclic_tm_matvec(R, S, sign)
    x = refresh_pseudofermion(key, (Lt * d,))

    y_explicit = B @ (jnp.conj(B.T) @ x)
    y_implicit = A_fn(x)

    rel_diff = jnp.linalg.norm(y_implicit - y_explicit) / jnp.linalg.norm(y_explicit)
    assert rel_diff < 1e-10


def test_block_cyclic_tm_action_is_real_and_finite():
    """The block-cyclic TM action is real and finite for a book-style psi."""
    L, Lt = 4, 4
    key = random.PRNGKey(37)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    phi_bulk, psi_tm = refresh_pseudofermion_tdf_block_cyclic_tm(
        key, theta, kappa
    )
    action = pseudofermion_action_tdf_block_cyclic_tm(
        theta, phi_bulk, psi_tm, kappa, tol=1e-6
    )
    assert jnp.isfinite(action)
    assert jnp.isreal(action)


def test_block_cyclic_tm_force_matches_numerical_gradient():
    """The block-cyclic TM force matches a finite-difference gradient."""
    L, Lt = 2, 2
    key = random.PRNGKey(38)
    theta = lattice.make_gauge_field(L, Lt, key)
    kappa = dirac.kappa_from_mass(0.0)

    phi_bulk, psi_tm = refresh_pseudofermion_tdf_block_cyclic_tm(
        key, theta, kappa
    )

    def action_fn(t):
        return pseudofermion_action_tdf_block_cyclic_tm(
            t, phi_bulk, psi_tm, kappa, tol=1e-6
        )

    # JAX gradient of the action.
    grad_action = jax.grad(action_fn)(theta)

    # Finite-difference check on one component.
    eps = 1e-5
    idx = (0, 0, 0)
    theta_plus = theta.at[idx].add(eps)
    theta_minus = theta.at[idx].add(-eps)
    fd = (action_fn(theta_plus) - action_fn(theta_minus)) / (2.0 * eps)

    rel_err = abs(fd - grad_action[idx]) / jnp.maximum(
        abs(grad_action[idx]), 1.0
    )
    assert rel_err < 1e-3


def test_hmc_pseudofermion_tdf_block_cyclic_tm_runs():
    """A short block-cyclic TM TDF HMC run executes and produces finite data."""
    L, Lt = 4, 4
    key = random.PRNGKey(39)
    key_field, key_run = random.split(key)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)
    kappa = dirac.kappa_from_mass(0.0)
    from tdf.hmc_pseudofermion import run_hmc_pseudofermion_tdf_block_cyclic_tm

    history, configs = run_hmc_pseudofermion_tdf_block_cyclic_tm(
        key_run, theta0, kappa=kappa,
        n_therm=2, n_measure=2, n_skip=1,
        dt=0.05, n_steps=3, tol=1e-6,
    )
    assert jnp.all(jnp.isfinite(history["plaquette"]))


def test_corrected_pseudofermion_weight_identity():
    """The corrected weight exp(-S_pf + |phi|²) estimates det A unbiasedly.

    For a positive-definite matrix A with all eigenvalues smaller than 2, the
    identity (Gattringer & Lang, eq. (8.63))

        det A = < exp(-phi^dagger (A^{-1} - I) phi) >

    holds.  We verify it on a small matrix where the variance is manageable.
    """
    N = 4
    key = random.PRNGKey(100)
    # Build a Hermitian positive-definite matrix with eigenvalues in (0.5, 1.5).
    eigs = 0.5 + random.uniform(key, (N,))
    Q, _ = jnp.linalg.qr(random.normal(random.PRNGKey(101), (N, N)))
    A = (Q * eigs) @ jnp.conj(Q.T)
    assert jnp.all(jnp.linalg.eigvalsh(A) < 2.0)
    exact_det = float(jnp.real(jnp.linalg.det(A)))

    n_samples = 1000
    keys = random.split(random.PRNGKey(102), n_samples)

    @jax.jit
    def weight_fn(k):
        phi = refresh_pseudofermion(k, (N,))
        chi = jnp.linalg.solve(A, phi)
        action = jnp.real(jnp.vdot(phi, chi))
        norm_term = jnp.real(jnp.vdot(phi, phi))
        return jnp.exp(-action + norm_term)

    weights = jax.vmap(weight_fn)(keys)

    mean_w = float(jnp.mean(weights))
    sem_w = float(jnp.std(weights) / jnp.sqrt(n_samples))
    # Allow a several-Sigma deviation because the variance, while finite, is
    # not tiny.
    assert abs(mean_w - exact_det) <= 10.0 * sem_w


def test_unified_determinant_comparison():
    """The unified comparison returns exact, standard and TDF estimates.

    With a modest number of pseudofermion samples the stochastic determinant
    estimates have large variance and are not guaranteed to agree with the
    exact value within one standard error.  This test only checks that the
    comparison executes, returns finite values, and reports the 1-sigma
    agreement flags.
    """
    from tdf.determinant_comparison import compare_for_size

    key = random.PRNGKey(9)
    result = compare_for_size(
        L=4, Lt=4, beta=5.0, mass=0.0, n_samples=50,
        tol=1e-6, maxiter=16, key=key
    )

    # TDF exact determinant agrees with full determinant.
    assert result["tdf_rel_diff"] < 1e-10

    # Agreement flags are present.
    assert isinstance(result["standard_agrees_1sigma"], bool)
    assert isinstance(result["tdf_agrees_1sigma"], bool)

    # Both pseudofermion estimates are finite.
    assert jnp.isfinite(result["standard"]["mean_det"])
    assert jnp.isfinite(result["standard"]["sem_det"])
    assert jnp.isfinite(result["tdf"]["mean_det"])
    assert jnp.isfinite(result["tdf"]["sem_det"])
