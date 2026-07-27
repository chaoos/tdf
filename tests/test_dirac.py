"""Tests for the Wilson-Dirac operator and lattice utilities."""

import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac


def test_gauge_action_shape():
    key = random.PRNGKey(0)
    theta = lattice.make_gauge_field(4, 6, key)
    assert theta.shape == (2, 6, 4)
    action = lattice.gauge_action(theta, beta=5.0)
    assert jnp.isfinite(action)


def test_plaquette_sum_consistency():
    key = random.PRNGKey(1)
    theta = lattice.make_gauge_field(4, 4, key)
    plaq = lattice.plaquette_phases(theta)
    action1 = -5.0 * jnp.sum(jnp.real(plaq))
    action2 = lattice.gauge_action(theta, beta=5.0)
    assert jnp.isclose(action1, action2)


def test_wilson_dirac_shape():
    key = random.PRNGKey(2)
    theta = lattice.make_gauge_field(4, 6, key)
    K = dirac.wilson_dirac(theta, mu=0.0, mass=0.0)
    assert K.shape == (2 * 4 * 6, 2 * 4 * 6)


def test_gamma5_hermiticity():
    """gamma5 K gamma5 = K^dagger for mu = 0."""
    key = random.PRNGKey(3)
    L, Lt = 6, 8
    theta = lattice.make_gauge_field(L, Lt, key)
    K = dirac.wilson_dirac(theta, mu=0.0, mass=0.0, boundary_phase=-1.0)

    gamma5 = dirac.GAMMA_5
    Id = jnp.eye(L * Lt, dtype=jnp.complex128)
    Gamma5 = jnp.kron(Id, gamma5)

    lhs = Gamma5 @ K @ Gamma5
    rhs = jnp.conj(K.T)
    assert jnp.allclose(lhs, rhs, atol=1e-12)


def test_determinant_real_for_zero_mu():
    key = random.PRNGKey(4)
    theta = lattice.make_gauge_field(6, 6, key)
    K = dirac.wilson_dirac(theta, mu=0.0, mass=0.0)
    detK = jnp.linalg.det(K)
    assert abs(detK.imag) < 1e-10


def test_boundary_phase_determinant_real():
    """Both periodic and anti-periodic determinants are real for mu = 0."""
    key = random.PRNGKey(5)
    theta = lattice.make_gauge_field(4, 5, key)
    K_anti = dirac.wilson_dirac(theta, mu=0.0, mass=0.0, boundary_phase=-1.0)
    K_per = dirac.wilson_dirac(theta, mu=0.0, mass=0.0, boundary_phase=1.0)
    assert abs(jnp.linalg.det(K_anti).imag) < 1e-10
    assert abs(jnp.linalg.det(K_per).imag) < 1e-10


def test_clifford_algebra():
    """Verify the 1+1D Euclidean Clifford algebra {gamma_mu, gamma_nu} = 2 delta_mu nu I."""
    g0 = dirac.GAMMA_0
    g1 = dirac.GAMMA_1
    I2 = jnp.eye(2, dtype=jnp.complex128)

    # Anti-commutation relations.
    assert jnp.allclose(g0 @ g0 + g0 @ g0, 2 * I2, atol=1e-12)
    assert jnp.allclose(g1 @ g1 + g1 @ g1, 2 * I2, atol=1e-12)
    assert jnp.allclose(g0 @ g1 + g1 @ g0, jnp.zeros_like(I2), atol=1e-12)

    # Hermiticity of gamma matrices.
    assert jnp.allclose(g0, jnp.conj(g0.T), atol=1e-12)
    assert jnp.allclose(g1, jnp.conj(g1.T), atol=1e-12)


def test_gamma5_properties():
    """Verify gamma_5 = -i gamma_0 gamma_1 and its algebraic properties."""
    g0 = dirac.GAMMA_0
    g1 = dirac.GAMMA_1
    g5 = dirac.GAMMA_5
    I2 = jnp.eye(2, dtype=jnp.complex128)

    # Definition used in the code.
    assert jnp.allclose(g5, -1.0j * g0 @ g1, atol=1e-12)

    # Hermitian and unitary.
    assert jnp.allclose(g5, jnp.conj(g5.T), atol=1e-12)
    assert jnp.allclose(g5 @ g5, I2, atol=1e-12)

    # Anti-commutes with gamma_mu.
    assert jnp.allclose(g5 @ g0 + g0 @ g5, jnp.zeros_like(I2), atol=1e-12)
    assert jnp.allclose(g5 @ g1 + g1 @ g5, jnp.zeros_like(I2), atol=1e-12)


def test_chiral_projectors():
    """Verify P_+ and P_- = (I +/- gamma_5)/2 are orthogonal projectors."""
    g5 = dirac.GAMMA_5
    I2 = jnp.eye(2, dtype=jnp.complex128)
    P_plus = 0.5 * (I2 + g5)
    P_minus = 0.5 * (I2 - g5)

    # Idempotence.
    assert jnp.allclose(P_plus @ P_plus, P_plus, atol=1e-12)
    assert jnp.allclose(P_minus @ P_minus, P_minus, atol=1e-12)

    # Orthogonality.
    assert jnp.allclose(P_plus @ P_minus, jnp.zeros_like(I2), atol=1e-12)
    assert jnp.allclose(P_minus @ P_plus, jnp.zeros_like(I2), atol=1e-12)

    # Completeness.
    assert jnp.allclose(P_plus + P_minus, I2, atol=1e-12)

    # Eigenprojectors of gamma_5.
    assert jnp.allclose(g5 @ P_plus, P_plus, atol=1e-12)
    assert jnp.allclose(g5 @ P_minus, -P_minus, atol=1e-12)


def test_spin_projectors():
    r"""Verify the Wilson hopping projectors P^(+/- mu) = (1 \mp gamma_mu)/2."""
    g0 = dirac.GAMMA_0
    g1 = dirac.GAMMA_1
    I2 = jnp.eye(2, dtype=jnp.complex128)

    # Forward/backward temporal projectors.
    assert jnp.allclose(dirac.P0, 0.5 * (I2 - g0), atol=1e-12)
    assert jnp.allclose(dirac.PM0, 0.5 * (I2 + g0), atol=1e-12)

    # Forward/backward spatial projectors.
    assert jnp.allclose(dirac.P1, 0.5 * (I2 - g1), atol=1e-12)
    assert jnp.allclose(dirac.PM1, 0.5 * (I2 + g1), atol=1e-12)

    for P in [dirac.P0, dirac.PM0, dirac.P1, dirac.PM1]:
        # Idempotence.
        assert jnp.allclose(P @ P, P, atol=1e-12)
        # Hermitian (since gamma_mu are hermitian).
        assert jnp.allclose(P, jnp.conj(P.T), atol=1e-12)

    # Complementary pairs sum to identity.
    assert jnp.allclose(dirac.P0 + dirac.PM0, I2, atol=1e-12)
    assert jnp.allclose(dirac.P1 + dirac.PM1, I2, atol=1e-12)

    # Orthogonality within a pair.
    assert jnp.allclose(dirac.P0 @ dirac.PM0, jnp.zeros_like(I2), atol=1e-12)
    assert jnp.allclose(dirac.P1 @ dirac.PM1, jnp.zeros_like(I2), atol=1e-12)


def test_dirac_blocks_match_full_matrix():
    """Build K from blocks and compare to wilson_dirac."""
    key = random.PRNGKey(6)
    L, Lt = 4, 5
    theta = lattice.make_gauge_field(L, Lt, key)
    K_direct = dirac.wilson_dirac(theta, mu=0.0, mass=0.0, boundary_phase=-1.0)

    kappa = dirac.kappa_from_mass(0.0)
    B, A_plus, A_minus = dirac.dirac_blocks(theta, mu=0.0, kappa=kappa)
    P0_full = dirac.site_major_projector(dirac.P0, L)
    PM0_full = dirac.site_major_projector(dirac.PM0, L)

    N = 2 * L * Lt
    K_blocks = jnp.zeros((N, N), dtype=jnp.complex128)
    for t in range(Lt):
        s_start = 2 * L * t
        K_blocks = K_blocks.at[s_start:s_start + 2 * L, s_start:s_start + 2 * L].set(B[t])
        # forward temporal block (t -> t+1) uses A_plus[t]
        tp1 = (t + 1) % Lt
        block = -2.0 * kappa * P0_full @ jnp.kron(A_plus[t], jnp.eye(2, dtype=jnp.complex128))
        if t == Lt - 1:
            block = -block  # anti-periodic adds a minus sign
        K_blocks = K_blocks.at[s_start:s_start + 2 * L, 2 * L * tp1:2 * L * tp1 + 2 * L].set(block)
        # backward temporal block (t -> t-1) uses A_minus[t-1]
        tm1 = (t - 1) % Lt
        block = -2.0 * kappa * PM0_full @ jnp.kron(A_minus[tm1], jnp.eye(2, dtype=jnp.complex128))
        if t == 0:
            block = -block
        K_blocks = K_blocks.at[s_start:s_start + 2 * L, 2 * L * tm1:2 * L * tm1 + 2 * L].set(block)

    assert jnp.allclose(K_direct, K_blocks, atol=1e-12)
