"""Temporal determinant factorization (reduced determinant) for the Schwinger model."""

import logging

import jax.numpy as jnp

from tdf.dirac import dirac_blocks, site_major_projector, P0, PM0

logger = logging.getLogger(__name__)


def transfer_matrices(theta, mu, kappa, boundary_phase=-1.0):
    """Build the time-slice transfer matrices T_i and the bulk factor.

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    mu : float
        Chemical potential.
    kappa : float
        Hopping parameter.
    boundary_phase : float
        +1 for periodic temporal BC, -1 for anti-periodic (default).

    Returns
    -------
    R : ndarray, shape (Lt, 2*L, 2*L)
        R_i matrices.
    S : ndarray, shape (Lt, 2*L, 2*L)
        S_i matrices.
    T : ndarray, shape (2*L, 2*L)
        Product of transfer matrices T = T_0 T_1 ... T_{Lt-1}.
    bulk : complex
        Product of determinants of R_i.
    """
    B, A_plus, A_minus = dirac_blocks(theta, mu, kappa)
    Lt, L = B.shape[0], B.shape[1] // 2

    P_plus = site_major_projector(P0, L)   # forward temporal projector
    P_minus = site_major_projector(PM0, L)  # backward temporal projector
    I2L = jnp.eye(2 * L, dtype=jnp.complex128)

    # Apply boundary phase to the temporal links at the wrap-around.
    if boundary_phase != 1.0:
        A_plus = A_plus.at[-1, :, :].set(boundary_phase * A_plus[-1, :, :])
        A_minus = A_minus.at[-1, :, :].set(boundary_phase * A_minus[-1, :, :])

    R = jnp.zeros((Lt, 2 * L, 2 * L), dtype=jnp.complex128)
    S = jnp.zeros((Lt, 2 * L, 2 * L), dtype=jnp.complex128)
    bulk = jnp.array(1.0 + 0.0j, dtype=jnp.complex128)

    for i in range(Lt):
        # R_i = B_i P_+ - 2*kappa * kron(A^-_{i-1}, P_-)
        im1 = (i - 1) % Lt
        R_i = B[i] @ P_plus - 2.0 * kappa * jnp.kron(A_minus[im1], PM0)
        # S_i = B_i P_- - 2*kappa * kron(A^+_i, P_+)
        S_i = B[i] @ P_minus - 2.0 * kappa * jnp.kron(A_plus[i], P0)
        R = R.at[i].set(R_i)
        S = S.at[i].set(S_i)
        bulk *= jnp.linalg.det(R_i)

    # T = T_0 T_1 ... T_{Lt-1} with T_i = R_i^{-1} S_i.
    T = I2L
    for i in range(Lt):
        T_i = jnp.linalg.solve(R[i], S[i])
        T = T @ T_i

    return R, S, T, bulk


def reduced_determinant(theta, mu, kappa=None, mass=None, boundary_phase=-1.0):
    """Compute det K[U, mu] via the temporal determinant factorization.

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    mu : float
        Chemical potential.
    kappa : float, optional
        Hopping parameter. Either kappa or mass must be given.
    mass : float, optional
        Bare mass m0.
    boundary_phase : float
        +1 for periodic temporal BC, -1 for anti-periodic (default).

    Returns
    -------
    detK : complex
        Reduced determinant.
    """
    from tdf.dirac import kappa_from_mass

    if kappa is None and mass is None:
        raise ValueError("Either kappa or mass must be provided.")
    if kappa is None:
        kappa = kappa_from_mass(mass)

    _, _, T, bulk = transfer_matrices(theta, mu, kappa, boundary_phase=boundary_phase)

    Lt, L = theta.shape[1], theta.shape[2]
    eye = jnp.eye(2 * L, dtype=jnp.complex128)
    # The boundary condition is already encoded in the transfer matrices T_i
    # computed by transfer_matrices().  The Schur-complement sign depends on Lt:
    #   det K = (prod det R_i) * det(I - (-1)^Lt T).
    phase = -1.0 if Lt % 2 == 1 else 1.0  # (-1)^Lt
    return bulk * jnp.linalg.det(eye - phase * T)
