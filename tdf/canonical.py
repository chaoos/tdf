"""Canonical determinants from the transfer matrix spectrum."""

import logging

import jax.numpy as jnp

from tdf.reduced import transfer_matrices

logger = logging.getLogger(__name__)


def elementary_symmetric(eigenvalues):
    """Compute elementary symmetric polynomials e_m(lambda_1, ..., lambda_n).

    Returns coefficients c_m of  prod_i (z + lambda_i) = sum_{m=0}^n c_m z^m.
    c_0 = 1, c_n = prod_i lambda_i.

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues of the transfer matrix.

    Returns
    -------
    coeffs : ndarray, shape (n+1,)
        Coefficients c_m.
    """
    n = eigenvalues.shape[0]
    coeffs = jnp.zeros(n + 1, dtype=eigenvalues.dtype)
    coeffs = coeffs.at[0].set(1.0)
    for i in range(n):
        # c_m^{new} = c_m^{old} + lambda_i * c_{m-1}^{old}
        # Process backwards to use old values.
        for m in range(i + 1, 0, -1):
            coeffs = coeffs.at[m].set(coeffs[m] + eigenvalues[i] * coeffs[m - 1])
    return coeffs


def canonical_determinants(theta, mu, kappa=None, mass=None, boundary_phase=-1.0):
    """Compute all canonical determinants det_k(K[U]) for k = -L, ..., L.

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
    dets : ndarray, shape (2*L+1,)
        dets[idx] corresponds to k = idx - L.
    """
    from tdf.dirac import kappa_from_mass

    if kappa is None and mass is None:
        raise ValueError("Either kappa or mass must be provided.")
    if kappa is None:
        kappa = kappa_from_mass(mass)

    Lt, L = theta.shape[1], theta.shape[2]
    _, _, T, bulk = transfer_matrices(theta, mu, kappa, boundary_phase=boundary_phase)
    eigenvalues = jnp.linalg.eigvals(T)
    # Canonical determinants are coefficients of det(z I - (-1)^Lt T).
    shifted = -((-1.0) ** Lt) * eigenvalues
    coeffs = elementary_symmetric(shifted)
    # det_k(K) = bulk * c_{k+L}
    return bulk * coeffs


def canonical_determinant(theta, k, mu, kappa=None, mass=None, boundary_phase=-1.0):
    """Compute a single canonical determinant det_k(K[U]).

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    k : int
        Net fermion number, -L <= k <= L.
    mu : float
        Chemical potential (not used, kept for API consistency).
    kappa : float, optional
        Hopping parameter. Either kappa or mass must be given.
    mass : float, optional
        Bare mass m0.
    boundary_phase : float
        +1 for periodic temporal BC, -1 for anti-periodic (default).

    Returns
    -------
    det_k : complex
        Canonical determinant of order k.
    """
    L = theta.shape[2]
    dets = canonical_determinants(theta, mu, kappa=kappa, mass=mass, boundary_phase=boundary_phase)
    return dets[k + L]
