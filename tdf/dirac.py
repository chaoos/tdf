"""Wilson-Dirac operator for the 2D U(1) Schwinger model."""

import logging

import jax.numpy as jnp

logger = logging.getLogger(__name__)


# Euclidean gamma matrices used in the thesis: gamma_0 = sigma_z, gamma_1 = sigma_x.
GAMMA_0 = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
GAMMA_1 = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
# Chirality matrix gamma_5 = -i * gamma_0 * gamma_1 = sigma_y.
GAMMA_5 = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)

# Projectors P^(+/- mu) = (1 \mp gamma_mu) / 2.
P0 = 0.5 * (jnp.eye(2, dtype=jnp.complex128) - GAMMA_0)   # forward temporal
PM0 = 0.5 * (jnp.eye(2, dtype=jnp.complex128) + GAMMA_0)  # backward temporal
P1 = 0.5 * (jnp.eye(2, dtype=jnp.complex128) - GAMMA_1)   # forward spatial
PM1 = 0.5 * (jnp.eye(2, dtype=jnp.complex128) + GAMMA_1)  # backward spatial


def kappa_from_mass(mass):
    """Convert bare mass m0 to hopping parameter kappa.

    The Wilson-Dirac operator is rescaled by 2*kappa = 1 / (m0 + 2*r),
    with Wilson parameter r = 1.
    """
    return 1.0 / (2.0 * (mass + 2.0))


def site_major_projector(P, L):
    r"""Return I_L \otimes P in site-major ordering (x, alpha).

    Parameters
    ----------
    P : ndarray, shape (2, 2)
        Dirac projector.
    L : int
        Spatial extent.

    Returns
    -------
    ndarray, shape (2*L, 2*L)
        Block-diagonal projector with 2x2 blocks P repeated L times.
    """
    return jnp.kron(jnp.eye(L, dtype=jnp.complex128), P)


def dirac_blocks(theta, mu, kappa):
    """Build the block-diagonal and hopping blocks of the Wilson-Dirac matrix.

    The returned B_t uses site-major ordering: index = x*2 + alpha.

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    mu : float
        Chemical potential.
    kappa : float
        Hopping parameter.

    Returns
    -------
    B : ndarray, shape (Lt, 2*L, 2*L)
        Spatial block B_t for each time slice.
    A_plus : ndarray, shape (Lt, L, L)
        Forward temporal link matrices (diagonal).
    A_minus : ndarray, shape (Lt, L, L)
        Backward temporal link matrices (diagonal).
    """
    cdtype = jnp.complex128
    Lt, L = theta.shape[1], theta.shape[2]
    U = jnp.exp(1j * theta)
    U0 = U[0]  # shape (Lt, L)
    U1 = U[1]  # shape (Lt, L)

    # A^+_t = e^{+mu} diag(U_0(t,x)), A^-_t = e^{-mu} diag(U_0(t,x)^*)
    idx = jnp.arange(L)
    A_plus = jnp.zeros((Lt, L, L), dtype=cdtype).at[:, idx, idx].set(jnp.exp(mu) * U0)
    A_minus = jnp.zeros((Lt, L, L), dtype=cdtype).at[:, idx, idx].set(jnp.exp(-mu) * jnp.conj(U0))

    # Build B_t in site-major ordering.
    B = jnp.zeros((Lt, 2 * L, 2 * L), dtype=cdtype)
    I2 = jnp.eye(2, dtype=cdtype)
    for t in range(Lt):
        Ut = U1[t]             # forward spatial links U_1(t,x)
        Ut_back = jnp.conj(jnp.roll(Ut, 1, axis=0))  # U_1(t, x-1)^*
        for x in range(L):
            row = 2 * x
            # diagonal identity
            B = B.at[t, row:row + 2, row:row + 2].add(I2)
            # forward spatial hop x -> x+1
            xp1 = (x + 1) % L
            col = 2 * xp1
            B = B.at[t, row:row + 2, col:col + 2].add(-2.0 * kappa * P1 * Ut[x])
            # backward spatial hop x -> x-1
            xm1 = (x - 1) % L
            col = 2 * xm1
            B = B.at[t, row:row + 2, col:col + 2].add(-2.0 * kappa * PM1 * Ut_back[x])

    return B, A_plus, A_minus


def wilson_dirac(theta, mu, kappa=None, mass=None, boundary_phase=-1.0):
    """Build the full Wilson-Dirac matrix K[U, mu].

    The matrix uses site-major ordering: index = (t*L + x)*2 + alpha.

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    mu : float
        Chemical potential.
    kappa : float, optional
        Hopping parameter. Either kappa or mass must be given.
    mass : float, optional
        Bare mass m0. Converted to kappa via kappa = 1/(2*(m0+2)).
    boundary_phase : float
        +1 for periodic temporal BC, -1 for anti-periodic (default).

    Returns
    -------
    K : ndarray, shape (2*L*Lt, 2*L*Lt)
        Full Wilson-Dirac matrix.
    """
    if kappa is None and mass is None:
        raise ValueError("Either kappa or mass must be provided.")
    if kappa is None:
        kappa = kappa_from_mass(mass)

    Lt, L = theta.shape[1], theta.shape[2]
    cdtype = jnp.complex128
    N = 2 * L * Lt
    K = jnp.zeros((N, N), dtype=cdtype)

    U = jnp.exp(1j * theta)
    U0 = U[0]
    U1 = U[1]

    # Apply boundary phase to temporal links at the wrap-around.
    if boundary_phase != 1.0:
        U0 = U0.at[-1, :].set(boundary_phase * U0[-1, :])

    def idx(t, x, alpha):
        return ((t * L + x) * 2 + alpha)

    for t in range(Lt):
        for x in range(L):
            s = t * L + x
            for alpha in range(2):
                i = s * 2 + alpha
                # diagonal
                K = K.at[i, i].add(1.0)

                # forward spatial: x -> x+1
                xp1 = (x + 1) % L
                for beta in range(2):
                    j = idx(t, xp1, beta)
                    K = K.at[i, j].add(-2.0 * kappa * P1[alpha, beta] * U1[t, x])

                # backward spatial: x -> x-1
                xm1 = (x - 1) % L
                for beta in range(2):
                    j = idx(t, xm1, beta)
                    K = K.at[i, j].add(-2.0 * kappa * PM1[alpha, beta] * jnp.conj(U1[t, xm1]))

                # forward temporal: t -> t+1
                tp1 = (t + 1) % Lt
                for beta in range(2):
                    j = idx(tp1, x, beta)
                    K = K.at[i, j].add(-2.0 * kappa * P0[alpha, beta] * U0[t, x] * jnp.exp(mu))

                # backward temporal: t -> t-1
                tm1 = (t - 1) % Lt
                for beta in range(2):
                    j = idx(tm1, x, beta)
                    K = K.at[i, j].add(-2.0 * kappa * PM0[alpha, beta] * jnp.conj(U0[tm1, x]) * jnp.exp(-mu))

    return K
