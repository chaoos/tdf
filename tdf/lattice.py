"""U(1) lattice gauge field utilities for the 2D Schwinger model."""

import logging

import jax.numpy as jnp
from jax import random

logger = logging.getLogger(__name__)


def make_gauge_field(L, Lt, key, dtype=jnp.float64):
    """Return a random U(1) gauge field.

    Parameters
    ----------
    L : int
        Spatial extent.
    Lt : int
        Temporal extent.
    key : jax.random.PRNGKey
        Random key.
    dtype : dtype
        Floating point dtype for the link angles.

    Returns
    -------
    theta : ndarray, shape (2, Lt, L)
        theta[mu, t, x] is the angle of the link U_mu(x) = exp(i theta).
    """
    return random.uniform(key, shape=(2, Lt, L), dtype=dtype) * (2.0 * jnp.pi)


def link_phases(theta):
    """Convert link angles to complex phases.

    Returns
    -------
    U : ndarray, shape (2, Lt, L)
        U[mu, t, x] = exp(i * theta[mu, t, x]).
    """
    return jnp.exp(1j * theta)


def plaquette_phases(theta):
    """Return the complex plaquette phases U_P for each site.

    Convention (counter-clockwise):
        U_P(t,x) = U_0(t,x) U_1(t+1,x) U_0(t,x+1)^* U_1(t,x)^*

    Returns
    -------
    plaq : ndarray, shape (Lt, L)
        Complex plaquette values.
    """
    U = link_phases(theta)
    U0 = U[0]
    U1 = U[1]
    # periodic boundary conditions in both directions
    U0_tp1 = jnp.roll(U0, -1, axis=0)
    U0_xp1 = jnp.roll(U0, -1, axis=1)
    U1_tp1 = jnp.roll(U1, -1, axis=0)
    return U0 * U1_tp1 * jnp.conj(U0_xp1) * jnp.conj(U1)


def plaquette_angles(theta):
    """Return the plaquette angles arg(U_P) in (-pi, pi]."""
    return jnp.angle(plaquette_phases(theta))


def gauge_action(theta, beta):
    """Gauge action S_g = - beta * sum_P Re(U_P).

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Link angles.
    beta : float
        Inverse gauge coupling.

    Returns
    -------
    float
        Gauge action.
    """
    return -beta * jnp.sum(jnp.real(plaquette_phases(theta)))


def topological_charge(theta):
    """Geometric topological charge Q = (1/2pi) sum_P arg(U_P).

    Returns
    -------
    float
        Topological charge (integer-valued in continuum limit).
    """
    return jnp.sum(plaquette_angles(theta)) / (2.0 * jnp.pi)


def average_plaquette(theta):
    """Average plaquette Re(U_P)."""
    return jnp.mean(jnp.real(plaquette_phases(theta)))
