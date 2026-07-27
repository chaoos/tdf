"""TDF-based canonical HMC sampler for the 2D Schwinger model."""

import logging

import jax.numpy as jnp

from tdf import lattice, hmc
from tdf.canonical import canonical_determinants

logger = logging.getLogger(__name__)


def action_canonical(theta, beta, n, mu, kappa):
    """Canonical action for fixed isospin / quark number n: S_n = S_g - log |det_n|^2.

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    beta : float
        Inverse gauge coupling.
    n : int
        Canonical sector index, -L <= n <= L.
    mu : float
        Quark chemical potential (kept for API consistency; the canonical
        determinants are computed at this mu).
    kappa : float
        Hopping parameter.

    Returns
    -------
    float
        Real action.
    """
    L = theta.shape[2]
    if not (-L <= n <= L):
        raise ValueError(f"Sector index n must satisfy -L <= n <= L (L={L}), got {n}.")

    S_g = lattice.gauge_action(theta, beta)
    dets = canonical_determinants(theta, mu, kappa, boundary_phase=-1.0)
    det_n = dets[n + L]
    log_abs_det = jnp.log(jnp.abs(det_n))
    return S_g - 2.0 * log_abs_det


def make_canonical_force(beta, n, mu, kappa):
    """Return the HMC force for the canonical isospin-n action."""
    def action_fn(t):
        return action_canonical(t, beta, n, mu, kappa)
    return hmc.make_force(action_fn)


def run_hmc_canonical(key, theta0, beta, n, mu, kappa, n_therm, n_measure,
                      n_skip, dt, n_steps, observables=None):
    """Run canonical HMC in fixed isospin sector n.

    Parameters are analogous to hmc.run_hmc; see that function for details.
    """
    def action_fn(t):
        return action_canonical(t, beta, n, mu, kappa)

    force_fn = hmc.make_force(action_fn)
    return hmc.run_hmc(
        key, theta0, action_fn, force_fn,
        n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        dt=dt, n_steps=n_steps, observables=observables
    )
