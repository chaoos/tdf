"""Hybrid Monte Carlo (HMC) samplers for the 2D Schwinger model."""

import jax
import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac


def action_standard(theta, beta, mu, kappa):
    """Standard Nf=2 action S[theta] = S_g - 2 log |det K[U, mu]|.

    The Boltzmann weight is e^{-S} = e^{-S_g} |det K[U, mu]|^2.
    For the 2-flavour model with isospin chemical potential mu_I,
    pass mu = mu_I / 2.

    Parameters
    ----------
    theta : ndarray, shape (2, Lt, L)
        Gauge link angles.
    beta : float
        Inverse gauge coupling.
    mu : float
        Quark chemical potential.
    kappa : float
        Hopping parameter.

    Returns
    -------
    float
        Real action.
    """
    S_g = lattice.gauge_action(theta, beta)
    K = dirac.wilson_dirac(theta, mu, kappa, boundary_phase=-1.0)
    logdet = jnp.linalg.slogdet(K)[1]
    return S_g - 2.0 * logdet


def action_standard_tdf(theta, beta, mu, kappa):
    """Standard Nf=2 action using the reduced determinant.

    This is equivalent to action_standard but uses the temporal determinant
    factorization.  It is intended as a cross-check and for later comparison.
    """
    from tdf.reduced import reduced_determinant

    S_g = lattice.gauge_action(theta, beta)
    detK = reduced_determinant(theta, mu, kappa, boundary_phase=-1.0)
    return S_g - 2.0 * jnp.log(jnp.abs(detK))


def make_force(action_fn):
    """Return the HMC force F = -dS/dtheta for a given action function."""
    return jax.jit(jax.grad(lambda t: -action_fn(t)))


def hmc_step(key, theta, action_fn, force_fn, dt, n_steps):
    """Single HMC trajectory with leapfrog integration.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    theta : ndarray
        Current gauge field.
    action_fn : callable
        Function theta -> S(theta).
    force_fn : callable
        Function theta -> F(theta) = -dS/dtheta.
    dt : float
        Leapfrog step size.
    n_steps : int
        Number of leapfrog steps per trajectory.

    Returns
    -------
    theta_new : ndarray
        New gauge field (accepted or old).
    accepted : bool
        Whether the proposal was accepted.
    """
    key_mom, key_acc = random.split(key)
    p = random.normal(key_mom, shape=theta.shape, dtype=theta.dtype)

    theta_prop = theta
    p = p + 0.5 * dt * force_fn(theta_prop)
    for i in range(n_steps):
        theta_prop = theta_prop + dt * p
        if i == n_steps - 1:
            p = p + 0.5 * dt * force_fn(theta_prop)
        else:
            p = p + dt * force_fn(theta_prop)

    H_old = 0.5 * jnp.sum(p ** 2) + action_fn(theta)
    H_new = 0.5 * jnp.sum(p ** 2) + action_fn(theta_prop)
    delta_H = H_new - H_old

    accept_prob = jnp.exp(-delta_H)
    accept_prob = jnp.clip(accept_prob, 0.0, 1.0)
    accepted = random.uniform(key_acc) < accept_prob

    theta_new = jnp.where(accepted, theta_prop, theta)
    return theta_new, accepted


def run_hmc(key, theta0, action_fn, force_fn, n_therm, n_measure, n_skip,
            dt, n_steps, observables=None):
    """Run HMC: thermalize, then measure.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    theta0 : ndarray
        Initial gauge field.
    action_fn, force_fn : callable
        Action and force functions.
    n_therm : int
        Number of thermalization trajectories.
    n_measure : int
        Number of measurements.
    n_skip : int
        Number of trajectories between measurements.
    dt : float
        Leapfrog step size.
    n_steps : int
        Number of leapfrog steps per trajectory.
    observables : dict[str, callable]
        Observables to measure. Each callable takes theta and returns a scalar.

    Returns
    -------
    history : dict
        Dictionary with arrays of measured observables plus 'accept_rate'.
    configs : ndarray
        Saved configurations after each measurement.
    """
    if observables is None:
        observables = {
            "plaquette": lattice.average_plaquette,
            "topological_charge": lattice.topological_charge,
        }

    theta = theta0
    keys = random.split(key, n_therm + n_measure * n_skip)

    # Thermalization
    for i in range(n_therm):
        theta, _ = hmc_step(keys[i], theta, action_fn, force_fn, dt, n_steps)

    # Measurement
    configs = []
    history = {name: [] for name in observables.keys()}
    history["accept"] = []

    key_idx = n_therm
    for m in range(n_measure):
        accepted_count = 0
        for _ in range(n_skip):
            theta, accepted = hmc_step(keys[key_idx], theta, action_fn, force_fn, dt, n_steps)
            accepted_count += int(accepted)
            key_idx += 1

        for name, obs_fn in observables.items():
            history[name].append(float(obs_fn(theta)))
        history["accept"].append(accepted_count / n_skip)
        configs.append(theta)

    history = {k: jnp.array(v) for k, v in history.items()}
    configs = jnp.stack(configs)
    return history, configs
