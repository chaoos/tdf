"""Pseudofermion HMC samplers for the 2-flavour Schwinger model."""

import logging

import jax.numpy as jnp
from jax import random

from tdf import hmc, lattice
from tdf.pseudofermion import (
    pseudofermion_action_standard,
    pseudofermion_action_tdf,
    pseudofermion_force_standard,
    pseudofermion_force_tdf,
    refresh_pseudofermion,
)

logger = logging.getLogger(__name__)


def _hmc_pseudofermion_step(key, theta, force_fn, action_fn, dt, n_steps):
    """Single HMC trajectory with leapfrog; pseudofermions are held fixed."""
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
    return theta_new, accepted, delta_H


def run_hmc_pseudofermion(
    key,
    theta0,
    action_factory,
    force_factory,
    pf_shape,
    n_therm,
    n_measure,
    n_skip,
    dt,
    n_steps,
    observables=None,
):
    """Run pseudofermion HMC.

    At the beginning of each trajectory the pseudofermion field is refreshed
    from a complex Gaussian.  The same pseudofermion vector is then used for
    the entire leapfrog trajectory.

    Parameters
    ----------
    key : jax.random.PRNGKey
    theta0 : ndarray
        Initial gauge field.
    action_factory : callable
        Function theta, pf -> S(theta, pf).
    force_factory : callable
        Function theta, pf -> F(theta) = -dS/dtheta.
    pf_shape : tuple
        Shape of the pseudofermion vector.
    n_therm, n_measure, n_skip, dt, n_steps : int/float
        Standard HMC parameters.
    observables : dict, optional
        Observables to measure.

    Returns
    -------
    history : dict
    configs : ndarray
    """
    if observables is None:
        observables = {
            "plaquette": lattice.average_plaquette,
            "topological_charge": lattice.topological_charge,
        }

    theta = theta0
    n_total = n_therm + n_measure * n_skip
    keys = random.split(key, n_total)

    logger.info("Starting pseudofermion HMC thermalization (%d trajectories)", n_therm)
    for i in range(n_therm):
        key_pf = random.fold_in(keys[i], 0)
        pf = refresh_pseudofermion(key_pf, pf_shape)
        action_fn = lambda t: action_factory(t, pf)
        force_fn = force_factory(theta, pf)  # use current theta to build force closure
        # Force closure depends on pf only; it is valid for any theta argument.
        theta, _, _ = _hmc_pseudofermion_step(
            keys[i], theta, force_fn, action_fn, dt, n_steps
        )
        if (i + 1) % max(1, n_therm // 10) == 0:
            logger.debug("Thermalization trajectory %d/%d complete", i + 1, n_therm)
    logger.info("Thermalization complete")

    configs = []
    history = {name: [] for name in observables.keys()}
    history["accept"] = []
    history["delta_H"] = []

    logger.info("Starting measurements (%d measurements, skip=%d)", n_measure, n_skip)
    key_idx = n_therm
    for m in range(n_measure):
        accepted_count = 0
        delta_H_block = []
        for _ in range(n_skip):
            key_pf = random.fold_in(keys[key_idx], 0)
            pf = refresh_pseudofermion(key_pf, pf_shape)
            action_fn = lambda t: action_factory(t, pf)
            force_fn = force_factory(theta, pf)
            theta, accepted, delta_H = _hmc_pseudofermion_step(
                keys[key_idx], theta, force_fn, action_fn, dt, n_steps
            )
            accepted_count += int(accepted)
            delta_H_block.append(float(delta_H))
            key_idx += 1

        for name, obs_fn in observables.items():
            history[name].append(float(obs_fn(theta)))
        history["accept"].append(accepted_count / n_skip)
        history["delta_H"].append(float(jnp.mean(jnp.array(delta_H_block))))
        configs.append(theta)
        if (m + 1) % max(1, n_measure // 10) == 0:
            logger.info("Measurement %d/%d complete", m + 1, n_measure)

    history = {k: jnp.array(v) for k, v in history.items()}
    configs = jnp.stack(configs)
    logger.info("Pseudofermion HMC run complete")
    return history, configs


def run_hmc_pseudofermion_standard(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None
):
    """Standard pseudofermion HMC for the 2-flavour Schwinger model."""
    L, Lt = theta0.shape[2], theta0.shape[1]
    pf_shape = (2 * L * Lt,)

    def action_factory(t, pf):
        return pseudofermion_action_standard(t, pf, kappa, tol=tol,
                                             maxiter=maxiter, verbose=verbose)

    def force_factory(t, pf):
        return pseudofermion_force_standard(t, pf, kappa, tol=tol,
                                            maxiter=maxiter, verbose=verbose)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, pf_shape,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )


def run_hmc_pseudofermion_tdf(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None
):
    """TDF pseudofermion HMC for the 2-flavour Schwinger model."""
    L, Lt = theta0.shape[2], theta0.shape[1]
    pf_shape = (2 * L,)

    def action_factory(t, pf):
        return pseudofermion_action_tdf(t, pf, kappa, tol=tol,
                                        maxiter=maxiter, verbose=verbose)

    def force_factory(t, pf):
        return pseudofermion_force_tdf(t, pf, kappa, tol=tol,
                                       maxiter=maxiter, verbose=verbose)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, pf_shape,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )
