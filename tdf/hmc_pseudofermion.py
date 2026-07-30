"""Pseudofermion HMC samplers for the 2-flavour Schwinger model."""

import logging

import jax.numpy as jnp
from jax import random

from tdf import hmc, lattice
from tdf.pseudofermion import (
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
    refresh_pseudofermion_standard,
    refresh_pseudofermion_tdf,
    refresh_pseudofermion_tdf_block_cyclic_tm,
    refresh_pseudofermion_tdf_block_diagonal,
    refresh_pseudofermion_tdf_bulk_product,
    refresh_pseudofermion_tdf_stochastic_bulk,
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
    refresh_fn,
    n_therm,
    n_measure,
    n_skip,
    dt,
    n_steps,
    observables=None,
):
    """Run pseudofermion HMC.

    At the beginning of each trajectory the pseudofermion field is refreshed
    from its correct conditional distribution (book-style: phi = D chi with
    chi ~ N(0, I)).  The same pseudofermion vector is then used for the entire
    leapfrog trajectory.

    Parameters
    ----------
    key : jax.random.PRNGKey
    theta0 : ndarray
        Initial gauge field.
    action_factory : callable
        Function theta, pf -> S(theta, pf).
    force_factory : callable
        Function theta, pf -> F(theta) = -dS/dtheta.
    refresh_fn : callable
        Function key, theta -> pf that refreshes the pseudofermion from its
        conditional distribution for the current gauge field.
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
        pf = refresh_fn(key_pf, theta)
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
            pf = refresh_fn(key_pf, theta)
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
    """Standard pseudofermion HMC for the 2-flavour Schwinger model.

    The pseudofermion is refreshed from N(0, K K^dagger) by phi = K chi with
    chi ~ N(0, I), as described in Gattringer & Lang.
    """
    def action_factory(t, pf):
        return pseudofermion_action_standard(t, pf, kappa, tol=tol,
                                             maxiter=maxiter, verbose=verbose)

    def force_factory(t, pf):
        return pseudofermion_force_standard(t, pf, kappa, tol=tol,
                                            maxiter=maxiter, verbose=verbose)

    def refresh_fn(key_pf, theta):
        return refresh_pseudofermion_standard(key_pf, theta, kappa)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, refresh_fn,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )


def run_hmc_pseudofermion_tdf(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None
):
    """TDF pseudofermion HMC for the 2-flavour Schwinger model.

    The pseudofermion is refreshed from N(0, M M^dagger) by psi = M eta with
    eta ~ N(0, I), where M = I - (-1)^Lt T.  The bulk determinant is kept
    exact.
    """
    def action_factory(t, pf):
        return pseudofermion_action_tdf(t, pf, kappa, tol=tol,
                                        maxiter=maxiter, verbose=verbose)

    def force_factory(t, pf):
        return pseudofermion_force_tdf(t, pf, kappa, tol=tol,
                                       maxiter=maxiter, verbose=verbose)

    def refresh_fn(key_pf, theta):
        return refresh_pseudofermion_tdf(key_pf, theta, kappa)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, refresh_fn,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )


def run_hmc_pseudofermion_tdf_stochastic_bulk(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None
):
    """Fully pseudofermionized TDF HMC for the 2-flavour Schwinger model.

    Both the bulk factors det R_t and the transfer-matrix factor det M are
    represented by book-style pseudofermions.  The total number of auxiliary
    fields is Lt + 1, each living in the 2L-dimensional time-slice space.
    """
    def action_factory(t, pf):
        phi_bulk, psi = pf
        return pseudofermion_action_tdf_stochastic_bulk(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter, verbose=verbose
        )

    def force_factory(t, pf):
        phi_bulk, psi = pf
        return pseudofermion_force_tdf_stochastic_bulk(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter, verbose=verbose
        )

    def refresh_fn(key_pf, theta):
        return refresh_pseudofermion_tdf_stochastic_bulk(key_pf, theta, kappa)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, refresh_fn,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )


def run_hmc_pseudofermion_tdf_bulk_product(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None,
    order="natural"
):
    """Single-product bulk TDF HMC for the 2-flavour Schwinger model.

    The whole bulk factor prod_t det R_t is represented by a single
    pseudofermion phi_bulk = M_bulk chi, where M_bulk = prod_t R_t using the
    requested multiplication order.  The transfer-matrix factor uses a second
    pseudofermion psi.  Only two auxiliary fields are needed, regardless of Lt.
    """
    def action_factory(t, pf):
        phi_bulk, psi = pf
        return pseudofermion_action_tdf_bulk_product(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter,
            verbose=verbose, order=order
        )

    def force_factory(t, pf):
        phi_bulk, psi = pf
        return pseudofermion_force_tdf_bulk_product(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter,
            verbose=verbose, order=order
        )

    def refresh_fn(key_pf, theta):
        return refresh_pseudofermion_tdf_bulk_product(
            key_pf, theta, kappa, order=order
        )

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, refresh_fn,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )


def run_hmc_pseudofermion_tdf_block_diagonal(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None
):
    """Block-diagonal bulk TDF HMC for the 2-flavour Schwinger model.

    The time-slice bulk factors are represented by one concatenated
    pseudofermion that lives in the Lt*(2L)-dimensional block-diagonal space.
    A single CG solve is performed on the block-diagonal matrix
    A = diag(R_t R_t^dagger), whose action is implemented without storing any
    off-diagonal zeros.  The transfer-matrix factor uses a second pseudofermion
    psi as usual.
    """
    def action_factory(t, pf):
        phi_bulk, psi = pf
        return pseudofermion_action_tdf_block_diagonal(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter, verbose=verbose
        )

    def force_factory(t, pf):
        phi_bulk, psi = pf
        return pseudofermion_force_tdf_block_diagonal(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter, verbose=verbose
        )

    def refresh_fn(key_pf, theta):
        return refresh_pseudofermion_tdf_block_diagonal(key_pf, theta, kappa)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, refresh_fn,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )


def run_hmc_pseudofermion_tdf_block_cyclic_tm(
    key, theta0, kappa, n_therm, n_measure, n_skip, dt, n_steps,
    tol=1e-9, maxiter=None, verbose=False, observables=None
):
    """Block-cyclic transfer-matrix TDF HMC for the 2-flavour Schwinger model.

    The time-slice bulk factors are represented by one pseudofermion per slice,
    and the transfer-matrix factor det M is represented by a block-cyclic
    pseudofermion on the Lt*(2L)-dimensional block-cyclic space.  The
    block-cyclic matrix B is never formed explicitly; its action is implemented
    by sequential solves against the well-conditioned R_t matrices.  This avoids
    the exponentially ill-conditioned transfer-matrix product T = T_0 T_1 ...
    T_{Lt-1} and makes the TDF pseudofermion formulation stable at large Lt.
    """
    def action_factory(t, pf):
        phi_bulk, psi_tm = pf
        return pseudofermion_action_tdf_block_cyclic_tm(
            t, phi_bulk, psi_tm, kappa, tol=tol, maxiter=maxiter,
            verbose=verbose
        )

    def force_factory(t, pf):
        phi_bulk, psi_tm = pf
        return pseudofermion_force_tdf_block_cyclic_tm(
            t, phi_bulk, psi_tm, kappa, tol=tol, maxiter=maxiter,
            verbose=verbose
        )

    def refresh_fn(key_pf, theta):
        return refresh_pseudofermion_tdf_block_cyclic_tm(key_pf, theta, kappa)

    return run_hmc_pseudofermion(
        key, theta0, action_factory, force_factory, refresh_fn,
        n_therm, n_measure, n_skip, dt, n_steps, observables=observables
    )
