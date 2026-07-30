"""Unified comparison of exact, TDF, and pseudofermion determinants."""

import logging

import jax.numpy as jnp
from jax import random

from tdf import dirac, lattice
from tdf.pseudofermion import (
    estimate_det_pseudofermion,
    estimate_pseudofermion_action_distribution,
)
from tdf.reduced import reduced_determinant

logger = logging.getLogger(__name__)


def compare_for_size(L, Lt, beta, mass, n_samples=200, tol=1e-9,
                     maxiter=None, key=None):
    """Run the comparison for a single lattice size.

    Parameters
    ----------
    L, Lt : int
        Spatial and temporal lattice extent.
    beta : float
        Inverse gauge coupling (only used if a fresh field is generated).
    mass : float
        Bare fermion mass.
    n_samples : int
        Number of pseudofermion samples.
    tol : float
        CG relative residual tolerance.
    maxiter : int, optional
        Maximum CG iterations.
    key : jax.random.PRNGKey

    Returns
    -------
    dict
        Dictionary with exact log determinant, TDF log determinant,
        pseudofermion determinant estimates (mean and standard error of the
        mean), 1-sigma agreement flags, and pseudofermion action distributions
        for both standard and TDF estimators.
    """
    if key is None:
        key = random.PRNGKey(42)

    kappa = dirac.kappa_from_mass(mass)

    key_field, key_pf = random.split(key)
    key_std, key_tdf = random.split(key_pf)
    theta = lattice.make_gauge_field(L, Lt, key_field)

    # Exact determinant.
    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    log_det_exact = 2.0 * jnp.linalg.slogdet(K)[1]

    # TDF determinant.
    det_tdf = reduced_determinant(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    log_det_tdf = 2.0 * jnp.log(jnp.abs(det_tdf))

    rel_diff_tdf = float(
        jnp.abs(log_det_tdf - log_det_exact) / jnp.maximum(jnp.abs(log_det_exact), 1.0)
    )

    logger.info(
        "%dx%d: exact log|det K|^2 = %.8f, TDF = %.8f, rel. diff. = %.3e",
        L, Lt, float(log_det_exact), float(log_det_tdf), rel_diff_tdf
    )

    exact_det = jnp.exp(log_det_exact)

    # Standard pseudofermion determinant estimate.
    std_det = estimate_det_pseudofermion(
        theta, kappa, n_samples=n_samples, tol=tol, maxiter=maxiter,
        algorithm="standard", key=key_std
    )

    # TDF pseudofermion determinant estimate.
    tdf_det = estimate_det_pseudofermion(
        theta, kappa, n_samples=n_samples, tol=tol, maxiter=maxiter,
        algorithm="tdf", key=key_tdf
    )

    # 1-sigma agreement checks: |estimate - exact| <= standard error of mean.
    std_agrees = bool(abs(std_det["mean_det"] - float(exact_det)) <= std_det["sem_det"])
    tdf_agrees = bool(abs(tdf_det["mean_det"] - float(exact_det)) <= tdf_det["sem_det"])

    # Action distributions for noise comparison.
    std_action = estimate_pseudofermion_action_distribution(
        theta, kappa, n_samples=n_samples, tol=tol, maxiter=maxiter,
        algorithm="standard", key=key_std
    )
    tdf_action = estimate_pseudofermion_action_distribution(
        theta, kappa, n_samples=n_samples, tol=tol, maxiter=maxiter,
        algorithm="tdf", key=key_tdf
    )

    return {
        "L": L,
        "Lt": Lt,
        "exact_det": float(exact_det),
        "exact_logdet": float(log_det_exact),
        "tdf_logdet": float(log_det_tdf),
        "tdf_rel_diff": rel_diff_tdf,
        "standard": std_det,
        "tdf": tdf_det,
        "standard_agrees_1sigma": std_agrees,
        "tdf_agrees_1sigma": tdf_agrees,
        "standard_action": std_action,
        "tdf_action": tdf_action,
    }
