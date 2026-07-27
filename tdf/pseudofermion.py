"""Pseudofermion estimators for the 2-flavour Schwinger model."""

import logging

import jax
import jax.numpy as jnp

from tdf import dirac
from tdf.reduced import transfer_matrices

logger = logging.getLogger(__name__)


def cg_solve(A_fn, b, tol=1e-9, maxiter=None, verbose=False):
    """Solve A x = b for a positive-definite A using the conjugate-gradient method.

    The stopping criterion is the relative residual
        ||A x - b|| / ||b||  <  tol .

    This implementation uses lax.scan so that it is differentiable through
    jax.grad.  The maximum number of iterations is always executed, but the
    solution is frozen once convergence is reached.

    Parameters
    ----------
    A_fn : callable
        Function that computes the matrix-vector product A @ x.
    b : ndarray
        Right-hand side vector.
    tol : float
        Relative residual tolerance.
    maxiter : int, optional
        Maximum number of iterations. Defaults to the dimension of b.
    verbose : bool
        If True, print the residual at every CG iteration using jax.debug.print.

    Returns
    -------
    x : ndarray
        Approximate solution.
    n_iter : int
        Number of CG iterations performed (or maxiter if not converged).
    relres : float
        Final relative residual.
    """
    b = jnp.asarray(b)
    x = jnp.zeros_like(b)
    r = b - A_fn(x)  # since x = 0 initially, this equals b
    p = r
    rsold = jnp.vdot(r, r)
    norm_b = jnp.linalg.norm(b)
    maxiter = b.shape[0] if maxiter is None else maxiter

    # Guard against zero RHS.
    norm_b = jnp.where(norm_b == 0.0, 1.0, norm_b)

    def scan_fn(carry, i):
        x, r, p, rsold, converged = carry
        Ap = A_fn(p)
        alpha = rsold / jnp.vdot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.vdot(r_new, r_new)
        relres = jnp.sqrt(jnp.real(rsnew)) / norm_b
        jax.lax.cond(
            verbose,
            lambda _: jax.debug.print(
                "CG iteration {i}: relative residual = {relres}",
                i=i, relres=relres
            ),
            lambda _: None,
            None,
        )
        beta = rsnew / rsold
        p_new = r_new + beta * p
        new_converged = converged | (relres < tol)
        # Freeze the solution once converged to avoid unnecessary updates.
        x_out = jnp.where(converged, x, x_new)
        r_out = jnp.where(converged, r, r_new)
        p_out = jnp.where(converged, p, p_new)
        rs_out = jnp.where(converged, rsold, rsnew)
        return (x_out, r_out, p_out, rs_out, new_converged), relres

    init = (x, r, p, rsold, False)
    (x_final, _, _, _, _), relres_history = jax.lax.scan(
        scan_fn, init, jnp.arange(maxiter)
    )
    final_relres = relres_history[-1]
    iters_to_converge = jnp.searchsorted(relres_history < tol, True) + 1
    iters_to_converge = jnp.where(
        jnp.any(relres_history < tol), iters_to_converge, maxiter
    )
    return x_final, iters_to_converge, final_relres


def refresh_pseudofermion(key, shape, dtype=jnp.complex128):
    """Draw a complex Gaussian pseudofermion vector.

    For a positive-definite matrix A, the complex Gaussian vector phi satisfies
        <|phi|^2> = 1  and  <phi_i phi_j^*> = delta_ij .
    """
    real = jax.random.normal(key, shape, dtype=jnp.float64)
    imag = jax.random.normal(key, shape, dtype=jnp.float64)
    return (real + 1j * imag) / jnp.sqrt(2.0)


def _make_standard_matvec(K):
    """Return A(x) = K K^dagger x for the standard pseudofermion estimator."""
    def A_fn(x):
        return K @ (jnp.conj(K.T) @ x)
    return A_fn


def _make_tdf_matvec(T, sign):
    """Return A(x) = M M^dagger x with M = I - sign * T."""
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    def A_fn(x):
        return M @ (jnp.conj(M.T) @ x)
    return A_fn


def pseudofermion_action_standard(theta, phi, kappa, tol=1e-9, maxiter=None,
                                  verbose=False):
    """Stochastic pseudofermion action for the standard 2-flavour estimator.

    Returns phi^dagger (K K^dagger)^{-1} phi for the Wilson-Dirac operator K.
    This is an unbiased estimator of log det(K K^dagger) up to an additive
    constant (the Gaussian normalization).
    """
    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    A_fn = _make_standard_matvec(K)
    chi, n_iter, relres = cg_solve(A_fn, phi, tol=tol, maxiter=maxiter,
                                   verbose=verbose)
    logger.debug("Standard CG converged in %d iterations, relres=%.3e", n_iter, relres)
    return jnp.real(jnp.vdot(phi, chi))


def pseudofermion_force_standard(theta, phi, kappa, tol=1e-9, maxiter=None,
                                 verbose=False):
    """Return the gauge force for the standard pseudofermion action."""
    def action_fn(t):
        return pseudofermion_action_standard(t, phi, kappa, tol=tol,
                                             maxiter=maxiter, verbose=verbose)
    return jax.grad(lambda t: -action_fn(t))


def pseudofermion_action_tdf(theta, psi, kappa, tol=1e-9, maxiter=None,
                             verbose=False):
    """Stochastic pseudofermion action for the TDF-based 2-flavour estimator.

    The full determinant factorisation reads
        det K = bulk * det(I - (-1)^Lt T) .
    The bulk factor is kept exact; pseudofermions estimate
        |det(I - (-1)^Lt T)|^2 .
    """
    Lt, L = theta.shape[1], theta.shape[2]
    _, _, T, bulk = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                      boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    A_fn = _make_tdf_matvec(T, sign)
    chi, n_iter, relres = cg_solve(A_fn, psi, tol=tol, maxiter=maxiter,
                                   verbose=verbose)
    logger.debug("TDF CG converged in %d iterations, relres=%.3e", n_iter, relres)
    return jnp.real(jnp.vdot(psi, chi)) - 2.0 * jnp.log(jnp.abs(bulk))


def pseudofermion_force_tdf(theta, psi, kappa, tol=1e-9, maxiter=None,
                            verbose=False):
    """Return the gauge force for the TDF pseudofermion action."""
    def action_fn(t):
        return pseudofermion_action_tdf(t, psi, kappa, tol=tol,
                                        maxiter=maxiter, verbose=verbose)
    return jax.grad(lambda t: -action_fn(t))


def estimate_pseudofermion_action_distribution(theta, kappa, n_samples=100,
                                                tol=1e-9, maxiter=None,
                                                algorithm="standard",
                                                key=None):
    """Sample the pseudofermion action and compare it to the exact log determinant.

    In pseudofermion HMC the fermion determinant is represented stochastically
    by the pseudofermion action S_pf.  For a fixed gauge field, S_pf is a random
    variable whose distribution depends on the pseudofermion noise.  The exact
    fermion action is
        S_exact = -log |det K|^2 .
    The pseudofermion action has expectation
        <S_pf> = tr((K K^dagger)^{-1})   (standard)
    which differs from S_exact by a configuration-dependent additive constant.
    The width of the distribution measures the stochastic noise of the
    estimator.

    Parameters
    ----------
    theta : ndarray
        Gauge field.
    kappa : float
        Hopping parameter.
    n_samples : int
        Number of pseudofermion fields to sample.
    tol, maxiter : passed to cg_solve.
    algorithm : "standard" or "tdf"
    key : jax.random.PRNGKey

    Returns
    -------
    dict with keys 'exact_logdet', 'mean_Spf', 'std_Spf', 'noise_Spf'.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    if algorithm == "standard":
        K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
        log_det_exact = 2.0 * jnp.linalg.slogdet(K)[1]
        shape = (K.shape[0],)
        action_fn = lambda phi: pseudofermion_action_standard(
            theta, phi, kappa, tol=tol, maxiter=maxiter, verbose=False
        )
    elif algorithm == "tdf":
        Lt, L = theta.shape[1], theta.shape[2]
        _, _, T, bulk = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                          boundary_phase=-1.0)
        sign = -1.0 if Lt % 2 == 1 else 1.0
        I = jnp.eye(T.shape[0], dtype=T.dtype)
        M = I - sign * T
        log_det_exact = 2.0 * jnp.log(jnp.abs(bulk)) + 2.0 * jnp.linalg.slogdet(M)[1]
        shape = (T.shape[0],)
        action_fn = lambda psi: pseudofermion_action_tdf(
            theta, psi, kappa, tol=tol, maxiter=maxiter, verbose=False
        )
    else:
        raise ValueError("algorithm must be 'standard' or 'tdf'")

    actions = []
    keys = jax.random.split(key, n_samples)
    for k in keys:
        phi = refresh_pseudofermion(k, shape)
        actions.append(action_fn(phi))
    actions = jnp.array(actions)

    return {
        "exact_logdet": float(log_det_exact),
        "mean_Spf": float(jnp.mean(actions)),
        "std_Spf": float(jnp.std(actions)),
        "noise_Spf": float(jnp.std(actions) / jnp.maximum(jnp.abs(log_det_exact), 1.0)),
    }
