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


def refresh_pseudofermion_standard(key, theta, kappa):
    """Book-style pseudofermion refresh for standard HMC.

    The pseudofermion is drawn from the conditional distribution
    N(0, K K^dagger) by applying the Wilson-Dirac operator to a standard
    complex Gaussian chi:

        phi = K chi,    chi ~ N(0, I).

    With this refresh the action S_pf = phi^dagger (K K^dagger)^{-1} phi gives
    the correct 2-flavour weight after marginalising over phi.
    """
    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    chi = refresh_pseudofermion(key, (K.shape[0],))
    return K @ chi


def refresh_pseudofermion_tdf(key, theta, kappa):
    """Book-style pseudofermion refresh for TDF HMC.

    The pseudofermion is drawn from the conditional distribution
    N(0, M M^dagger) where M = I - (-1)^Lt T:

        psi = M eta,    eta ~ N(0, I).

    The bulk determinant is kept exact and enters the action as
    -2 log |bulk|.  After marginalising over psi the weight is
    |bulk|^2 |det M|^2 = |det K|^2.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    _, _, T, bulk = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                      boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T
    eta = refresh_pseudofermion(key, (T.shape[0],))
    return M @ eta


def refresh_pseudofermion_tdf_stochastic_bulk(key, theta, kappa):
    """Book-style refresh for fully pseudofermionized TDF HMC.

    In addition to the transfer-matrix pseudofermion psi ~ N(0, M M^dagger),
    each time-slice bulk factor det R_t is represented by an independent
    pseudofermion phi_t ~ N(0, R_t R_t^dagger):

        phi_t = R_t chi_t,    chi_t ~ N(0, I),
        psi   = M eta,        eta   ~ N(0, I).

    The full 2-flavour weight is recovered after marginalising over all
    pseudofermion fields.

    Returns
    -------
    phi_bulk : ndarray, shape (Lt, 2*L)
        Time-slice bulk pseudofermions.
    psi : ndarray, shape (2*L,)
        Transfer-matrix pseudofermion.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    key_bulk, key_tm = jax.random.split(key)
    chi = refresh_pseudofermion(key_bulk, (Lt, T.shape[0]))
    phi_bulk = jax.vmap(lambda R_t, chi_t: R_t @ chi_t)(R, chi)

    eta = refresh_pseudofermion(key_tm, (T.shape[0],))
    psi = M @ eta
    return phi_bulk, psi


def _make_standard_matvec(K):
    """Return A(x) = K K^dagger x for the standard pseudofermion estimator."""
    def A_fn(x):
        return K @ (jnp.conj(K.T) @ x)
    return A_fn


def _make_bulk_slice_matvec(R_t):
    """Return A(x) = R_t R_t^dagger x for one time-slice bulk factor."""
    def A_fn(x):
        return R_t @ (jnp.conj(R_t.T) @ x)
    return A_fn


def _make_tdf_matvec(T, sign):
    """Return A(x) = M M^dagger x with M = I - sign * T."""
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    def A_fn(x):
        return M @ (jnp.conj(M.T) @ x)
    return A_fn


def _build_bulk_product(R, order="natural"):
    """Build M_bulk = prod_t R_t using the requested multiplication order.

    Parameters
    ----------
    R : ndarray, shape (Lt, d, d)
        Time-slice matrices.
    order : str
        One of "natural" (R_0 R_1 ... R_{Lt-1}),
        "reverse" (R_{Lt-1} ... R_1 R_0), or
        "balanced" (recursive pairwise products).

    Returns
    -------
    M_bulk : ndarray, shape (d, d)
        Product of the R_t matrices.
    """
    Lt = R.shape[0]
    if order == "natural":
        matrices = list(R)
    elif order == "reverse":
        matrices = list(R[::-1])
    elif order == "balanced":
        matrices = list(R)
    else:
        raise ValueError("order must be one of 'natural', 'reverse', 'balanced'")

    # For balanced order we repeatedly multiply adjacent pairs.
    if order == "balanced":
        while len(matrices) > 1:
            new_matrices = []
            for i in range(0, len(matrices), 2):
                if i + 1 < len(matrices):
                    new_matrices.append(matrices[i] @ matrices[i + 1])
                else:
                    new_matrices.append(matrices[i])
            matrices = new_matrices
        return matrices[0]

    # Natural or reverse: sequential product.
    M_bulk = matrices[0]
    for i in range(1, len(matrices)):
        M_bulk = M_bulk @ matrices[i]
    return M_bulk


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


def pseudofermion_action_tdf_stochastic_bulk(theta, phi_bulk, psi, kappa,
                                             tol=1e-9, maxiter=None,
                                             verbose=False):
    """Stochastic pseudofermion action for fully pseudofermionized TDF HMC.

    Every determinant factor is represented by pseudofermions:

        S = sum_t phi_t^dagger (R_t R_t^dagger)^{-1} phi_t
            + psi^dagger (M M^dagger)^{-1} psi .

    The transfer matrices and time-slice matrices are built once; CG solves
    are performed for each of the Lt + 1 pseudofermion fields.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0

    def bulk_slice_action(R_t, phi_t):
        A_fn = _make_bulk_slice_matvec(R_t)
        chi_t, n_iter, relres = cg_solve(A_fn, phi_t, tol=tol,
                                         maxiter=maxiter, verbose=verbose)
        logger.debug("Bulk-slice CG converged in %d iterations, relres=%.3e",
                     n_iter, relres)
        return jnp.real(jnp.vdot(phi_t, chi_t))

    bulk_action = jnp.sum(jax.vmap(bulk_slice_action)(R, phi_bulk))

    A_fn_tm = _make_tdf_matvec(T, sign)
    chi_tm, n_iter, relres = cg_solve(A_fn_tm, psi, tol=tol, maxiter=maxiter,
                                      verbose=verbose)
    logger.debug("TDF CG converged in %d iterations, relres=%.3e", n_iter, relres)
    tm_action = jnp.real(jnp.vdot(psi, chi_tm))

    return bulk_action + tm_action


def pseudofermion_force_tdf_stochastic_bulk(theta, phi_bulk, psi, kappa,
                                            tol=1e-9, maxiter=None,
                                            verbose=False):
    """Return the gauge force for the fully pseudofermionized TDF action."""
    def action_fn(t):
        return pseudofermion_action_tdf_stochastic_bulk(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter, verbose=verbose
        )
    return jax.grad(lambda t: -action_fn(t))


def refresh_pseudofermion_tdf_bulk_product(key, theta, kappa, order="natural"):
    """Book-style refresh for the single-product bulk TDF HMC.

    The whole bulk factor prod_t det R_t is represented by one pseudofermion

        phi_bulk = M_bulk chi,    chi ~ N(0, I),

    where M_bulk = prod_t R_t with the requested multiplication order.  The
    transfer-matrix factor still uses psi = M eta with eta ~ N(0, I).
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    M_bulk = _build_bulk_product(R, order=order)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    key_bulk, key_tm = jax.random.split(key)
    chi = refresh_pseudofermion(key_bulk, (M_bulk.shape[0],))
    phi_bulk = M_bulk @ chi
    eta = refresh_pseudofermion(key_tm, (T.shape[0],))
    psi = M @ eta
    return phi_bulk, psi


def pseudofermion_action_tdf_bulk_product(theta, phi_bulk, psi, kappa,
                                          tol=1e-9, maxiter=None,
                                          verbose=False, order="natural"):
    """Stochastic action for single-product bulk TDF HMC.

    The bulk factor is represented by one pseudofermion:

        S_bulk = phi_bulk^dagger (M_bulk M_bulk^dagger)^{-1} phi_bulk ,

    with M_bulk = prod_t R_t.  The transfer-matrix factor uses psi as before.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    M_bulk = _build_bulk_product(R, order=order)
    sign = -1.0 if Lt % 2 == 1 else 1.0

    A_fn_bulk = _make_standard_matvec(M_bulk)
    chi_bulk, n_iter, relres = cg_solve(A_fn_bulk, phi_bulk, tol=tol,
                                        maxiter=maxiter, verbose=verbose)
    logger.debug("Bulk-product CG converged in %d iterations, relres=%.3e",
                 n_iter, relres)
    bulk_action = jnp.real(jnp.vdot(phi_bulk, chi_bulk))

    A_fn_tm = _make_tdf_matvec(T, sign)
    chi_tm, n_iter, relres = cg_solve(A_fn_tm, psi, tol=tol, maxiter=maxiter,
                                      verbose=verbose)
    logger.debug("TDF CG converged in %d iterations, relres=%.3e", n_iter, relres)
    tm_action = jnp.real(jnp.vdot(psi, chi_tm))

    return bulk_action + tm_action


def pseudofermion_force_tdf_bulk_product(theta, phi_bulk, psi, kappa,
                                         tol=1e-9, maxiter=None,
                                         verbose=False, order="natural"):
    """Return the gauge force for the single-product bulk TDF action."""
    def action_fn(t):
        return pseudofermion_action_tdf_bulk_product(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter,
            verbose=verbose, order=order
        )
    return jax.grad(lambda t: -action_fn(t))


def _make_block_diagonal_bulk_matvec(R):
    """Return A(x) for the block-diagonal bulk matrix A = diag(R_t R_t^dagger).

    The operator never stores the off-diagonal zeros.  It reshapes the input
    vector into Lt blocks of size d = 2L, applies each R_t R_t^dagger block,
    and flattens the result back.  This lets a single CG solve act on the
    concatenated vector, which is more GPU-friendly than Lt separate solves.
    """
    Lt = R.shape[0]

    def A_fn(x):
        # x shape (Lt * d,).  Reshape to (Lt, d) for block-wise application.
        x_blocks = x.reshape(Lt, -1)
        y_blocks = jax.vmap(lambda R_t, x_t: R_t @ (jnp.conj(R_t.T) @ x_t))(R, x_blocks)
        return y_blocks.reshape(-1)

    return A_fn


def refresh_pseudofermion_tdf_block_diagonal(key, theta, kappa):
    """Book-style refresh for block-diagonal bulk TDF HMC.

    The bulk factor prod_t det R_t is represented by one concatenated
    pseudofermion

        phi_bulk = R chi,    chi ~ N(0, I),

    where R = diag(R_0, R_1, ..., R_{Lt-1}) is the block-diagonal bulk matrix.
    The transfer-matrix factor uses psi = M eta as before.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    I = jnp.eye(T.shape[0], dtype=T.dtype)
    M = I - sign * T

    key_bulk, key_tm = jax.random.split(key)
    d = R.shape[1]
    chi = refresh_pseudofermion(key_bulk, (Lt * d,))
    chi_blocks = chi.reshape(Lt, d)
    phi_blocks = jax.vmap(lambda R_t, chi_t: R_t @ chi_t)(R, chi_blocks)
    phi_bulk = phi_blocks.reshape(-1)

    eta = refresh_pseudofermion(key_tm, (T.shape[0],))
    psi = M @ eta
    return phi_bulk, psi


def pseudofermion_action_tdf_block_diagonal(theta, phi_bulk, psi, kappa,
                                            tol=1e-9, maxiter=None,
                                            verbose=False):
    """Stochastic action for block-diagonal bulk TDF HMC.

    The bulk factor is represented by one concatenated pseudofermion and one
    big CG solve on the block-diagonal matrix A = diag(R_t R_t^dagger):

        S_bulk = phi_bulk^dagger A^{-1} phi_bulk .

    The transfer-matrix factor uses psi as before.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, _, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0

    A_fn_bulk = _make_block_diagonal_bulk_matvec(R)
    chi_bulk, n_iter, relres = cg_solve(A_fn_bulk, phi_bulk, tol=tol,
                                        maxiter=maxiter, verbose=verbose)
    logger.debug("Block-diagonal bulk CG converged in %d iterations, relres=%.3e",
                 n_iter, relres)
    bulk_action = jnp.real(jnp.vdot(phi_bulk, chi_bulk))

    A_fn_tm = _make_tdf_matvec(T, sign)
    chi_tm, n_iter, relres = cg_solve(A_fn_tm, psi, tol=tol, maxiter=maxiter,
                                      verbose=verbose)
    logger.debug("TDF CG converged in %d iterations, relres=%.3e", n_iter, relres)
    tm_action = jnp.real(jnp.vdot(psi, chi_tm))

    return bulk_action + tm_action


def pseudofermion_force_tdf_block_diagonal(theta, phi_bulk, psi, kappa,
                                           tol=1e-9, maxiter=None,
                                           verbose=False):
    """Return the gauge force for the block-diagonal bulk TDF action."""
    def action_fn(t):
        return pseudofermion_action_tdf_block_diagonal(
            t, phi_bulk, psi, kappa, tol=tol, maxiter=maxiter, verbose=verbose
        )
    return jax.grad(lambda t: -action_fn(t))


def _make_block_cyclic_tm_matvec(R, S, sign):
    """Return A(x) = B B^dagger x for the block-cyclic transfer-matrix matrix.

    The temporal determinant factorisation gives

        det K = (prod_t det R_t) * det M,

    with M = I - sign T and T = T_0 T_1 ... T_{Lt-1},  T_t = R_t^{-1} S_t.
    Instead of forming the product T explicitly, we introduce the block-cyclic
    matrix B with blocks

        B_{t,t}     = I,
        B_{t,t+1}   = -sign T_t   (indices mod Lt).

    The standard block-cyclic determinant identity gives

        det B = det(I - sign T_0 T_1 ... T_{Lt-1}) = det M .

    Therefore |det M|^2 = det(B B^dagger), and we can represent the
    transfer-matrix factor with pseudofermions on the block-cyclic matrix
    A = B B^dagger.  Crucially, B never contains the exponentially
    ill-conditioned product T; its blocks are just the individual T_t, which
    are applied by solving the well-conditioned R_t matrices.

    Parameters
    ----------
    R, S : ndarray, shape (Lt, d, d)
        Time-slice matrices from ``transfer_matrices``.
    sign : float
        ``(-1)**Lt``.

    Returns
    -------
    A_fn : callable
        Function that computes B B^dagger x for a flattened vector x of length
        Lt * d.
    """
    Lt = R.shape[0]

    def apply_T_to_blocks(t_blocks):
        """Apply T_t = R_t^{-1} S_t block-wise to t_blocks."""
        return jax.vmap(lambda R_t, S_t, x_t: jnp.linalg.solve(R_t, S_t @ x_t))(R, S, t_blocks)

    def apply_T_dag_to_blocks(t_blocks):
        """Apply T_t^dagger = S_t^dagger (R_t^dagger)^{-1} block-wise.

        The blocks are shifted by one because B^dagger couples site i to the
        T_{i-1}^dagger block at site i-1.  The solve is performed first and
        S^dagger is applied afterwards: T^dagger x = S^dagger (R^dagger)^{-1} x.
        """
        R_prev = jnp.roll(R, 1, axis=0)
        S_prev = jnp.roll(S, 1, axis=0)
        y_blocks = jax.vmap(
            lambda R_t, x_t: jnp.linalg.solve(jnp.conj(R_t.T), x_t)
        )(R_prev, t_blocks)
        return jax.vmap(
            lambda S_t, y_t: jnp.conj(S_t.T) @ y_t
        )(S_prev, y_blocks)

    def A_fn(x):
        # x shape (Lt * d,).  Reshape to (Lt, d) for block-wise application.
        x_blocks = x.reshape(Lt, -1)

        # B^dagger x: (B^dagger x)_t = x_t - sign^* T_{t-1}^dagger x_{t-1}.
        x_prev = jnp.roll(x_blocks, 1, axis=0)
        y_blocks = x_blocks - jnp.conj(sign) * apply_T_dag_to_blocks(x_prev)

        # B y: (B y)_t = y_t - sign T_t y_{t+1}.
        y_next = jnp.roll(y_blocks, -1, axis=0)
        z_blocks = y_blocks - sign * apply_T_to_blocks(y_next)

        return z_blocks.reshape(-1)

    return A_fn


def refresh_pseudofermion_tdf_block_cyclic_tm(key, theta, kappa):
    """Book-style refresh for block-cyclic transfer-matrix TDF HMC.

    The bulk factor prod_t det R_t is represented by one pseudofermion per
    time slice, and the transfer-matrix factor det M is represented by a
    block-cyclic pseudofermion

        psi = B eta,    eta ~ N(0, I),

    where B is the block-cyclic matrix with identity on the diagonal and
    -sign T_t on the super-diagonal (with wrap-around).  This avoids forming
    the exponentially ill-conditioned transfer-matrix product T explicitly.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, S, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0
    d = R.shape[1]

    key_bulk, key_tm = jax.random.split(key)

    # Bulk pseudofermions: one per time slice.
    chi_bulk = refresh_pseudofermion(key_bulk, (Lt, d))
    phi_bulk = jax.vmap(lambda R_t, chi_t: R_t @ chi_t)(R, chi_bulk)

    # Block-cyclic transfer-matrix pseudofermion: psi = B eta.
    eta = refresh_pseudofermion(key_tm, (Lt * d,))
    eta_blocks = eta.reshape(Lt, d)
    eta_next = jnp.roll(eta_blocks, -1, axis=0)
    T_eta_next = jax.vmap(
        lambda R_t, S_t, x_t: jnp.linalg.solve(R_t, S_t @ x_t)
    )(R, S, eta_next)
    psi_blocks = eta_blocks - sign * T_eta_next
    psi_tm = psi_blocks.reshape(-1)

    return phi_bulk, psi_tm


def pseudofermion_action_tdf_block_cyclic_tm(theta, phi_bulk, psi_tm, kappa,
                                             tol=1e-9, maxiter=None,
                                             verbose=False):
    """Stochastic action for block-cyclic transfer-matrix TDF HMC.

    The bulk factor is represented by per-slice pseudofermions and the
    transfer-matrix factor by a block-cyclic pseudofermion:

        S = sum_t phi_t^dagger (R_t R_t^dagger)^{-1} phi_t
            + psi^dagger (B B^dagger)^{-1} psi .

    The block-cyclic matrix B is never formed explicitly; its action is
    implemented via sequential solves against the well-conditioned R_t
    matrices.
    """
    Lt, L = theta.shape[1], theta.shape[2]
    R, S, T, _ = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                   boundary_phase=-1.0)
    sign = -1.0 if Lt % 2 == 1 else 1.0

    def bulk_slice_action(R_t, phi_t):
        A_fn = _make_bulk_slice_matvec(R_t)
        chi_t, n_iter, relres = cg_solve(A_fn, phi_t, tol=tol,
                                         maxiter=maxiter, verbose=verbose)
        logger.debug("Bulk-slice CG converged in %d iterations, relres=%.3e",
                     n_iter, relres)
        return jnp.real(jnp.vdot(phi_t, chi_t))

    bulk_action = jnp.sum(jax.vmap(bulk_slice_action)(R, phi_bulk))

    A_fn_tm = _make_block_cyclic_tm_matvec(R, S, sign)
    chi_tm, n_iter, relres = cg_solve(A_fn_tm, psi_tm, tol=tol,
                                      maxiter=maxiter, verbose=verbose)
    logger.debug("Block-cyclic TM CG converged in %d iterations, relres=%.3e",
                 n_iter, relres)
    tm_action = jnp.real(jnp.vdot(psi_tm, chi_tm))

    return bulk_action + tm_action


def pseudofermion_force_tdf_block_cyclic_tm(theta, phi_bulk, psi_tm, kappa,
                                            tol=1e-9, maxiter=None,
                                            verbose=False):
    """Return the gauge force for the block-cyclic transfer-matrix TDF action."""
    def action_fn(t):
        return pseudofermion_action_tdf_block_cyclic_tm(
            t, phi_bulk, psi_tm, kappa, tol=tol, maxiter=maxiter,
            verbose=verbose
        )
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

    log_det_exact, shape, action_fn = _pseudofermion_setup(
        theta, kappa, tol, maxiter, algorithm
    )

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


def estimate_det_pseudofermion(theta, kappa, n_samples=100, tol=1e-9,
                               maxiter=None, algorithm="standard", key=None):
    """Unbiased stochastic estimate of |det K|^2 using pseudofermions.

    The pseudofermion field is drawn from the standard complex Gaussian
    N(0, I).  For a positive-definite matrix M the identity (Gattringer &
    Lang, *Quantum Chromodynamics on the Lattice*, eq. (8.63))

        det M = < exp( -chi^dagger (M^{-1} - I) chi ) >

    holds, where the expectation is over chi ~ N(0, I).  Writing the
    pseudofermion action as S_pf = chi^dagger M^{-1} chi, the per-sample
    weight is therefore

        w = exp(-S_pf + |chi|^2).

    For the TDF estimator the bulk determinant is kept exact, so the weight
    carries an additional factor |bulk|^2.

    The sample mean of w is an unbiased estimator of |det K|^2.  The standard
    error of the mean is std(w) / sqrt(n_samples).  The variance can be very
    large (formally infinite when M has eigenvalues larger than 2), so modest
    sample counts give honest but possibly wide error bars.

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
    dict with keys 'exact_det', 'mean_det', 'std_det', 'sem_det',
    'relative_error'.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    log_det_exact, shape, action_fn = _pseudofermion_setup(
        theta, kappa, tol, maxiter, algorithm
    )
    exact_det = jnp.exp(log_det_exact)

    # JIT-compile the per-sample weight computation and vectorise over
    # independent pseudofermion fields.
    @jax.jit
    def weight_fn(k):
        phi = refresh_pseudofermion(k, shape)
        action = action_fn(phi)
        # Corrected weight using Gattringer & Lang eq. (8.63):
        #   det M = < exp(-phi^dagger (M^{-1} - I) phi) >
        #         = < exp(-S_pf + |phi|^2) >.
        norm_term = jnp.real(jnp.vdot(phi, phi))
        return jnp.exp(-action + norm_term)

    keys = jax.random.split(key, n_samples)
    weights = jax.vmap(weight_fn)(keys)

    mean_det = jnp.mean(weights)
    std_det = jnp.std(weights)
    sem_det = std_det / jnp.sqrt(n_samples)
    return {
        "exact_det": float(exact_det),
        "mean_det": float(mean_det),
        "std_det": float(std_det),
        "sem_det": float(sem_det),
        "relative_error": float(jnp.abs(mean_det - exact_det) / jnp.abs(exact_det)),
    }


def _pseudofermion_setup(theta, kappa, tol, maxiter, algorithm):
    """Return exact logdet, pseudofermion shape, and action function.

    The action function builds the matrix-vector operator once and re-uses it
    across pseudofermion samples.
    """
    if algorithm == "standard":
        K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
        log_det_exact = 2.0 * jnp.linalg.slogdet(K)[1]
        shape = (K.shape[0],)
        A_fn = _make_standard_matvec(K)

        def action_fn(phi):
            chi, _, _ = cg_solve(A_fn, phi, tol=tol, maxiter=maxiter,
                                 verbose=False)
            return jnp.real(jnp.vdot(phi, chi))

    elif algorithm == "tdf":
        Lt, L = theta.shape[1], theta.shape[2]
        _, _, T, bulk = transfer_matrices(theta, mu=0.0, kappa=kappa,
                                          boundary_phase=-1.0)
        sign = -1.0 if Lt % 2 == 1 else 1.0
        I = jnp.eye(T.shape[0], dtype=T.dtype)
        M = I - sign * T
        log_det_exact = 2.0 * jnp.log(jnp.abs(bulk)) + 2.0 * jnp.linalg.slogdet(M)[1]
        shape = (T.shape[0],)
        A_fn = _make_tdf_matvec(T, sign)
        bulk_log = -2.0 * jnp.log(jnp.abs(bulk))

        def action_fn(psi):
            chi, _, _ = cg_solve(A_fn, psi, tol=tol, maxiter=maxiter,
                                 verbose=False)
            return jnp.real(jnp.vdot(psi, chi)) + bulk_log

    else:
        raise ValueError("algorithm must be 'standard' or 'tdf'")
    return log_det_exact, shape, action_fn
