#!/usr/bin/env python3
"""Verify the environment, lattice utilities, and Wilson-Dirac operator."""

import logging
import os

# Do not pre-allocate the entire GPU memory pool; the RTX 3050 has only 6 GB.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import random

from tdf import configure_logging, lattice, dirac

logger = logging.getLogger(__name__)


def main():
    configure_logging(level=logging.INFO)

    logger.info("=" * 60)
    logger.info("Environment check")
    logger.info("=" * 60)
    logger.info("JAX version: %s", jax.__version__)
    logger.info("JAX devices: %s", jax.devices())
    x = jax.numpy.ones(3).block_until_ready()
    logger.info("Quick GPU/CPU op succeeded, device: %s", x.device)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Lattice + Wilson-Dirac operator")
    logger.info("=" * 60)

    key = random.PRNGKey(42)
    L, Lt = 6, 8
    theta = lattice.make_gauge_field(L, Lt, key)
    logger.info("Gauge field shape: %s  (expected (2, %d, %d))", theta.shape, Lt, L)

    beta = 5.0
    sg = lattice.gauge_action(theta, beta)
    plaq = lattice.average_plaquette(theta)
    q = lattice.topological_charge(theta)
    logger.info("Gauge action S_g = %.6f", float(sg))
    logger.info("Average plaquette = %.6f", float(plaq))
    logger.info("Topological charge = %.6f", float(q))

    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)
    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    logger.info("Wilson-Dirac matrix shape: %s", K.shape)
    logger.info("K dtype: %s", K.dtype)

    detK = jnp.linalg.det(K)
    logger.info("det(K) = %s", detK)
    logger.info("|Im(det(K))| = %.3e  (should be ~0 for mu=0)", float(abs(detK.imag)))

    # gamma5 hermiticity
    gamma5 = dirac.GAMMA_5
    Gamma5 = jnp.kron(jnp.eye(L * Lt, dtype=jnp.complex128), gamma5)
    g5Kg5 = Gamma5 @ K @ Gamma5
    diff_g5 = jnp.max(jnp.abs(g5Kg5 - jnp.conj(K.T)))
    logger.info("max |gamma5 K gamma5 - K^dagger| = %.3e  (should be ~0)", float(diff_g5))

    # blocks vs full matrix
    B, A_plus, A_minus = dirac.dirac_blocks(theta, mu=0.0, kappa=kappa)
    logger.info("B_t shape: %s", B.shape)
    logger.info("A_plus shape: %s", A_plus.shape)
    logger.info("A_minus shape: %s", A_minus.shape)

    P0_full = dirac.site_major_projector(dirac.P0, L)
    PM0_full = dirac.site_major_projector(dirac.PM0, L)
    I2 = jnp.eye(2, dtype=jnp.complex128)
    N = 2 * L * Lt
    K_from_blocks = jnp.zeros((N, N), dtype=jnp.complex128)
    for t in range(Lt):
        s = 2 * L * t
        K_from_blocks = K_from_blocks.at[s:s + 2 * L, s:s + 2 * L].set(B[t])
        tp1 = (t + 1) % Lt
        forward = -2.0 * kappa * P0_full @ jnp.kron(A_plus[t], I2)
        if t == Lt - 1:
            forward = -forward
        K_from_blocks = K_from_blocks.at[s:s + 2 * L, 2 * L * tp1:2 * L * tp1 + 2 * L].set(forward)
        tm1 = (t - 1) % Lt
        backward = -2.0 * kappa * PM0_full @ jnp.kron(A_minus[tm1], I2)
        if t == 0:
            backward = -backward
        K_from_blocks = K_from_blocks.at[s:s + 2 * L, 2 * L * tm1:2 * L * tm1 + 2 * L].set(backward)

    diff_blocks = jnp.max(jnp.abs(K - K_from_blocks))
    logger.info("max |K_direct - K_from_blocks| = %.3e  (should be ~0)", float(diff_blocks))

    logger.info("")
    logger.info("=" * 60)
    logger.info("All checks passed if the three numbers above are ~0.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
