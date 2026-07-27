#!/usr/bin/env python3
"""Verify Phase 0 (environment) and Phase 1 (lattice + Dirac operator)."""

import os

# Do not pre-allocate the entire GPU memory pool; the RTX 3050 has only 6 GB.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac


def main():
    print("=" * 60)
    print("Phase 0: environment check")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    x = jax.numpy.ones(3).block_until_ready()
    print(f"Quick GPU/CPU op succeeded, device: {x.device}")

    print()
    print("=" * 60)
    print("Phase 1: lattice + Wilson-Dirac operator")
    print("=" * 60)

    key = random.PRNGKey(42)
    L, Lt = 6, 8
    theta = lattice.make_gauge_field(L, Lt, key)
    print(f"Gauge field shape: {theta.shape}  (expected (2, {Lt}, {L}))")

    beta = 5.0
    sg = lattice.gauge_action(theta, beta)
    plaq = lattice.average_plaquette(theta)
    q = lattice.topological_charge(theta)
    print(f"Gauge action S_g = {sg:.6f}")
    print(f"Average plaquette = {plaq:.6f}")
    print(f"Topological charge = {q:.6f}")

    mass = 0.0
    kappa = dirac.kappa_from_mass(mass)
    K = dirac.wilson_dirac(theta, mu=0.0, kappa=kappa, boundary_phase=-1.0)
    print(f"Wilson-Dirac matrix shape: {K.shape}")
    print(f"K dtype: {K.dtype}")

    detK = jnp.linalg.det(K)
    print(f"det(K) = {detK}")
    print(f"|Im(det(K))| = {abs(detK.imag):.3e}  (should be ~0 for mu=0)")

    # gamma5 hermiticity
    gamma5 = dirac.GAMMA_5
    Gamma5 = jnp.kron(jnp.eye(L * Lt, dtype=jnp.complex128), gamma5)
    g5Kg5 = Gamma5 @ K @ Gamma5
    diff_g5 = jnp.max(jnp.abs(g5Kg5 - jnp.conj(K.T)))
    print(f"max |gamma5 K gamma5 - K^dagger| = {diff_g5:.3e}  (should be ~0)")

    # blocks vs full matrix
    B, A_plus, A_minus = dirac.dirac_blocks(theta, mu=0.0, kappa=kappa)
    print(f"B_t shape: {B.shape}")
    print(f"A_plus shape: {A_plus.shape}")
    print(f"A_minus shape: {A_minus.shape}")

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
    print(f"max |K_direct - K_from_blocks| = {diff_blocks:.3e}  (should be ~0)")

    print()
    print("=" * 60)
    print("All Phase 1 checks passed if the three numbers above are ~0.")
    print("=" * 60)


if __name__ == "__main__":
    main()
