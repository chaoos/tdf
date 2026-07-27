#!/usr/bin/env python3
"""Run the reference standard HMC for the 2-flavour Schwinger model."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse

import jax.numpy as jnp
from jax import random

from tdf import lattice, dirac, hmc


def main():
    parser = argparse.ArgumentParser(description="Standard Nf=2 HMC")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--Lt", type=int, default=6)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--n-skip", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kappa = dirac.kappa_from_mass(args.mass)

    key = random.PRNGKey(args.seed)
    key_field, key_run = random.split(key)
    theta = lattice.make_gauge_field(args.L, args.Lt, key_field)

    def action_fn(t):
        return hmc.action_standard(t, beta=args.beta, mu=args.mu, kappa=kappa)

    force_fn = hmc.make_force(action_fn)

    print(f"Running standard Nf=2 HMC on {args.L}x{args.Lt} lattice")
    print(f"beta={args.beta}, mass={args.mass}, kappa={kappa:.4f}, mu={args.mu}")
    print(f"trajectory: dt={args.dt}, n_steps={args.n_steps}")
    print(f"thermalization={args.n_therm}, measurements={args.n_measure}, skip={args.n_skip}")

    history, configs = hmc.run_hmc(
        key_run, theta, action_fn, force_fn,
        n_therm=args.n_therm,
        n_measure=args.n_measure,
        n_skip=args.n_skip,
        dt=args.dt,
        n_steps=args.n_steps,
    )

    plaquette = history["plaquette"]
    q = history["topological_charge"]
    accept = history["accept"]

    print("\nResults:")
    print(f"  acceptance rate = {float(jnp.mean(accept)):.3f}")
    print(f"  plaquette       = {float(jnp.mean(plaquette)):.6f} +/- {float(jnp.std(plaquette)):.6f}")
    print(f"  topo charge     = {float(jnp.mean(q)):.3f} +/- {float(jnp.std(q)):.3f}")

    # Save a simple summary.
    jnp.savez(
        "hmc_standard_summary.npz",
        plaquette=plaquette,
        topological_charge=q,
        accept=accept,
    )
    print("\nSummary saved to hmc_standard_summary.npz")


if __name__ == "__main__":
    main()
