#!/usr/bin/env python3
"""Run the reference standard HMC for the 2-flavour Schwinger model."""

import argparse
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
from jax import random

from tdf import configure_logging, lattice, dirac, hmc

logger = logging.getLogger(__name__)


def main():
    configure_logging(level=logging.INFO)

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

    logger.info("Running standard Nf=2 HMC on %dx%d lattice", args.L, args.Lt)
    logger.info("beta=%.4f, mass=%.4f, kappa=%.4f, mu=%.4f",
                args.beta, args.mass, kappa, args.mu)
    logger.info("Trajectory: dt=%.4f, n_steps=%d", args.dt, args.n_steps)
    logger.info("Thermalization=%d, measurements=%d, skip=%d",
                args.n_therm, args.n_measure, args.n_skip)

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

    logger.info("Results:")
    logger.info("  acceptance rate = %.3f", float(jnp.mean(accept)))
    logger.info("  plaquette       = %.6f +/- %.6f",
                float(jnp.mean(plaquette)), float(jnp.std(plaquette)))
    logger.info("  topo charge     = %.3f +/- %.3f",
                float(jnp.mean(q)), float(jnp.std(q)))

    jnp.savez(
        "hmc_standard_summary.npz",
        plaquette=plaquette,
        topological_charge=q,
        accept=accept,
    )
    logger.info("Summary saved to hmc_standard_summary.npz")


if __name__ == "__main__":
    main()
