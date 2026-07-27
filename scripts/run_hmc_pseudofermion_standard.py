#!/usr/bin/env python3
"""Run standard pseudofermion HMC for the 2-flavour Schwinger model."""

import argparse
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
from jax import random

from tdf import configure_logging, lattice, dirac
from tdf.hmc_pseudofermion import run_hmc_pseudofermion_standard

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Standard pseudofermion HMC for the 2-flavour Schwinger model"
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--Lt", type=int, default=6)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--n-skip", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--cg-maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging (prints CG residuals)")
    args = parser.parse_args()

    configure_logging(level=logging.INFO)
    if args.verbose:
        logging.getLogger("tdf").setLevel(logging.DEBUG)

    kappa = dirac.kappa_from_mass(args.mass)

    key = random.PRNGKey(args.seed)
    key_field, key_run = random.split(key)
    theta = lattice.make_gauge_field(args.L, args.Lt, key_field)

    logger.info("Running standard pseudofermion HMC on %dx%d lattice", args.L, args.Lt)
    logger.info("beta=%.4f, mass=%.4f, kappa=%.4f", args.beta, args.mass, kappa)
    logger.info("Trajectory: dt=%.4f, n_steps=%d", args.dt, args.n_steps)
    logger.info("CG: tol=%.0e, maxiter=%s", args.tol, args.cg_maxiter)
    logger.info("Thermalization=%d, measurements=%d, skip=%d",
                args.n_therm, args.n_measure, args.n_skip)

    history, configs = run_hmc_pseudofermion_standard(
        key_run, theta, kappa=kappa,
        n_therm=args.n_therm, n_measure=args.n_measure, n_skip=args.n_skip,
        dt=args.dt, n_steps=args.n_steps,
        tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
    )

    plaquette = history["plaquette"]
    q = history["topological_charge"]
    accept = history["accept"]
    delta_H = history["delta_H"]

    logger.info("Results:")
    logger.info("  acceptance rate = %.3f", float(jnp.mean(accept)))
    logger.info("  <|delta H|>     = %.3f", float(jnp.mean(jnp.abs(delta_H))))
    logger.info("  max |delta H|   = %.3f", float(jnp.max(jnp.abs(delta_H))))
    logger.info("  plaquette       = %.6f +/- %.6f",
                float(jnp.mean(plaquette)), float(jnp.std(plaquette)))
    logger.info("  topo charge     = %.3f +/- %.3f",
                float(jnp.mean(q)), float(jnp.std(q)))

    jnp.savez(
        "hmc_pseudofermion_standard_summary.npz",
        plaquette=plaquette,
        topological_charge=q,
        accept=accept,
        delta_H=delta_H,
    )
    logger.info("Summary saved to hmc_pseudofermion_standard_summary.npz")


if __name__ == "__main__":
    main()
