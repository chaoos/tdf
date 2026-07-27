#!/usr/bin/env python3
"""Algorithmic benchmark of standard vs. TDF pseudofermion HMC."""

import argparse
import json
import logging
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import random

from tdf import configure_logging, lattice, dirac
from tdf.hmc_pseudofermion import (
    run_hmc_pseudofermion_standard,
    run_hmc_pseudofermion_tdf,
)
from tdf.pseudofermion import estimate_pseudofermion_action_distribution

logger = logging.getLogger(__name__)


def integrated_autocorr_time(x, cutoff=0.05):
    """Rough integrated autocorrelation time for a 1-D series."""
    x = jnp.asarray(x)
    x = x - jnp.mean(x)
    n = x.shape[0]
    if n < 2:
        return 0.5
    c0 = jnp.mean(x ** 2)
    if c0 == 0.0:
        return 0.5
    tau = 0.5
    for lag in range(1, n):
        c = jnp.mean(x[:-lag] * x[lag:])
        rho = c / c0
        if rho < cutoff or rho < 0.0:
            break
        tau += rho
    return float(tau)


def run_single(label, run_fn, key, theta0, args):
    """Run one pseudofermion HMC variant and collect diagnostics."""
    logger.info("Starting %s pseudofermion HMC", label)
    t0 = time.perf_counter()
    history, configs = run_fn(key, theta0, args)
    elapsed = time.perf_counter() - t0

    n_total = args.n_therm + args.n_measure * args.n_skip
    time_per_traj = elapsed / max(1, n_total)

    plaquette = history["plaquette"]
    q = history["topological_charge"]
    accept = history["accept"]
    delta_H = history["delta_H"]

    result = {
        "label": label,
        "elapsed_sec": elapsed,
        "time_per_trajectory_sec": time_per_traj,
        "accept_rate": float(jnp.mean(accept)),
        "delta_H_mean": float(jnp.mean(delta_H)),
        "delta_H_std": float(jnp.std(delta_H)),
        "delta_H_max": float(jnp.max(jnp.abs(delta_H))),
        "plaquette_mean": float(jnp.mean(plaquette)),
        "plaquette_std": float(jnp.std(plaquette)),
        "plaquette_tau_int": integrated_autocorr_time(plaquette),
        "topological_charge_mean": float(jnp.mean(q)),
        "topological_charge_std": float(jnp.std(q)),
    }
    logger.info(
        "%s: accept=%.3f, <|dH|>=%.3f, max|dH|=%.3f, tau_int(P)=%.2f, time/traj=%.4fs",
        label,
        result["accept_rate"],
        result["delta_H_std"],
        result["delta_H_max"],
        result["plaquette_tau_int"],
        result["time_per_trajectory_sec"],
    )
    return result, history


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark standard vs. TDF pseudofermion HMC"
    )
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--Lt", type=int, default=4)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-therm", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=30)
    parser.add_argument("--n-skip", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--cg-maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-det-samples", type=int, default=50,
                        help="Number of pseudofermion fields for determinant estimate")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging (prints CG residuals)")
    args = parser.parse_args()

    configure_logging(level=logging.INFO)
    if args.verbose:
        logging.getLogger("tdf").setLevel(logging.DEBUG)

    kappa = dirac.kappa_from_mass(args.mass)
    L, Lt = args.L, args.Lt

    key = random.PRNGKey(args.seed)
    key_field, key_std, key_tdf, key_det = random.split(key, 4)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)

    logger.info("Pseudofermion benchmark: %dx%d, beta=%.3f, mass=%.3f, kappa=%.4f",
                L, Lt, args.beta, args.mass, kappa)
    logger.info("CG: tol=%.0e, maxiter=%s", args.tol, args.cg_maxiter)

    # HMC runs.
    def run_std(key, theta0, args):
        return run_hmc_pseudofermion_standard(
            key, theta0, kappa=kappa,
            n_therm=args.n_therm, n_measure=args.n_measure, n_skip=args.n_skip,
            dt=args.dt, n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
        )

    def run_tdf(key, theta0, args):
        return run_hmc_pseudofermion_tdf(
            key, theta0, kappa=kappa,
            n_therm=args.n_therm, n_measure=args.n_measure, n_skip=args.n_skip,
            dt=args.dt, n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
        )

    std_result, std_hist = run_single("standard", run_std, key_std, theta0, args)
    tdf_result, tdf_hist = run_single("tdf", run_tdf, key_tdf, theta0, args)

    # Determinant approximation quality.
    logger.info("Sampling pseudofermion action distribution (%d samples)", args.n_det_samples)
    std_det = estimate_pseudofermion_action_distribution(
        theta0, kappa, n_samples=args.n_det_samples, tol=args.tol,
        maxiter=args.cg_maxiter, algorithm="standard", key=key_det
    )
    tdf_det = estimate_pseudofermion_action_distribution(
        theta0, kappa, n_samples=args.n_det_samples, tol=args.tol,
        maxiter=args.cg_maxiter, algorithm="tdf", key=key_det
    )
    logger.info("Standard: exact logdet=%.6f, <S_pf>=%.3f, std(S_pf)=%.3f, noise=%.3f",
                std_det["exact_logdet"], std_det["mean_Spf"],
                std_det["std_Spf"], std_det["noise_Spf"])
    logger.info("TDF:      exact logdet=%.6f, <S_pf>=%.3f, std(S_pf)=%.3f, noise=%.3f",
                tdf_det["exact_logdet"], tdf_det["mean_Spf"],
                tdf_det["std_Spf"], tdf_det["noise_Spf"])

    summary = {
        "args": vars(args),
        "kappa": kappa,
        "standard": std_result,
        "tdf": tdf_result,
        "determinant_estimate": {
            "standard": std_det,
            "tdf": tdf_det,
        },
    }

    out_base = f"pseudofermion_benchmark_L{L}_Lt{Lt}"
    with open(f"{out_base}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s_results.json", out_base)

    jnp.savez(
        f"{out_base}_histories.npz",
        standard_plaquette=std_hist["plaquette"],
        standard_delta_H=std_hist["delta_H"],
        tdf_plaquette=tdf_hist["plaquette"],
        tdf_delta_H=tdf_hist["delta_H"],
    )
    logger.info("Histories written to %s_histories.npz", out_base)


if __name__ == "__main__":
    main()
