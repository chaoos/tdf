#!/usr/bin/env python3
"""Benchmark comparing standard and TDF canonical HMC."""

import argparse
import json
import logging
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
from jax import random

from tdf import configure_logging, lattice, dirac, hmc
from tdf.hmc_canonical import run_hmc_canonical

logger = logging.getLogger(__name__)


def integrated_autocorr_time(x, cutoff=0.05):
    """Rough integrated autocorrelation time for a 1-D series.

    Uses the normalized autocorrelation summed until it first drops below the
    cutoff or becomes negative.  Returns 0.5 if no positive lags are found.
    """
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
    """Run one HMC variant and collect timing, acceptance, and observables."""
    logger.info("Starting %s HMC", label)
    t0 = time.perf_counter()
    history, configs = run_fn(key, theta0, args)
    elapsed = time.perf_counter() - t0

    n_total = args.n_therm + args.n_measure * args.n_skip
    time_per_traj = elapsed / max(1, n_total)

    plaquette = history["plaquette"]
    q = history["topological_charge"]
    accept = history["accept"]

    result = {
        "label": label,
        "elapsed_sec": elapsed,
        "time_per_trajectory_sec": time_per_traj,
        "accept_rate_mean": float(jnp.mean(accept)),
        "accept_rate_std": float(jnp.std(accept)),
        "plaquette_mean": float(jnp.mean(plaquette)),
        "plaquette_std": float(jnp.std(plaquette)),
        "plaquette_tau_int": integrated_autocorr_time(plaquette),
        "topological_charge_mean": float(jnp.mean(q)),
        "topological_charge_std": float(jnp.std(q)),
    }
    logger.info(
        "%s: accept=%.3f +/- %.3f, <P>=%.5f +/- %.5f, tau_int(P)=%.2f, time/traj=%.4fs",
        label,
        result["accept_rate_mean"],
        result["accept_rate_std"],
        result["plaquette_mean"],
        result["plaquette_std"],
        result["plaquette_tau_int"],
        result["time_per_trajectory_sec"],
    )
    return result, history, configs


def main():
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compare standard and TDF canonical HMC")
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
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    kappa = dirac.kappa_from_mass(args.mass)
    L = args.L

    key = random.PRNGKey(args.seed)
    key_field, key_run = random.split(key)
    theta0 = lattice.make_gauge_field(L, args.Lt, key_field)

    logger.info("Benchmark: %dx%d, beta=%.3f, mass=%.3f, kappa=%.4f, mu=%.3f",
                L, args.Lt, args.beta, args.mass, kappa, args.mu)
    logger.info("Trajectory dt=%.3f, n_steps=%d", args.dt, args.n_steps)
    logger.info("Thermalization=%d, measurements=%d, skip=%d",
                args.n_therm, args.n_measure, args.n_skip)

    results = []
    histories = {}

    # Standard grand-canonical Nf=2 HMC.
    def run_standard(key_run, theta0, args):
        def action_fn(t):
            return hmc.action_standard(t, beta=args.beta, mu=args.mu, kappa=kappa)
        force_fn = hmc.make_force(action_fn)
        return hmc.run_hmc(key_run, theta0, action_fn, force_fn,
                           n_therm=args.n_therm, n_measure=args.n_measure,
                           n_skip=args.n_skip, dt=args.dt, n_steps=args.n_steps)

    res, hist, _ = run_single("standard", run_standard, key_run, theta0, args)
    results.append(res)
    histories["standard"] = hist

    # TDF canonical HMC for a few representative sectors.
    sectors = [0, L // 2]
    for n in sectors:
        def make_canonical_runner(n):
            def run_canonical(key_run, theta0, args):
                return run_hmc_canonical(
                    key_run, theta0, beta=args.beta, n=n, mu=args.mu, kappa=kappa,
                    n_therm=args.n_therm, n_measure=args.n_measure,
                    n_skip=args.n_skip, dt=args.dt, n_steps=args.n_steps,
                )
            return run_canonical

        res, hist, _ = run_single(f"canonical_n{n}", make_canonical_runner(n),
                                  key_run, theta0, args)
        results.append(res)
        histories[f"canonical_n{n}"] = hist

    summary = {
        "args": vars(args),
        "kappa": kappa,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", args.output)

    # Also save raw plaquette histories for plotting.
    jnp.savez(
        "benchmark_histories.npz",
        **{k: v["plaquette"] for k, v in histories.items()}
    )
    logger.info("Plaquette histories written to benchmark_histories.npz")


if __name__ == "__main__":
    main()
