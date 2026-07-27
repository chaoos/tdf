#!/usr/bin/env python3
"""Algorithmic benchmark of standard vs. TDF canonical HMC.

This script compares the two algorithms directly, i.e. it measures properties
that are independent of the particular implementation or hardware:

- acceptance rate as a function of step size,
- Hamiltonian energy violation (delta H) distribution,
- integrated autocorrelation time of the plaquette,
- cost scaling of a single force evaluation with the temporal extent Lt.

Wall-clock timings are only reported in the scaling section, where they are
used to extract the asymptotic behaviour of the force evaluation.
"""

import argparse
import json
import logging
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import random

from tdf import configure_logging, lattice, dirac, hmc
from tdf.hmc_canonical import action_canonical

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


def run_algorithm(label, theta0, action_fn, force_fn, key, n_therm, n_measure,
                  n_skip, dt, n_steps, observables=None):
    """Run HMC and record acceptance, delta H, and observables."""
    if observables is None:
        observables = {
            "plaquette": lattice.average_plaquette,
            "topological_charge": lattice.topological_charge,
        }

    theta = theta0
    n_total = n_therm + n_measure * n_skip
    keys = random.split(key, n_total)

    for i in range(n_therm):
        theta, _ = hmc.hmc_step(keys[i], theta, action_fn, force_fn, dt, n_steps)

    history = {name: [] for name in observables.keys()}
    history["accept"] = []
    history["delta_H"] = []

    key_idx = n_therm
    for _ in range(n_measure):
        accepted_count = 0
        delta_H_block = []
        for _ in range(n_skip):
            theta, accepted, delta_H = hmc.hmc_step_diagnostics(
                keys[key_idx], theta, action_fn, force_fn, dt, n_steps
            )
            accepted_count += int(accepted)
            delta_H_block.append(float(delta_H))
            key_idx += 1

        for name, obs_fn in observables.items():
            history[name].append(float(obs_fn(theta)))
        history["accept"].append(accepted_count / n_skip)
        history["delta_H"].append(float(jnp.mean(jnp.array(delta_H_block))))

    history = {k: jnp.array(v) for k, v in history.items()}
    return history


def acceptance_sweep(theta0, action_fn, force_fn, key, dt_values, n_steps,
                     n_therm=5, n_traj=20):
    """Measure acceptance rate for a list of step sizes."""
    rates = []
    for dt in dt_values:
        keys = random.split(key, n_therm + n_traj)
        key = random.fold_in(key, int(dt * 1000))
        theta = theta0
        for i in range(n_therm):
            theta, _ = hmc.hmc_step(keys[i], theta, action_fn, force_fn, dt, n_steps)
        accepted = 0
        for i in range(n_traj):
            theta, acc = hmc.hmc_step(keys[n_therm + i], theta, action_fn, force_fn, dt, n_steps)
            accepted += int(acc)
        rates.append(accepted / n_traj)
    return [float(r) for r in rates]


def measure_force_cost(L, Lt, kappa, mu, key, algorithm="standard", n=0,
                       n_warm=3, n_reps=10):
    """Time a single force evaluation for a given lattice and algorithm."""
    theta = lattice.make_gauge_field(L, Lt, key)
    beta = 5.0  # value does not affect force timing

    if algorithm == "standard":
        def action_fn(t):
            return hmc.action_standard(t, beta=beta, mu=mu, kappa=kappa)
    else:
        def action_fn(t):
            return action_canonical(t, beta=beta, n=n, mu=mu, kappa=kappa)

    force_fn = hmc.make_force(action_fn)

    # Warm-up JIT compilation.
    for _ in range(n_warm):
        _ = force_fn(theta)
    jax.block_until_ready(force_fn(theta))

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        f = force_fn(theta)
        jax.block_until_ready(f)
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": float(jnp.mean(jnp.array(times)) * 1000),
        "std_ms": float(jnp.std(jnp.array(times)) * 1000),
    }


def main():
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Algorithmic benchmark of standard and TDF canonical HMC"
    )
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--Lt", type=int, default=4)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-therm", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=50)
    parser.add_argument("--n-skip", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    kappa = dirac.kappa_from_mass(args.mass)
    L, Lt = args.L, args.Lt

    key = random.PRNGKey(args.seed)
    key_field, key_sweep, key_run, key_scaling = random.split(key, 4)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)

    logger.info("Algorithmic benchmark: %dx%d, beta=%.3f, mass=%.3f, kappa=%.4f, mu=%.3f",
                L, Lt, args.beta, args.mass, kappa, args.mu)

    # ------------------------------------------------------------------
    # 1. Acceptance-rate vs step-size sweep
    # ------------------------------------------------------------------
    dt_values = [0.05, 0.075, 0.10, 0.125, 0.15]

    def std_action(t):
        return hmc.action_standard(t, beta=args.beta, mu=args.mu, kappa=kappa)
    std_force = hmc.make_force(std_action)

    logger.info("Step-size sweep for standard HMC")
    std_accept_sweep = acceptance_sweep(
        theta0, std_action, std_force, key_sweep, dt_values, args.n_steps
    )

    def can_action(t):
        return action_canonical(t, beta=args.beta, n=L // 2, mu=args.mu, kappa=kappa)
    can_force = hmc.make_force(can_action)

    logger.info("Step-size sweep for canonical HMC (n=%d)", L // 2)
    can_accept_sweep = acceptance_sweep(
        theta0, can_action, can_force, key_sweep, dt_values, args.n_steps
    )

    # ------------------------------------------------------------------
    # 2. Detailed run at the chosen step size: delta H and autocorrelation
    # ------------------------------------------------------------------
    logger.info("Detailed run at dt=%.3f, n_steps=%d", args.dt, args.n_steps)
    std_hist = run_algorithm(
        "standard", theta0, std_action, std_force, key_run,
        args.n_therm, args.n_measure, args.n_skip, args.dt, args.n_steps
    )
    can_hist = run_algorithm(
        "canonical", theta0, can_action, can_force, key_run,
        args.n_therm, args.n_measure, args.n_skip, args.dt, args.n_steps
    )

    def summarise(history):
        return {
            "accept_rate": float(jnp.mean(history["accept"])),
            "delta_H_mean": float(jnp.mean(history["delta_H"])),
            "delta_H_std": float(jnp.std(history["delta_H"])),
            "delta_H_max": float(jnp.max(jnp.abs(history["delta_H"]))),
            "plaquette_mean": float(jnp.mean(history["plaquette"])),
            "plaquette_std": float(jnp.std(history["plaquette"])),
            "plaquette_tau_int": integrated_autocorr_time(history["plaquette"]),
        }

    std_summary = summarise(std_hist)
    can_summary = summarise(can_hist)

    logger.info("Standard:   accept=%.3f, <|dH|>=%.3f, max|dH|=%.3f, tau_int(P)=%.2f",
                std_summary["accept_rate"],
                std_summary["delta_H_std"],
                std_summary["delta_H_max"],
                std_summary["plaquette_tau_int"])
    logger.info("Canonical:  accept=%.3f, <|dH|>=%.3f, max|dH|=%.3f, tau_int(P)=%.2f",
                can_summary["accept_rate"],
                can_summary["delta_H_std"],
                can_summary["delta_H_max"],
                can_summary["plaquette_tau_int"])

    # ------------------------------------------------------------------
    # 3. Force-evaluation scaling with Lt
    # ------------------------------------------------------------------
    scaling_Lt = [4, 6, 8]
    std_scaling = []
    can_scaling = []
    for lt in scaling_Lt:
        logger.info("Force scaling study, Lt=%d", lt)
        std_scaling.append({
            "Lt": lt,
            **measure_force_cost(L, lt, kappa, args.mu, key_scaling,
                                 algorithm="standard")
        })
        can_scaling.append({
            "Lt": lt,
            **measure_force_cost(L, lt, kappa, args.mu, key_scaling,
                                 algorithm="canonical", n=L // 2)
        })
        key_scaling = random.fold_in(key_scaling, lt)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "args": vars(args),
        "kappa": kappa,
        "acceptance_sweep": {
            "dt_values": dt_values,
            "standard": std_accept_sweep,
            "canonical": can_accept_sweep,
        },
        "detailed_run": {
            "standard": std_summary,
            "canonical": can_summary,
        },
        "force_scaling": {
            "standard": std_scaling,
            "canonical": can_scaling,
        },
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", args.output)

    jnp.savez(
        "benchmark_histories.npz",
        standard_plaquette=std_hist["plaquette"],
        standard_delta_H=std_hist["delta_H"],
        canonical_plaquette=can_hist["plaquette"],
        canonical_delta_H=can_hist["delta_H"],
    )
    logger.info("Histories written to benchmark_histories.npz")


if __name__ == "__main__":
    main()
