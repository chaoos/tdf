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
    run_hmc_pseudofermion_tdf_block_diagonal,
    run_hmc_pseudofermion_tdf_bulk_product,
    run_hmc_pseudofermion_tdf_stochastic_bulk,
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


def _run_short_for_accept(run_fn, key, theta0, dt, n_steps, n_traj, args):
    """Run a short HMC and return the acceptance rate.

    Half of the trajectories are used for thermalisation and half for the
    acceptance estimate, so the measurement is more representative.
    """
    n_therm = n_traj // 2
    n_measure = n_traj - n_therm
    history, _ = run_fn(
        key, theta0, kappa=dirac.kappa_from_mass(args.mass),
        n_therm=n_therm, n_measure=n_measure, n_skip=1,
        dt=dt, n_steps=n_steps,
        tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
    )
    return float(jnp.mean(history["accept"]))


def tune_dt(label, run_fn, key, theta0, args, target_accept=0.7,
            dt_candidates=None, n_tune_traj=8):
    """Pick the dt that gives an acceptance rate closest to the target.

    A short HMC run is performed for each candidate dt and the one with the
    acceptance rate closest to ``target_accept`` is returned.  The default
    ``args.dt`` is also included as a candidate.
    """
    if dt_candidates is None:
        dt_candidates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]

    logger.info("Tuning dt for %s (target accept=%.2f)", label, target_accept)
    best_dt = args.dt
    best_err = abs(_run_short_for_accept(
        run_fn, key, theta0, args.dt, args.n_steps, n_tune_traj, args
    ) - target_accept)
    logger.info("  dt=%.4f -> accept=%.3f", args.dt, 1.0 - best_err)

    keys = random.split(key, len(dt_candidates))
    for dt, k in zip(dt_candidates, keys):
        accept = _run_short_for_accept(
            run_fn, k, theta0, dt, args.n_steps, n_tune_traj, args
        )
        err = abs(accept - target_accept)
        logger.info("  dt=%.4f -> accept=%.3f", dt, accept)
        if err < best_err:
            best_err = err
            best_dt = dt

    logger.info("Selected dt=%.4f for %s", best_dt, label)
    return best_dt


def run_single(label, run_fn, key, theta0, args, dt=None):
    """Run one pseudofermion HMC variant and collect diagnostics."""
    if dt is None:
        dt = args.dt
    logger.info("Starting %s pseudofermion HMC (dt=%.4f)", label, dt)
    t0 = time.perf_counter()
    history, configs = run_fn(
        key, theta0, kappa=dirac.kappa_from_mass(args.mass),
        n_therm=args.n_therm, n_measure=args.n_measure, n_skip=args.n_skip,
        dt=dt, n_steps=args.n_steps,
        tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    n_total = args.n_therm + args.n_measure * args.n_skip
    time_per_traj = elapsed / max(1, n_total)

    plaquette = history["plaquette"]
    q = history["topological_charge"]
    accept = history["accept"]
    delta_H = history["delta_H"]

    result = {
        "label": label,
        "dt": float(dt),
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
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Default leapfrog step size")
    parser.add_argument("--dt-std", type=float, default=None,
                        help="Leapfrog step size for the standard sampler (defaults to --dt)")
    parser.add_argument("--dt-tdf", type=float, default=None,
                        help="Leapfrog step size for the TDF exact-bulk sampler (defaults to --dt)")
    parser.add_argument("--dt-tdf-stoch", type=float, default=None,
                        help="Leapfrog step size for the TDF stochastic-bulk sampler (defaults to --dt)")
    parser.add_argument("--dt-tdf-bulk-product", type=float, default=None,
                        help="Leapfrog step size for the TDF bulk-product sampler (defaults to --dt)")
    parser.add_argument("--dt-tdf-block-diag", type=float, default=None,
                        help="Leapfrog step size for the TDF block-diagonal sampler (defaults to --dt)")
    parser.add_argument("--bulk-product-order", type=str, default="natural",
                        choices=["natural", "reverse", "balanced"],
                        help="Multiplication order for the bulk-product matrix")
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-therm", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=30)
    parser.add_argument("--n-skip", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--cg-maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-det-samples", type=int, default=50,
                        help="Number of pseudofermion fields for determinant estimate")
    parser.add_argument("--target-accept", type=float, default=0.7,
                        help="Target acceptance rate for dt tuning")
    parser.add_argument("--tune-dt", action="store_true",
                        help="Enable per-sampler dt tuning to the target acceptance rate")
    parser.add_argument("--n-tune-traj", type=int, default=8,
                        help="Number of trajectories per dt candidate during tuning")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging (prints CG residuals)")
    args = parser.parse_args()

    configure_logging(level=logging.INFO)
    if args.verbose:
        logging.getLogger("tdf").setLevel(logging.DEBUG)

    kappa = dirac.kappa_from_mass(args.mass)
    L, Lt = args.L, args.Lt

    key = random.PRNGKey(args.seed)
    key_field, key_std, key_tdf, key_tdf_stoch, key_tdf_bp, key_tdf_bd, key_det = random.split(key, 7)
    theta0 = lattice.make_gauge_field(L, Lt, key_field)

    logger.info("Pseudofermion benchmark: %dx%d, beta=%.3f, mass=%.3f, kappa=%.4f",
                L, Lt, args.beta, args.mass, kappa)
    logger.info("CG: tol=%.0e, maxiter=%s", args.tol, args.cg_maxiter)

    # HMC runs.
    def run_std(key, theta0, **kwargs):
        return run_hmc_pseudofermion_standard(
            key, theta0, kappa=kappa,
            n_therm=kwargs.get("n_therm", args.n_therm),
            n_measure=kwargs.get("n_measure", args.n_measure),
            n_skip=kwargs.get("n_skip", args.n_skip),
            dt=kwargs["dt"], n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
        )

    def run_tdf(key, theta0, **kwargs):
        return run_hmc_pseudofermion_tdf(
            key, theta0, kappa=kappa,
            n_therm=kwargs.get("n_therm", args.n_therm),
            n_measure=kwargs.get("n_measure", args.n_measure),
            n_skip=kwargs.get("n_skip", args.n_skip),
            dt=kwargs["dt"], n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
        )

    def run_tdf_stoch(key, theta0, **kwargs):
        return run_hmc_pseudofermion_tdf_stochastic_bulk(
            key, theta0, kappa=kappa,
            n_therm=kwargs.get("n_therm", args.n_therm),
            n_measure=kwargs.get("n_measure", args.n_measure),
            n_skip=kwargs.get("n_skip", args.n_skip),
            dt=kwargs["dt"], n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
        )

    def run_tdf_bulk_product(key, theta0, **kwargs):
        return run_hmc_pseudofermion_tdf_bulk_product(
            key, theta0, kappa=kappa,
            n_therm=kwargs.get("n_therm", args.n_therm),
            n_measure=kwargs.get("n_measure", args.n_measure),
            n_skip=kwargs.get("n_skip", args.n_skip),
            dt=kwargs["dt"], n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
            order=args.bulk_product_order,
        )

    def run_tdf_block_diagonal(key, theta0, **kwargs):
        return run_hmc_pseudofermion_tdf_block_diagonal(
            key, theta0, kappa=kappa,
            n_therm=kwargs.get("n_therm", args.n_therm),
            n_measure=kwargs.get("n_measure", args.n_measure),
            n_skip=kwargs.get("n_skip", args.n_skip),
            dt=kwargs["dt"], n_steps=args.n_steps,
            tol=args.tol, maxiter=args.cg_maxiter, verbose=args.verbose,
        )

    std_dt = args.dt if args.dt_std is None else args.dt_std
    tdf_dt = args.dt if args.dt_tdf is None else args.dt_tdf
    tdf_stoch_dt = args.dt if args.dt_tdf_stoch is None else args.dt_tdf_stoch
    tdf_bp_dt = args.dt if args.dt_tdf_bulk_product is None else args.dt_tdf_bulk_product
    tdf_bd_dt = args.dt if args.dt_tdf_block_diag is None else args.dt_tdf_block_diag

    if args.tune_dt:
        std_dt = tune_dt("standard", run_std, key_std, theta0, args,
                         target_accept=args.target_accept,
                         n_tune_traj=args.n_tune_traj)
        tdf_dt = tune_dt("tdf", run_tdf, key_tdf, theta0, args,
                         target_accept=args.target_accept,
                         n_tune_traj=args.n_tune_traj)
        tdf_stoch_dt = tune_dt("tdf-stoch", run_tdf_stoch,
                               key_tdf_stoch, theta0, args,
                               target_accept=args.target_accept,
                               n_tune_traj=args.n_tune_traj)
        tdf_bp_dt = tune_dt("tdf-bulk-product", run_tdf_bulk_product,
                            key_tdf_bp, theta0, args,
                            target_accept=args.target_accept,
                            n_tune_traj=args.n_tune_traj)
        tdf_bd_dt = tune_dt("tdf-block-diag", run_tdf_block_diagonal,
                            key_tdf_bd, theta0, args,
                            target_accept=args.target_accept,
                            n_tune_traj=args.n_tune_traj)

    std_result, std_hist = run_single("standard", run_std, key_std, theta0, args, dt=std_dt)
    tdf_result, tdf_hist = run_single("tdf", run_tdf, key_tdf, theta0, args, dt=tdf_dt)
    tdf_stoch_result, tdf_stoch_hist = run_single(
        "tdf-stoch", run_tdf_stoch, key_tdf_stoch, theta0, args, dt=tdf_stoch_dt
    )
    tdf_bp_result, tdf_bp_hist = run_single(
        "tdf-bulk-product", run_tdf_bulk_product, key_tdf_bp, theta0, args, dt=tdf_bp_dt
    )
    tdf_bd_result, tdf_bd_hist = run_single(
        "tdf-block-diag", run_tdf_block_diagonal, key_tdf_bd, theta0, args, dt=tdf_bd_dt
    )

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
        "tdf_stochastic_bulk": tdf_stoch_result,
        "tdf_bulk_product": tdf_bp_result,
        "tdf_block_diagonal": tdf_bd_result,
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
        tdf_stoch_plaquette=tdf_stoch_hist["plaquette"],
        tdf_stoch_delta_H=tdf_stoch_hist["delta_H"],
        tdf_bp_plaquette=tdf_bp_hist["plaquette"],
        tdf_bp_delta_H=tdf_bp_hist["delta_H"],
        tdf_bd_plaquette=tdf_bd_hist["plaquette"],
        tdf_bd_delta_H=tdf_bd_hist["delta_H"],
    )
    logger.info("Histories written to %s_histories.npz", out_base)


if __name__ == "__main__":
    main()
