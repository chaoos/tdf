#!/usr/bin/env python3
"""Unified comparison of exact, TDF, and pseudofermion determinants.

For each lattice size this script computes three estimates of the 2-flavour
weight |det K|^2:

1. the exact determinant from the full Wilson-Dirac matrix,
2. the standard pseudofermion estimate <|det K|^2> = <exp(-S_pf)> with its
   standard error of the mean,
3. the TDF pseudofermion estimate <|det K|^2> = <exp(-S_pf)> with its standard
   error of the mean.

The TDF reduced determinant is also reported as a consistency check.

A modest number of pseudofermion samples is used so that the error bars are
honest: if the estimate is unbiased, the discrepancy between the pseudofermion
mean and the exact value should be no larger than one standard error of the
mean about 68% of the time.  The script flags each estimate accordingly.
"""

import argparse
import json
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
from jax import random

from tdf import configure_logging
from tdf.determinant_comparison import compare_for_size

logger = logging.getLogger(__name__)


def main():
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Compare exact, TDF, and pseudofermion determinants"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8],
                        help="Spatial and temporal extents to compare")
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of pseudofermion samples per lattice size")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--cg-maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="determinant_comparison.json")
    args = parser.parse_args()

    logger.info("Unified determinant comparison")
    logger.info("beta=%.3f, mass=%.3f, n_samples=%d, tol=%.0e",
                args.beta, args.mass, args.n_samples, args.tol)

    results = []
    key = random.PRNGKey(args.seed)
    for size in args.sizes:
        key = random.fold_in(key, size)
        result = compare_for_size(
            size, size, args.beta, args.mass, args.n_samples,
            args.tol, args.cg_maxiter, key
        )
        results.append(result)

    summary = {
        "args": vars(args),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", args.output)

    # Print concise markdown tables.
    print("\n| Lattice | Exact |det K|² | TDF |det K|² | TDF rel. diff. |")
    print("|---:|---:|---:|---:|")
    for r in results:
        print(
            f"{r['L']}×{r['Lt']} | "
            f"{r['exact_det']:.4e} | "
            f"{jnp.exp(r['tdf_logdet']):.4e} | "
            f"{r['tdf_rel_diff']:.3e} |"
        )

    print("\n| Lattice | Exact |det K|² | Std. pf. estimate ± σ_mean | "
          "TDF pf. estimate ± σ_mean | Std. within 1σ? | TDF within 1σ? |")
    print("|---:|---:|---:|---:|:--:|:--:|")
    for r in results:
        std_mean = r["standard"]["mean_det"]
        std_sem = r["standard"]["sem_det"]
        tdf_mean = r["tdf"]["mean_det"]
        tdf_sem = r["tdf"]["sem_det"]
        print(
            f"{r['L']}×{r['Lt']} | "
            f"{r['exact_det']:.4e} | "
            f"{std_mean:.4e} ± {std_sem:.4e} | "
            f"{tdf_mean:.4e} ± {tdf_sem:.4e} | "
            f"{'yes' if r['standard_agrees_1sigma'] else 'no'} | "
            f"{'yes' if r['tdf_agrees_1sigma'] else 'no'} |"
        )

    print("\n| Lattice | Std. pf. <S> | Std. pf. σ(S) | "
          "TDF pf. <S> | TDF pf. σ(S) |")
    print("|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"{r['L']}×{r['Lt']} | "
            f"{r['standard_action']['mean_Spf']:.2f} | "
            f"{r['standard_action']['std_Spf']:.2f} | "
            f"{r['tdf_action']['mean_Spf']:.2f} | "
            f"{r['tdf_action']['std_Spf']:.2f} |"
        )


if __name__ == "__main__":
    main()
