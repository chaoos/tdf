#!/usr/bin/env python3
"""Unified comparison of exact, TDF, and pseudofermion determinants.

For each lattice size this script computes:

1. the exact 2-flavour log determinant log |det K|^2 using jnp.linalg.slogdet,
2. the TDF log determinant using the reduced determinant,
3. the standard pseudofermion action distribution,
4. the TDF pseudofermion action distribution.

The pseudofermion actions are stochastic estimators of the fermion weight.
Their width relative to the exact log determinant measures the estimator noise.
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

    # Print a concise markdown table.
    print("\n| Lattice | Exact log|det K|² | TDF log|det K|² | TDF rel. diff. | "
          "Std. pf. <S> | Std. pf. σ | TDF pf. <S> | TDF pf. σ |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"{r['L']}×{r['Lt']} | "
            f"{r['exact_logdet']:.4f} | "
            f"{r['tdf_logdet']:.4f} | "
            f"{r['tdf_rel_diff']:.3e} | "
            f"{r['standard_pseudofermion']['mean_Spf']:.2f} | "
            f"{r['standard_pseudofermion']['std_Spf']:.2f} | "
            f"{r['tdf_pseudofermion']['mean_Spf']:.2f} | "
            f"{r['tdf_pseudofermion']['std_Spf']:.2f} |"
        )


if __name__ == "__main__":
    main()
