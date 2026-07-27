# Benchmark

The repository contains two benchmarks.  The first (`scripts/run_benchmark.py`)
compares the exact-determinant standard and TDF canonical HMC samplers.  The
second (`scripts/run_pseudofermion_benchmark.py`) compares the stochastic
pseudofermion estimators described in [`pseudofermion.md`](pseudofermion.md).

This page documents the exact-determinant benchmark.  For the pseudofermion
benchmark see [`pseudofermion.md`](pseudofermion.md) and [`RESULTS.md`](../RESULTS.md).

## Running the benchmark

```bash
.venv/bin/python scripts/run_benchmark.py --L 4 --Lt 4 --beta 5.0 \
    --mass 0.0 --dt 0.1 --n-steps 5 --n-therm 20 --n-measure 50 --n-skip 3
```

The script performs three studies:

1. **Step-size sweep.**  Runs both algorithms at several values of `dt` with
   fixed `n_steps` and reports the acceptance rate.  This reveals how sensitive
   each algorithm is to the integrator step size.

2. **Detailed run.**  Runs both algorithms at the chosen `dt` and records:
   - acceptance rate,
   - mean, standard deviation, and maximum of `delta H`,
   - plaquette mean, standard deviation, and integrated autocorrelation time.

3. **Force scaling.**  Measures the wall-clock time of a single force evaluation
   for `Lt = 4, 6, 8` (with `L` fixed).  This exposes the asymptotic scaling of
   the two algorithms.

## Output files

- `benchmark_results.json` – JSON summary of all three studies.
- `benchmark_histories.npz` – plaquette and `delta H` histories for both
  algorithms.

## Interpreting the results

See [`RESULTS.md`](../RESULTS.md) for the benchmark output on a `4 × 4`
lattice.  The main points are:

- Acceptance and `delta H` are comparable at matched leapfrog parameters.
- Autocorrelation depends on volume and sector; it should be compared only
  after tuning acceptance to a common target.
- The canonical force has a larger constant overhead on small lattices because
  its gradient goes through a transfer-matrix eigenvalue decomposition.  The
  crossover where it becomes cheaper than the standard force occurs at
  moderately large lattice size.
