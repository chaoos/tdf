# Benchmark

The benchmark compares the reference standard HMC against the TDF-based
canonical HMC on identical lattices and with identical random seeds.

## Running the benchmark

Use `scripts/run_benchmark.py`:

```bash
.venv/bin/python scripts/run_benchmark.py --L 4 --Lt 4 --beta 5.0 \
    --mass 0.0 --dt 0.1 --n-steps 5 --n-therm 20 --n-measure 50 --n-skip 3
```

The script runs three simulations:

1. Standard grand-canonical Nf=2 HMC.
2. Canonical HMC with sector `n = 0`.
3. Canonical HMC with sector `n = L // 2`.

It records for each run:

- wall-clock time per trajectory,
- acceptance rate per measurement block,
- average plaquette and its standard deviation,
- rough integrated autocorrelation time of the plaquette,
- average topological charge.

## Output files

- `benchmark_results.json` – JSON summary of the parameters and measured
  quantities.
- `benchmark_histories.npz` – NumPy archive containing the plaquette history
  of each run.  Load with `numpy.load`.

## Interpreting the results

The key metric is the **cost per independent sample**, defined as

```
cost = time_per_trajectory * n_skip * tau_int(P) .
```

This combines raw trajectory speed with the autocorrelation of the observable
being measured.  On the `4 × 4` lattice reported in [`RESULTS.md`](../RESULTS.md)
the canonical `n = 2` run is about ten times cheaper per independent plaquette
measurement than the standard run.

Keep in mind:

- The benchmark uses a very small volume and modest statistics.
- Acceptance rates are on the low side; production runs should tune `dt` and
  `n_steps` first.
- The canonical sectors are not the same ensemble as the grand-canonical
  standard run, so plaquette averages need not agree.
- The speed-up is expected to grow with `L_t` because the standard force scales
  with the full `L·Lt × L·Lt` determinant, whereas the canonical force uses the
  fixed `2L × 2L` transfer matrix.
