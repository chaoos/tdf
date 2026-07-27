# Temporal determinant factorization for the 2D Schwinger model

This repository contains a Python/JAX implementation of the temporal determinant
factorization (TDF) for the two-flavour Schwinger model in two dimensions,
alongside a reference Hybrid Monte Carlo (HMC) sampler that uses the full
Wilson–Dirac determinant.

The TDF approach factorises the fermion determinant into time-slice transfer
matrices and expresses fixed-quark-number (canonical) determinants in terms of
the transfer-matrix spectrum.  This can be substantially cheaper than the
conventional full-determinant formulation, especially for lattices with a large
temporal extent.

## Quick start

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project uses JAX with CUDA 12 support.  GPU memory pre-allocation is
disabled by default in the scripts and in `tests/conftest.py` because the
development GPU has only 6 GB of VRAM.

## Running the code

```bash
# Run the test suite
.venv/bin/pytest tests/ -v

# Verify the environment, lattice utilities, and Dirac operator
.venv/bin/python scripts/verify_setup.py

# Reference standard Nf=2 HMC
.venv/bin/python scripts/run_hmc_standard.py --L 4 --Lt 4 --beta 3.0 \
    --n-therm 10 --n-measure 10 --n-skip 2 --n-steps 5

# TDF canonical HMC in a fixed sector n
.venv/bin/python scripts/run_hmc_canonical.py --L 4 --Lt 4 --n 2 --beta 3.0 \
    --n-therm 5 --n-measure 5 --n-skip 2 --n-steps 5

# Benchmark comparing standard and canonical HMC
.venv/bin/python scripts/run_benchmark.py --L 4 --Lt 4 --beta 5.0 \
    --mass 0.0 --dt 0.1 --n-steps 5 --n-therm 20 --n-measure 50 --n-skip 3
```

## Package overview

| Module | Purpose |
|--------|---------|
| `tdf/lattice.py` | U(1) gauge fields, plaquette, gauge action, topological charge |
| `tdf/dirac.py` | Wilson–Dirac operator `K[U, μ]` and time-slice blocks |
| `tdf/reduced.py` | Time-slice transfer matrices and reduced determinant |
| `tdf/canonical.py` | Fixed-quark-number determinants from the transfer-matrix spectrum |
| `tdf/hmc.py` | Reference standard HMC sampler |
| `tdf/hmc_canonical.py` | TDF-based canonical HMC sampler |

## Documentation

More detailed documentation lives in the [`docs/`](docs/) directory:

- [`docs/index.md`](docs/index.md) – project overview and directory guide
- [`docs/algorithm.md`](docs/algorithm.md) – the TDF and canonical-determinant formalism
- [`docs/hmc.md`](docs/hmc.md) – using the HMC samplers
- [`docs/benchmark.md`](docs/benchmark.md) – reproducing and interpreting the benchmark

## Results

See [`RESULTS.md`](RESULTS.md) for a head-to-head comparison of the standard
and TDF canonical samplers on a `4 × 4` lattice.

## Logging

All modules and scripts use the standard Python `logging` library.  Scripts
configure logging via `tdf.configure_logging()`.  Use `logging.DEBUG` for
verbose trajectory output, or `logging.INFO` (default) for concise progress
messages.
