# Documentation index

This directory contains supplementary documentation for the temporal determinant
factorization (TDF) project.  For a concise overview and quick-start commands,
see the top-level [`README.md`](../README.md).

## Files

- [`algorithm.md`](algorithm.md) – mathematical background: lattice setup,
  Wilson–Dirac operator, time-slice factorisation, transfer matrices, and
  canonical determinants.
- [`hmc.md`](hmc.md) – how to run and tune the standard and canonical HMC
  samplers.
- [`pseudofermion.md`](pseudofermion.md) – stochastic HMC with pseudofermions,
  both in the full Dirac space and in the TDF transfer-matrix space.
- [`determinant_formulas.md`](determinant_formulas.md) – exact formulas for the
  exact, TDF, and pseudofermion determinant estimators used in the comparison.
- [`benchmark.md`](benchmark.md) – how to reproduce the benchmarks reported in
  [`RESULTS.md`](../RESULTS.md) and how to interpret the output.

## Directory layout

```
.
├── tdf/                      # core library
│   ├── lattice.py            # gauge-field utilities
│   ├── dirac.py              # Wilson–Dirac operator
│   ├── reduced.py            # TDF reduced determinant
│   ├── canonical.py          # canonical determinants
│   ├── hmc.py                # reference standard HMC
│   └── hmc_canonical.py      # TDF canonical HMC
├── scripts/                  # command-line drivers
│   ├── verify_setup.py       # sanity checks
│   ├── run_hmc_standard.py   # standard HMC driver
│   ├── run_hmc_canonical.py  # canonical HMC driver
│   └── run_benchmark.py      # comparison benchmark
├── tests/                    # pytest test suite
├── docs/                     # this documentation
├── README.md                 # top-level readme
└── RESULTS.md                # benchmark results
```
