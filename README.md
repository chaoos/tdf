# Temporal determinant factorization

Proof-of-concept implementation of the temporal determinant factorization (TDF)
for the 2-flavour Schwinger model in two dimensions, together with a reference
standard HMC sampler.  The goal is to compare the TDF approach against a
conventional HMC that uses the full Wilson–Dirac determinant directly.

## Environment

The code uses **Python + JAX** with GPU support.  A local virtual environment is
recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs JAX with CUDA 12 support.  The RTX 3050 in this environment has
only 6 GB of VRAM, so GPU memory pre-allocation is disabled via
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (set in `tests/conftest.py` and in the
verification scripts).

## Implemented so far

### Phase 0 – Setup
- `pyproject.toml`, `requirements.txt`
- JAX x64 precision enabled in `tdf/__init__.py`

### Phase 1 – Lattice and Wilson–Dirac operator
- `tdf/lattice.py`: U(1) gauge fields, plaquette, gauge action, topological charge.
- `tdf/dirac.py`: Wilson–Dirac operator `K[U, μ]` in site-major ordering and the
  time-slice blocks `B_t`, `A^+_t`, `A^-_t`.
- Validation tests:
  - `γ5 K γ5 = K†` for `μ = 0`
  - `det K` is real for `μ = 0`
  - Direct construction matches block reconstruction

### Phase 3 – Reduced determinant (TDF)
- `tdf/reduced.py`: time-slice transfer matrices `T_i = R_i^{-1} S_i`, the cyclic
  product `T = T_0 T_1 … T_{L_t-1}`, and the reduced determinant

```
det K = (∏_t det R_t) * det(I - (-1)^{L_t} T).
```

- Validation tests confirm the reduced determinant equals the full determinant
  for periodic and anti-periodic boundary conditions, even and odd `L_t`, and
  with non-zero mass and chemical potential.

### Phase 4 – Canonical determinants
- `tdf/canonical.py`: fixed-quark-number determinants `det_k(K[U])` for
  `k = -L, …, L` from the characteristic polynomial of the transfer matrix.

```
det_k(K[U]) = (∏_t det R_t) * c_{k+L},
```

where `c_m` are the coefficients of `det(z I - (-1)^{L_t} T)`.

- Validation tests:
  - reflection symmetry `det_k^* = det_{-k}`
  - sum rule `Σ_k det_k = det K[U, μ=0]`
  - `det_0` is real

### Phase 5 – Reference standard HMC
- `tdf/hmc.py`: reference Hybrid Monte Carlo sampler for the 2-flavour model
  using the full Wilson–Dirac determinant.

```
S[U] = S_g[U] - 2 log |det K[U, μ]|
```

- Leapfrog integration with automatic differentiation for the fermion force.
- `scripts/run_hmc_standard.py`: command-line driver.
- Validation tests:
  - action is real and finite
  - standard action matches TDF-based action
  - force has correct shape
  - quenched HMC has near-perfect acceptance for tiny steps
  - full `run_hmc` pipeline executes

## Running tests and verification

```bash
# all tests
.venv/bin/pytest tests/ -v

# Phase 1 verification
.venv/bin/python scripts/verify_phase1.py

# Reference standard HMC (small example)
.venv/bin/python scripts/run_hmc_standard.py --L 4 --Lt 4 --beta 3.0 \
    --n-therm 10 --n-measure 10 --n-skip 2 --n-steps 5
```

## Roadmap

- Phase 4: canonical determinants `det_k(K)` via the characteristic polynomial of `T`
- Phase 5: reference standard HMC for the 2-flavour model
- Phase 6: TDF-based canonical HMC for fixed isospin sectors
- Phase 7: comparison of acceptance rates, autocorrelations, timings, and physics
