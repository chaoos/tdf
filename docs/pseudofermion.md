# Pseudofermion HMC

This page describes the two stochastic HMC samplers that estimate the 2-flavour
fermion determinant with pseudofermions instead of evaluating the determinant
exactly.

## The pseudofermion trick

For a positive-definite matrix `A` the Gaussian integral identity

```
det A = ∫ Dφ Dφ† exp(-φ† A^{-1} φ)
```

allows us to replace the exact determinant by an auxiliary bosonic field `φ`.
In HMC, `φ` is refreshed from a complex Gaussian at the start of each
trajectory and the gauge field is evolved with the stochastic action

```
S_pf(θ, φ) = φ† A(θ)^{-1} φ .
```

The same gauge-field marginal distribution is obtained as with the exact
fermion determinant.

## Standard pseudofermion HMC

For the 2-flavour Schwinger model the relevant matrix is

```
A = K[U] K[U]† ,
```

where `K[U]` is the Wilson–Dirac operator.  The pseudofermion vector lives in
the full Dirac space of dimension `2 L Lt`.  Each force evaluation requires
one Conjugate-Gradient solve of

```
A χ = φ .
```

Use `scripts/run_hmc_pseudofermion_standard.py`:

```bash
.venv/bin/python scripts/run_hmc_pseudofermion_standard.py --L 4 --Lt 4 \
    --beta 5.0 --mass 0.0 --dt 0.05 --n-steps 5 --n-therm 10 \
    --n-measure 20 --n-skip 2 --tol 1e-6
```

## TDF pseudofermion HMC

Using the temporal determinant factorisation

```
det K[U] = (prod_t det R_t) * det(I - (-1)^Lt T) ,
```

the bulk factor is kept exact and pseudofermions are only used for the
transfer-matrix factor.  Define

```
M = I - (-1)^Lt T .
```

The pseudofermion vector now lives in the much smaller transfer-matrix space
of dimension `2L`, and each force evaluation solves

```
(M M†) χ = ψ .
```

Use `scripts/run_hmc_pseudofermion_tdf.py`:

```bash
.venv/bin/python scripts/run_hmc_pseudofermion_tdf.py --L 4 --Lt 4 \
    --beta 5.0 --mass 0.0 --dt 0.1 --n-steps 5 --n-therm 10 \
    --n-measure 20 --n-skip 2 --tol 1e-6
```

## CG solver and verbosity

Both scripts accept

- `--tol`: relative residual stopping criterion `||A χ - φ|| / ||φ|| < tol`,
- `--cg-maxiter`: maximum CG iterations (defaults to the matrix dimension),
- `-v` / `--verbose`: print the CG residual at every iteration.

With `-v` the `tdf` loggers are set to `DEBUG` and `jax.debug.print` emits one
line per CG iteration.

## Algorithmic comparison

The benchmark script `scripts/run_pseudofermion_benchmark.py` runs both
samplers with identical leapfrog parameters and reports:

- acceptance rate,
- distribution of the Hamiltonian violation ΔH,
- plaquette autocorrelation time,
- wall-clock time per trajectory,
- statistics of the pseudofermion action versus the exact log determinant.

See [`RESULTS.md`](../RESULTS.md) for representative numbers on `4x4`,
`6x6` and `8x8` lattices.

## Caveats

- The pseudofermion action is a stochastic estimator, so the Hamiltonian is
  noisy.  Smaller step sizes are usually required than with the exact
  determinant.
- At the parameters used in the benchmark (`m0=0`, `beta=5`) the standard
  pseudofermion force is considerably noisier than the TDF one.  A production
  run should tune `dt` and `n_steps` independently for each algorithm to reach
  the desired acceptance rate.
- The current implementation is not JIT-optimised across trajectories because
  the pseudofermion vector changes every trajectory.  Absolute timings should
  therefore be interpreted cautiously; the relevant comparison is the relative
  scaling and the stochastic noise.
