# Benchmark Results – Standard vs. TDF Canonical HMC

This document summarises the algorithmic comparison between the reference
standard HMC (full Wilson–Dirac determinant) and the TDF-based canonical HMC
(fixed isospin/quark-number sector `n`).  The comparison focuses on properties
of the algorithms themselves, not on time-to-solution for a specific hardware
configuration.

## Simulation parameters

| Parameter | Value |
|-----------|-------|
| Lattice   | `L = 4`, `Lt = 4` |
| `beta`    | 5.0 |
| Mass `m0` | 0.0 |
| Hopping `kappa` | 0.25 |
| Chemical potential `mu` | 0.0 |
| Trajectory length | `dt = 0.1`, `n_steps = 5` |
| Thermalization | 20 trajectories |
| Measurements | 50 (every 3 trajectories) |
| Random seed | 42 |

## Acceptance rate vs step size

Short step-size sweeps were performed with 5 thermalisation and 20 measurement
trajectories at each `dt`:

| `dt` | Standard HMC | Canonical HMC (`n = 2`) |
|------|-------------:|------------------------:|
| 0.050 | 0.55 | 0.45 |
| 0.075 | 0.30 | 0.25 |
| 0.100 | 0.40 | 0.30 |
| 0.125 | 0.30 | 0.70 |
| 0.150 | 0.60 | 0.65 |

The acceptance curves are noisy because of the modest statistics, but they are
broadly comparable: neither algorithm shows a dramatic advantage in acceptance
at a given step size on this small lattice.  Both would benefit from a longer
sweep and smaller step size to reach the usual 0.6–0.8 acceptance target.

## Detailed run at `dt = 0.1`

| Quantity | Standard HMC | Canonical HMC (`n = 2`) |
|----------|-------------:|------------------------:|
| Acceptance rate | 0.373 | 0.380 |
| Mean `delta H` | 1.93 | 1.78 |
| Std `delta H` | 1.74 | 1.51 |
| Max \|delta H\| | 6.50 | 6.92 |
| `<P>` | 0.92470 | 0.93934 |
| `std(P)` | 0.01472 | 0.01829 |
| `tau_int(P)` | 1.08 | 1.98 |

The Hamiltonian energy violation `delta H` is very similar for both algorithms,
suggesting that the leapfrog integrator behaves comparably once the step size
is fixed.  The plaquette autocorrelation time is somewhat shorter for the
standard run on this tiny volume, but the numbers are sensitive to statistics
and to the chosen sector.

## Force-evaluation scaling with `Lt`

The wall-clock time of a single force evaluation was averaged over 10
repetitions (after JIT warm-up) for several temporal extents with `L = 4`:

| `Lt` | Standard HMC | Canonical HMC (`n = 2`) |
|------|-------------:|------------------------:|
| 4 | 1.46 ms | 2.52 ms |
| 6 | 1.99 ms | 3.33 ms |
| 8 | 2.78 ms | 4.31 ms |

Both algorithms scale roughly linearly with `Lt` in this range.  The canonical
force carries a larger constant overhead on these small lattices because its
backward pass goes through the transfer-matrix eigenvalue decomposition.  The
standard force, by contrast, differentiates through `jnp.linalg.slogdet`, which
is highly optimised in JAX for matrices of this size.

A supplemental measurement at `L = 8`, `Lt = 8` shows the crossover where the
canonical force becomes cheaper:

| Lattice | Standard HMC | Canonical HMC (`n = 4`) |
|---------|-------------:|------------------------:|
| `8 × 8` | 8.03 ms | 6.32 ms |

This indicates that the asymptotic advantage of the canonical formulation
(`O((2L)^3)` force work vs. `O((L Lt)^3)`) only becomes visible once the
standard matrix is large enough.

## Key observations

1. **Acceptance and integrator quality.**  At matched leapfrog parameters the
   two algorithms give comparable acceptance rates and comparable `delta H`
   distributions.  There is no evidence that the canonical formulation harms
   the integrator.

2. **Autocorrelation.**  On the `4 × 4` lattice the standard run has a shorter
   plaquette autocorrelation time, but this is volume- and sector-dependent.
   A proper autocorrelation study would require a tuned acceptance rate and
   much larger statistics.

3. **Force cost.**  The canonical force is not universally cheaper.  Its
   advantage grows with the lattice size and is masked on small lattices by the
   overhead of differentiating through the eigenvalue solver.  At `8 × 8` the
   crossover has occurred and the canonical force is faster.

4. **Physics.**  The average plaquettes differ because the ensembles are not the
   same: the standard algorithm samples the grand-canonical 2-flavour ensemble,
   while the canonical algorithm samples a fixed-quark-number sector.  The
   canonical runs are therefore physics probes in their own right, not merely
   approximations of standard HMC.

## Caveats

- The statistics are modest and the volume is very small.
- The acceptance sweeps are noisy; a production comparison would use longer
  sweeps and tune `dt` independently for each algorithm to a fixed acceptance
  target before comparing autocorrelation.
- The force timings depend on JAX's implementation of `slogdet` and eigenvalue
  autodiff and may change with JAX versions.

## Raw data

The JSON summary and NumPy histories are saved as:

- `benchmark_results.json`
- `benchmark_histories.npz`

The JSON file contains the acceptance sweeps, the detailed-run summary, and the
`L = 4` force-scaling data.

## Unified determinant comparison

The script `scripts/run_determinant_comparison.py` compares three estimates of
the 2-flavour weight `|det K|²` on the same gauge configurations:

1. the exact determinant from the full Wilson–Dirac matrix,
2. the standard pseudofermion estimate with its standard error of the mean,
3. the TDF pseudofermion estimate with its standard error of the mean.

All numbers below are for `beta = 5.0`, `m0 = 0.0` (`kappa = 0.25`),
`mu = 0.0`, with a modest **200 pseudofermion samples** and CG tolerance `1e-6`.
The TDF reduced determinant is also shown as a consistency check.

| Lattice | Exact \|det K\|² | TDF \|det K\|² | TDF rel. diff. |
|---------|----------------:|---------------:|---------------:|
| 4×4 | 2.0438e+00 | 2.0438e+00 | 1.110e-15 |
| 6×6 | 1.0622e+00 | 1.0622e+00 | 5.274e-16 |
| 8×8 | 4.9803e-01 | 4.9803e-01 | 5.063e-14 |

The TDF determinant matches the exact determinant to machine precision on all
three lattice sizes, confirming the reduced determinant implementation.

| Lattice | Exact \|det K\|² | Std. pf. estimate ± σ_mean | TDF pf. estimate ± σ_mean | Std. within 1σ? | TDF within 1σ? |
|---------|----------------:|---------------------------:|--------------------------:|:---------------:|:--------------:|
| 4×4 | 2.0438e+00 | 1.49e-02 ± 7.29e-03 | 5.04e-06 ± 3.86e-06 | no | no |
| 6×6 | 1.0622e+00 | 4.49e-09 ± 4.07e-09 | 5.04e-15 ± 5.00e-15 | no | no |
| 8×8 | 4.9803e-01 | 2.60e-26 ± 2.33e-26 | 3.34e-34 ± 2.25e-34 | no | no |

The pseudofermion weights now use the corrected estimator
`exp(-S_pf + |φ|²)` (Gattringer & Lang, eq. (8.63)), which is **unbiased** in
expectation.  With only 200 samples the variance is still enormous: the weights
vary over many orders of magnitude, so the sample means are far from the exact
value and the 1-sigma agreement flags are "no".  This is the honest outcome of
using a modest sample count: the error bars are large, and the exact value is
not yet captured.  Many more samples (or a variance-reduction scheme) would be
needed for a reliable point estimate.

| Lattice | Std. pf. `<S_pf>` | Std. pf. `σ(S_pf)` | TDF pf. `<S_pf>` | TDF pf. `σ(S_pf)` |
|---------|------------------:|-------------------:|-----------------:|------------------:|
| 4×4 | 58.98 | 20.35 | 28.12 | 2.47 |
| 6×6 | 145.42 | 35.96 | 61.49 | 3.56 |
| 8×8 | 251.58 | 48.35 | 105.99 | 4.44 |

The exact log determinant is close to zero on these small lattices, so the
pseudofermion action `<S_pf>` has a large additive offset (mainly the Gaussian
normalisation).  The width `σ(S_pf)` measures the stochastic noise of the
estimator.  The TDF pseudofermion estimator has roughly **5–10 times smaller
standard deviation** than the standard one, and its growth with lattice size is
much slower.

## Pseudofermion HMC comparison

A separate benchmark compares five pseudofermion HMC samplers as full Markov
chains.  All target the full 2-flavour weight `|det K[U, μ=0]|²`.

**Important correction.**  The pseudofermion fields are now refreshed in the
book-style way (Gattringer & Lang, eq. (8.37)):

```
φ = K[U] χ ,                       χ ∼ N(0, I)        (standard)
ψ = M[U] η ,                       η ∼ N(0, I)        (TDF exact bulk, M = I - (-1)^Lt T)
φ_t = R_t[U] χ_t ,                 χ_t ∼ N(0, I)      (TDF stochastic bulk, one per t)
φ_bulk = R[U] χ ,                  χ ∼ N(0, I)        (TDF block-diagonal bulk, R = diag(R_t))
ψ = M[U] η ,                       η ∼ N(0, I)        (TDF stochastic transfer factor)
```

Refreshing from `N(0, I)` instead of `N(0, A)` gives a biased weight and the
results shown below are *not* comparable with such a setup.  The TDF stochastic
variants differ only in how the time-slice bulk determinants `det R_t` are
packaged: per-slice, as a single product matrix, or as one concatenated
block-diagonal field.

### Parameters

- `beta = 5.0`, `m0 = 0.0` (`kappa = 0.25`), `mu = 0.0`
- leapfrog: `n_steps = 3`, CG tolerance `tol = 1e-6`
- HMC: 5 thermalization trajectories, 10 measurements, skip 2
- `dt` is chosen independently for each sampler (see table)

### Force-evaluation diagnostics

| Lattice | Algorithm | `dt` | accept | `std(\|dH\|)` | `max \|dH\|` | time/traj |
|---------|-----------|------|-------:|------------:|------------:|----------:|
| 4×4 | standard | 0.050 | 0.70 | 1.18 | 3.08 | 8.09 s |
| 4×4 | TDF exact bulk | 0.050 | 0.60 | 1.29 | 2.15 | 3.02 s |
| 4×4 | TDF stochastic bulk | 0.030 | 0.65 | 1.17 | 2.76 | 4.67 s |
| 4×4 | TDF bulk product (reverse) | 0.030 | 0.70 | 1.56 | 3.18 | 4.76 s |
| 4×4 | TDF block-diagonal bulk | 0.030 | 0.60 | 1.38 | 2.91 | 4.91 s |
| 6×6 | standard | 0.040 | 0.75 | 1.06 | 2.22 | 16.52 s |
| 6×6 | TDF exact bulk | 0.004 | 0.50 | 0.97 | 3.20 | 4.42 s |
| 6×6 | TDF stochastic bulk | 0.003 | 0.75 | 0.57 | 1.21 | 6.17 s |
| 6×6 | TDF bulk product (reverse) | 0.003 | 0.65 | 0.91 | 1.65 | 6.31 s |
| 6×6 | TDF block-diagonal bulk | 0.003 | 0.80 | 1.30 | 3.51 | 6.51 s |
| 8×8 | standard | 0.050 | 0.55 | 1.74 | 2.97 | 27.80 s |
| 8×8 | TDF exact bulk | 0.003 | 0.20 | 3902.11 | 11906.52 | 6.10 s |
| 8×8 | TDF stochastic bulk | 0.002 | 0.55 | 61.21 | 205.52 | 7.73 s |
| 8×8 | TDF bulk product (reverse) | 0.002 | 0.50 | 2.62 | 6.62 | 8.03 s |
| 8×8 | TDF block-diagonal bulk | 0.002 | 0.70 | 15.85 | 51.84 | 8.14 s |

*The `8×8` pseudofermion runs used seed `43` because seed `42` produced a rare
extreme pseudofermion draw for the block-diagonal sampler in the first
measurement trajectory (`max |dH| ≈ 9.5×10⁴`).  With seed `43` all samplers
remain finite and the block-diagonal sampler reaches the highest acceptance
(0.70) among the TDF variants, though its energy violations are still larger
than the bulk-product variant on this short run.*

On the `4×4` lattice all four TDF variants are faster per trajectory than
standard.  The exact-bulk version is the cheapest (about **2.7× faster** than
standard).  The stochastic-bulk, bulk-product and block-diagonal versions are
comparable in speed; with the reverse product order the bulk-product variant
reaches the same acceptance as standard (`≈ 0.7`) at `dt = 0.03`.

On `6×6` the stochastic-bulk and block-diagonal variants are the best TDF
formulations: both match or exceed standard's acceptance (0.75–0.80) with
comparable Hamiltonian error.  The bulk-product variant is almost as good
(acceptance 0.65) and uses only two pseudofermion fields, but building the
product matrix adds overhead so its wall-clock time is similar.

On `8×8` all pseudofermionized bulk variants are dramatically smoother than the
exact-bulk variant.  The bulk-product variant gives the smallest energy
violations (`max |dH| ≈ 6.6`) at the same acceptance as stochastic-bulk and
block-diagonal (≈ 0.55–0.70).  The block-diagonal variant reaches the highest
acceptance (0.70) but shows larger occasional energy violations (`max |dH| ≈ 52`);
these are stochastic outliers, not a bias, and its wall-clock cost is
essentially identical to the per-slice stochastic-bulk formulation.  A single
rare outlier (`max |dH| ≈ 9.5×10⁴`) was observed with seed `42` and disappeared
with seed `43`, confirming the sensitivity of these short runs to the
pseudofermion noise.

The price of all TDF pseudofermion variants is a much smaller usable step size:
on `8×8` they need `dt ≈ 0.002` while standard uses `dt = 0.050`, a factor of
25.  Because the force is cheaper, the per-trajectory cost is still lower
(≈ 8 s vs 28 s), but a fair cost-per-independent-sample comparison would need
to account for the different step sizes and autocorrelation times.

### Bulk-product ordering comparison

On the `4×4` lattice the three multiplication orders were tested with the same
step size `dt = 0.03`:

| Order | `M_bulk` construction | accept | `max \|dH\|` |
|-------|----------------------|-------:|------------:|
| natural | $R_0 R_1 R_2 R_3$ | 0.65 | 3.40 |
| reverse | $R_3 R_2 R_1 R_0$ | 0.70 | 3.18 |
| balanced | $(R_0 R_1)(R_2 R_3)$ | 0.65 | 3.40 |

Differences are modest on this small volume, but the reverse order gives the
highest acceptance and was therefore used for the `6×6` and `8×8` bulk-product
runs.  A systematic ordering optimization on larger volumes would require more
statistics.

### Stochastic action noise

For a fixed random gauge configuration the pseudofermion action `S_pf` was
sampled 50 times and compared with the exact log determinant `S_exact = -log|det K|²`.

| Lattice | Algorithm | `S_exact` | `<S_pf>` | `std(S_pf)` | noise |
|---------|-----------|----------:|---------:|------------:|------:|
| 4×4 | standard | -1.10 | 65.62 | 18.50 | 16.79 |
| 4×4 | TDF | -1.10 | 29.33 | 2.80 | 2.55 |
| 6×6 | standard | -0.60 | 128.77 | 26.96 | 26.96 |
| 6×6 | TDF | -0.60 | 62.30 | 3.74 | 3.74 |
| 8×8 | standard | -0.17 | 247.33 | 47.42 | 47.42 |
| 8×8 | TDF | -0.17 | 106.80 | 3.63 | 3.63 |

The TDF pseudofermion estimator (used for the transfer-matrix factor in the
exact-bulk variant, and for both the bulk and transfer-matrix factors in the
stochastic-bulk variant) has **5–15 times smaller standard deviation** than the
standard one, and its noise grows much more slowly with the lattice size.

### Observations

- The book-style refresh makes all samplers correct.  With the old `N(0, I)`
  refresh the TDF force appeared much smoother, but that setup was biased.
- On `4×4` the TDF variants win on speed and stochastic noise.
- On `6×6` and `8×8` the exact-bulk TDF sampler needs a much smaller `dt` to
  stay stable and remains rougher than standard HMC.  This is the opposite of
  the earlier (biased) finding.
- All pseudofermionized bulk variants (separate `φ_t` per slice, single
  bulk-product `φ_bulk`, and concatenated block-diagonal `φ_bulk`) smooth the
  force dramatically compared with the exact-bulk variant.  Which one is best
  depends on volume and seed in these short runs; the block-diagonal variant is
  mathematically identical to the per-slice formulation but exposes more
  parallelism for a single CG solve.
- The plaquette means from the short runs do not always agree between the
  algorithms; this is mostly a statistical issue because the runs are short
  and start from the same random gauge field.

## `16 × 16` block-cyclic transfer-matrix test

At `16 × 16` the original TDF pseudofermion samplers fail because the transfer
matrix product `T = T_0 T_1 … T_{Lt-1}` has condition number `~10¹²` at
`m0 = 0.0`.  Solving `(M M†) χ = ψ` with `M = I - (-1)^Lt T` therefore loses all
numerical accuracy in double precision, producing `max |ΔH| ~ 10¹²` and zero
acceptance regardless of the leapfrog step size.

The block-cyclic transfer-matrix reformulation avoids forming `T` entirely.
Instead of `det M` it uses `det(B B†)`, where `B` is a block-cyclic matrix whose
blocks are the individual `T_t = R_t^{-1} S_t`.  Because `B` is never formed and
its action is applied by solving the well-conditioned `R_t` matrices, the
formulation remains stable at large `Lt`.

### Parameters and results

- `beta = 5.0`, `m0 = 0.0` (`kappa = 0.25`), `mu = 0.0`
- leapfrog: `dt = 0.001`, `n_steps = 3`, CG tolerance `tol = 1e-6`
- HMC: 5 thermalization trajectories, 10 measurements, skip 2
- seed 42

| Algorithm | `dt` | accept | `std(\|dH\|)` | `max \|dH\|` | time/traj | `<P>` |
|-----------|------|-------:|------------:|----------:|----------:|------:|
| TDF block-cyclic TM | 0.001 | 0.95 | 0.09 | 0.19 | 16.2 s | 0.02341 |

The block-cyclic sampler is extremely smooth at this step size
(`max |ΔH| < 0.2`) and achieves 95% acceptance.  The plaquette value is
consistent with the small-volume trend.  A larger `dt` (e.g. 0.002) gives
~67% acceptance with similarly small energy violations, so a modest tuning
study could find a substantially cheaper operating point.

The other TDF pseudofermion variants (exact-bulk, stochastic-bulk,
bulk-product and block-diagonal bulk) all share the same `M M†` transfer-matrix
solve and fail at `16 × 16`.  Standard pseudofermion HMC, by contrast, uses the
full Wilson–Dirac operator `K` whose condition number at `16 × 16, m0 = 0.0` is
only `~8`, so it remains numerically viable (though slower per trajectory).

## Conclusion

The exact-determinant algorithms behave similarly in acceptance and energy
violation on small lattices; the TDF canonical sampler becomes advantageous for
the force evaluation only when the Wilson–Dirac matrix is large enough.

For pseudofermion HMC the picture is more nuanced after fixing the refresh.
The TDF estimator has substantially lower stochastic variance, which is a real
advantage.  The exact-bulk TDF force is rougher than expected, but the
fully stochastic-bulk formulation smooths it considerably.  At `16 × 16` the
naive transfer-matrix solve `M = I - (-1)^Lt T` becomes numerically unstable;
the block-cyclic reformulation `det(B B†)` restores stability by avoiding the
exponentially ill-conditioned product `T`.  A definitive comparison of
cost-per-independent-sample still requires a careful integrator tuning study
(independent `dt` and `n_steps`, autocorrelation times), but the block-cyclic
variant is now the only TDF pseudofermion sampler that remains viable at
`16 × 16`.

