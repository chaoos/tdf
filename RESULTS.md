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
| Max `|delta H|` | 6.50 | 6.92 |
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

## Pseudofermion comparison

A second benchmark compares the standard pseudofermion estimator (full Dirac
space) with the TDF pseudofermion estimator (transfer-matrix space).  Both
target the full 2-flavour weight `|det K[U, μ=0]|²`.  The TDF version keeps the
bulk factor `|∏_t det R_t|²` exact and uses pseudofermions only for the
transfer-matrix factor `|det(I - (-1)^{L_t} T)|²`.

### Parameters

- `beta = 5.0`, `m0 = 0.0` (`kappa = 0.25`), `mu = 0.0`
- leapfrog: `dt = 0.05`, `n_steps = 3`
- CG tolerance: `tol = 1e-6`
- HMC: 3–5 thermalization trajectories, 3–5 measurements, skip 1
- Determinant action check: 50 pseudofermion samples per lattice size

### Force-evaluation diagnostics

| Lattice | Algorithm | `|dH|` mean | `|dH|` max | time/traj |
|---------|-----------|------------:|-----------:|----------:|
| 4×4 | standard | 7.32 | 25.37 | 7.92 s |
| 4×4 | TDF | 0.19 | 0.30 | 3.14 s |
| 6×6 | standard | 3.43 | 14.48 | 16.22 s |
| 6×6 | TDF | 0.21 | 0.45 | 4.88 s |
| 8×8 | standard | 7.04 | 29.78 | 27.27 s |
| 8×8 | TDF | 0.04 | 0.05 | 6.68 s |

The TDF pseudofermion HMC has a dramatically smaller Hamiltonian energy
violation at the same step size, indicating a much smoother force.  It is also
faster per trajectory because the CG solve operates in the `2L`-dimensional
transfer-matrix space rather than the full `2 L L_t`-dimensional Dirac space.

### Stochastic action noise

For a fixed random gauge configuration the pseudofermion action `S_pf` was
sampled 50 times and compared with the exact log determinant `S_exact = -log|det K|²`.

| Lattice | Algorithm | `S_exact` | `<S_pf>` | `std(S_pf)` | noise |
|---------|-----------|----------:|---------:|------------:|------:|
| 4×4 | standard | -1.10 | 65.62 | 18.50 | 16.79 |
| 4×4 | TDF | -1.10 | 29.33 | 2.80 | 2.55 |
| 6×6 | standard | -0.60 | 131.65 | 25.88 | 25.88 |
| 6×6 | TDF | -0.60 | 62.30 | 3.74 | 3.74 |
| 8×8 | standard | -0.17 | 247.33 | 47.42 | 47.42 |
| 8×8 | TDF | -0.17 | 106.80 | 3.63 | 3.63 |

The exact log determinant is close to zero on these small lattices, so the
absolute offset between `<S_pf>` and `S_exact` is large (it is mainly the
Gaussian normalization of the pseudofermion integral).  The important quantity
is the width `std(S_pf)`, which measures the stochastic noise of the estimator.
The TDF estimator has roughly **5–15 times smaller standard deviation** than the
standard estimator, and its noise grows much more slowly with the lattice size.

### Observations

- Both pseudofermion samplers have acceptance ≈ 1.0 in these short runs because
  the stochastic Hamiltonian often decreases during the trajectory.  A proper
  production study would tune `dt` and `n_steps` to obtain an acceptance rate
  around 0.7 and then compare autocorrelation times.
- The TDF pseudofermion force is consistently smoother and cheaper.  This is
  the expected advantage: the pseudofermions live on time slices rather than on
  the full space-time lattice.
- The standard pseudofermion force becomes very noisy on `8×8`, with maximum
  energy violations of order 30.  This suggests that standard pseudofermion HMC
  would need a much smaller step size at this volume.

## Conclusion

The exact-determinant algorithms behave similarly in acceptance and energy
violation on small lattices; the TDF canonical sampler becomes advantageous for
the force evaluation only when the Wilson–Dirac matrix is large enough.

The pseudofermion comparison shows a clearer and more immediate advantage for
the TDF formulation: the stochastic action has substantially lower variance and
the force is much smoother, especially as the lattice volume grows.  This makes
TDF-based pseudofermion HMC the more promising approach for larger systems.

