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

## Conclusion

The two algorithms behave similarly in acceptance and energy violation on small
lattices.  The TDF canonical sampler becomes advantageous for the force
evaluation only when the Wilson–Dirac matrix is large enough; on the `4 × 4`
lattice its eigenvalue-based backward pass is slower than `slogdet`.  The main
algorithmic value of the canonical formulation lies in enabling direct sampling
of fixed-charge sectors rather than in a universal speed-up at small volume.
