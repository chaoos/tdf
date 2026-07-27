# Benchmark Results – Standard vs. TDF Canonical HMC

This document summarises the head-to-head comparison between the reference
standard HMC (full Wilson–Dirac determinant) and the TDF-based canonical HMC
(fixed isospin/quark-number sector `n`).

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

The same initial gauge field and PRNG key were used for all three runs so that
any difference in performance comes from the algorithm, not from the random
starting point.

## Measured quantities

| Algorithm | Accept rate | `<P>` | `std(P)` | `tau_int(P)` | Time/traj [s] | Cost/ind. sample [s] |
|-----------|------------:|------:|---------:|-------------:|--------------:|---------------------:|
| Standard Nf=2 HMC | 0.307 ± 0.265 | 0.91920 | 0.01763 | 3.62 | 0.671 | 7.28 |
| Canonical HMC, `n = 0` | 0.247 ± 0.239 | 0.92636 | 0.02409 | 2.64 | 0.239 | 1.89 |
| Canonical HMC, `n = 2` | 0.227 ± 0.262 | 0.93529 | 0.01026 | 0.99 | 0.233 | 0.69 |

*`<P>`*: average plaquette.  
*`tau_int(P)`*: rough integrated autocorrelation time of the plaquette.  
*Cost/ind. sample*: `time_per_trajectory × n_skip × tau_int(P)`, i.e. the wall
clock time needed for one effectively independent measurement.

## Key observations

1. **Speed per trajectory.**  The canonical sampler is roughly **2.8× faster**
   per trajectory than the standard sampler on this lattice.  The standard
   algorithm differentiates through the full `L·Lt = 16` site Wilson–Dirac
   determinant (a `32 × 32` complex matrix), whereas the canonical algorithm
   only needs the transfer-matrix spectrum (a `2L × 2L = 8 × 8` real matrix).

2. **Autocorrelation.**  The canonical sector `n = 2` shows a much shorter
   plaquette autocorrelation time (`tau_int ≈ 1`) than the standard run
   (`tau_int ≈ 3.6`).  The `n = 0` sector is in between.  This is partly a
   volume/sector effect: constraining the quark number can change the shape of
the distribution being sampled.

3. **Effective cost per independent sample.**  Combining speed and
   autocorrelation, the `n = 2` canonical run is about **10× cheaper** per
   independent plaquette measurement than the standard run on the `4 × 4`
   lattice.  The `n = 0` sector is about **3.9× cheaper**.

4. **Acceptance rates.**  All three runs use the same leapfrog parameters
   (`dt = 0.1`, 5 steps).  Acceptance is comparable but on the low side
   (~0.2–0.3), indicating that a smaller step size or more leapfrog steps would
   be needed for production-quality runs.  Importantly, the canonical algorithm
   does not suffer a dramatic acceptance penalty compared with the standard
   algorithm; the dominant gain comes from the cheaper force evaluation.

5. **Physics.**  The average plaquettes differ because the ensembles are not the
   same:
   - Standard HMC samples the grand-canonical 2-flavour ensemble.
   - Canonical HMC samples a fixed-quark-number sector `n`.
   At `mu = 0` these sectors are related by the fugacity expansion, but a single
   sector is not identical to the full grand-canonical ensemble.  The canonical
   runs are therefore physics probes in their own right (e.g. for finite-density
   or fixed-charge simulations), not just an approximation of standard HMC.

6. **Topological charge.**  On the tiny `4 × 4` lattice at `beta = 5` the
   topological charge is essentially zero for all runs (`std(Q) ~ 10⁻¹⁷`), so no
   meaningful comparison of tunneling or autocorrelation in the topological
   sector can be made at this volume.

## Caveats

- The statistics are modest (50 measurements) and the volume is very small.
  The autocorrelation times are rough estimates and should not be over-interpreted.
- The acceptance rates are low; a proper tuning of `dt` and `n_steps` would be
  required before drawing strong conclusions about cost.
- The speed-up from TDF is expected to grow with `Lt` because the standard
  algorithm must repeatedly differentiate through an `L·Lt × L·Lt` determinant,
  while the canonical algorithm works with the fixed `2L × 2L` transfer matrix.
  On the `4 × 4` lattice the gain is already visible; it should become more
  pronounced on `6 × 6` or `8 × 8` lattices.

## Raw data

The JSON summary and NumPy plaquette histories are saved as:

- `benchmark_results.json`
- `benchmark_histories.npz`

These files can be loaded for plotting or for longer runs with increased
statistics.

## Conclusion

The TDF-based canonical HMC sampler is implemented and functional.  On a
`4 × 4` lattice it is already faster per trajectory than the reference standard
HMC, and when autocorrelation is taken into account the `n = 2` canonical sector
appears to deliver independent measurements roughly an order of magnitude more
cheaply.  The main remaining work for a production study is to tune trajectory
parameters and collect statistics on larger volumes.
