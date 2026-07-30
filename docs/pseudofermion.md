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
In HMC, `φ` must be refreshed from its **conditional Gaussian distribution**
rather than from a plain complex Gaussian.  Following Gattringer & Lang,
introduce an auxiliary noise vector `χ ∼ N(0, I)` and set

```
φ = D χ ,     D D† = A .
```

For example, for the standard formulation `D = K[U]` so that
`φ ∼ N(0, K K†)`.  The gauge field is then evolved with the stochastic action

```
S_pf(θ, φ) = φ† A(θ)^{-1} φ .
```

The same gauge-field marginal distribution is obtained as with the exact
fermion determinant.  Using `φ ∼ N(0, I)` instead of `φ ∼ N(0, A)` gives a
biased weight and must be avoided.

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

there are two TDF pseudofermion samplers.

### Exact-bulk variant

The bulk factor `prod_t det R_t` is kept exact and pseudofermions are only
used for the transfer-matrix factor.  Define

```
M = I - (-1)^Lt T .
```

The pseudofermion vector lives in the transfer-matrix space of dimension `2L`.
It is refreshed in the book-style way,

```
ψ = M η ,     η ∼ N(0, I) ,
```

and each force evaluation solves

```
(M M†) χ = ψ .
```

Use `scripts/run_hmc_pseudofermion_tdf.py`:

```bash
.venv/bin/python scripts/run_hmc_pseudofermion_tdf.py --L 4 --Lt 4 \
    --beta 5.0 --mass 0.0 --dt 0.1 --n-steps 5 --n-therm 10 \
    --n-measure 20 --n-skip 2 --tol 1e-6
```

### Stochastic-bulk variant

Every determinant factor is represented by pseudofermions.  For each time
slice an independent pseudofermion is drawn from `N(0, R_t R_t†)`,

```
φ_t = R_t χ_t ,     χ_t ∼ N(0, I) ,
```

and the transfer-matrix factor uses an additional field `ψ = M η` as above.
The total action is

```
S = Σ_t φ_t† (R_t R_t†)^{-1} φ_t + ψ† (M M†)^{-1} ψ .
```

This variant avoids the exact evaluation of the bulk determinant and its
gradient entirely.  It is implemented in
`tdf.hmc_pseudofermion.run_hmc_pseudofermion_tdf_stochastic_bulk` and can be
run through `scripts/run_pseudofermion_benchmark.py`.

### Bulk-product variant

Instead of one pseudofermion per time slice, the whole bulk factor can be
represented by a single pseudofermion using

```
prod_t det R_t = det(prod_t R_t) .
```

Build the product matrix

```
M_bulk = R_{π(0)} R_{π(1)} ... R_{π(Lt-1)}
```

for a chosen ordering `π` (natural, reverse, or balanced pairwise), draw
`χ ∼ N(0, I)`, and set `φ_bulk = M_bulk χ`.  The action is

```
S = φ_bulk† (M_bulk M_bulk†)^{-1} φ_bulk + ψ† (M M†)^{-1} ψ .
```

This reduces the number of pseudofermion fields from `Lt + 1` to two, but the
CG solve for `M_bulk M_bulk†` can be more ill-conditioned than the individual
`R_t R_t†` solves.  The ordering can be tuned; see `RESULTS.md` for a
comparison.

### Block-diagonal bulk variant

The stochastic-bulk action can be rewritten as a single quadratic form on a
block-diagonal matrix:

```
A_bulk = diag(R_0 R_0†, R_1 R_1†, ..., R_{Lt-1} R_{Lt-1}†) .
```

A single concatenated pseudofermion is drawn in the book-style way,

```
φ_bulk = R χ ,     R = diag(R_0, R_1, ..., R_{Lt-1}) ,     χ ∼ N(0, I) ,
```

and the action is

```
S = φ_bulk† A_bulk^{-1} φ_bulk + ψ† (M M†)^{-1} ψ .
```

Mathematically this is identical to the per-slice stochastic-bulk formulation;
the difference is purely computational.  The block-diagonal operator is applied
with a custom matrix-vector product that reshapes the concatenated vector into
`Lt` blocks of size `2L`, applies `R_t R_t†` to each block, and flattens the
result back.  This avoids storing the off-diagonal zeros and lets a single CG
solve act on the whole bulk vector, which is more GPU-friendly than `Lt`
separate solves.  It is implemented in
`tdf.hmc_pseudofermion.run_hmc_pseudofermion_tdf_block_diagonal`.

### Block-cyclic transfer-matrix variant

The transfer-matrix factor `det M = det(I - (-1)^Lt T)` with
`T = T_0 T_1 … T_{Lt-1}` becomes numerically unstable when `T` is formed
explicitly: its eigenvalues grow exponentially with `Lt`.  We can avoid the
product entirely by introducing a block-cyclic matrix `B` with blocks

```
B_{t,t}   = I,
B_{t,t+1} = -(-1)^Lt T_t    (indices mod Lt).
```

The block-cyclic determinant identity gives

```
det B = det(I - (-1)^Lt T_0 T_1 … T_{Lt-1}) = det M ,
```

so `|det M|^2 = det(B B†)`.  The transfer-matrix factor is therefore
represented by a block-cyclic pseudofermion

```
ψ = B η ,    η ∼ N(0, I) ,
```

with action

```
S_tm = ψ† (B B†)^{-1} ψ .
```

The matrix `B` is never formed.  Its action is implemented by applying each
`T_t` (and `T_t†`) through sequential solves against the well-conditioned
`R_t` matrices.  This makes the TDF pseudofermion formulation stable even at
large temporal extent: on a `16 × 16` lattice at `m0 = 0.0` the block-cyclic
sampler reaches ~95% acceptance with `max |ΔH| < 0.2`, while the original
`M M†` formulation fails because `cond(T) ~ 10¹²`.  It is implemented in
`tdf.hmc_pseudofermion.run_hmc_pseudofermion_tdf_block_cyclic_tm`.

## CG solver and verbosity

Both scripts accept

- `--tol`: relative residual stopping criterion `||A χ - φ|| / ||φ|| < tol`,
- `--cg-maxiter`: maximum CG iterations (defaults to the matrix dimension),
- `-v` / `--verbose`: print the CG residual at every iteration.

With `-v` the `tdf` loggers are set to `DEBUG` and `jax.debug.print` emits one
line per CG iteration.

## Algorithmic comparison

The benchmark script `scripts/run_pseudofermion_benchmark.py` runs all five
bulk samplers (standard, TDF exact-bulk, TDF stochastic-bulk, TDF bulk-product,
TDF block-diagonal bulk).  The block-cyclic transfer-matrix sampler is used for
large-lattice focused tests because it avoids the ill-conditioned transfer-
matrix product.  All samplers use independently chosen leapfrog step sizes and
report:

- acceptance rate,
- distribution of the Hamiltonian violation ΔH,
- plaquette autocorrelation time,
- wall-clock time per trajectory,
- statistics of the pseudofermion action versus the exact log determinant.

For the bulk-product sampler the multiplication order can be chosen with
`--bulk-product-order natural|reverse|balanced`.  See [`RESULTS.md`](../RESULTS.md)
for representative numbers on `4x4`, `6x6` and `8x8` lattices.

## Caveats

- The pseudofermion action is a stochastic estimator, so the Hamiltonian is
  noisy.  Smaller step sizes are usually required than with the exact
  determinant.
- With the book-style refresh the TDF pseudofermion force is cheaper per
  evaluation, but on the benchmark lattices it also requires a smaller step
  size than the standard force to reach the same acceptance rate.  The
  stochastic-bulk and bulk-product variants are smoother than the exact-bulk
  variant, but the bulk-product CG solve can be more ill-conditioned than the
  per-slice solves.  A fair comparison must therefore tune `dt` (and possibly
  `n_steps`) independently for each algorithm.
- The current implementation is not JIT-optimised across trajectories because
  the pseudofermion vector changes every trajectory.  Absolute timings should
  therefore be interpreted cautiously; the relevant comparison is the relative
  scaling and the stochastic noise.
