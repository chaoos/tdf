# Determinant formulas used in the comparison

This page collects the exact formulas that are evaluated in the unified
determinant comparison (`scripts/run_determinant_comparison.py`).  All three
methods target the same 2-flavour Schwinger-model weight

```
exp(-S) = exp(-S_g[U]) * |det K[U, μ = 0]|^2 .
```

The comparison fixes a gauge configuration `U`, ignores the common gauge action
`S_g[U]`, and compares only the fermion determinant factor.

## 1. Exact determinant

The Wilson–Dirac operator for one flavour is denoted `K[U, μ]` with
`μ = 0`.  In the 2-flavour, mass-degenerate case with isospin chemical
potential zero the fermion weight is

```
|det K[U, 0]|^2 = det(K[U, 0]) * det(K[U, 0])^*
                = det( K[U, 0] K[U, 0]^† ) .
```

The exact value used in the comparison is therefore

```
S_exact[U] = log |det K[U, 0]|^2
           = 2 * Re log det K[U, 0]
           = 2 * log |det K[U, 0]| .
```

In code this is computed with a stable matrix factorisation:

```python
log_det_exact = 2.0 * jnp.linalg.slogdet(K)[1]
```

This is the reference value against which the TDF and pseudofermion
estimators are judged.

## 2. Standard pseudofermion estimator

The Gaussian integral identity for a positive-definite matrix `A` is

```
det A = ∫ Dφ Dφ† exp(- φ† A^{-1} φ) ,
```

with the normalised complex Gaussian measure
`Dφ Dφ† = ∏_i (dφ_i dφ_i^* / π)` and `φ` drawn from `N(0, I)`.

Setting `A = K K†` gives

```
|det K|^2 = det(K K†)
          = ∫ Dφ Dφ† exp(- φ† (K K†)^{-1} φ) .
```

For a fixed gauge field the pseudofermion action is

```
S_pf^std[U, φ] = φ† (K[U] K[U]^†)^{-1} φ .
```

The field `φ` is a complex Gaussian vector of length `2 L L_t` (the full Dirac
space).  In the joint distribution over `U` and `φ`,

```
P(U, φ) ∝ exp(-S_g[U]) exp(-S_pf^std[U, φ]) ,
```

the marginal distribution of `U` is the desired one with weight
`|det K[U]|^2`.  For a fixed `U`, `S_pf^std[U, φ]` is a random variable.  Its
expectation value is `tr((K K†)^{-1})`, which differs from `S_exact[U]` by a
large, dimension-dependent constant.  The physically relevant quantity is the
width of its distribution, which measures the stochastic noise of the
estimator.

Computationally, each evaluation requires one Conjugate-Gradient solve of

```
(K K†) χ = φ .
```

The action is then `S_pf^std = φ† χ`.

## 3. TDF pseudofermion estimator

The temporal determinant factorisation rewrites the single-flavour determinant
as

```
det K[U, 0] = (prod_{t=0}^{L_t-1} det R_t) * det( I - (-1)^{L_t} T ) .
```

The first factor is called the **bulk determinant**,

```
bulk[U] = prod_t det R_t .
```

The second factor contains the cyclic product of transfer matrices,

```
T = T_0 T_1 ... T_{L_t-1},    T_t = R_t^{-1} S_t .
```

Define the transfer-matrix factor

```
M[U] = I - (-1)^{L_t} T[U] .
```

Then

```
|det K[U, 0]|^2 = |bulk[U]|^2 * |det M[U]|^2 .
```

The TDF pseudofermion estimator keeps the bulk determinant exact and uses
pseudofermions only for `|det M|^2`.  Applying the same Gaussian identity with
`A = M M†` gives

```
|det M|^2 = ∫ Dψ Dψ† exp(- ψ† (M M†)^{-1} ψ) .
```

The TDF pseudofermion action is therefore

```
S_pf^TDF[U, ψ] = ψ† (M[U] M[U]^†)^{-1} ψ - 2 log |bulk[U]| .
```

The pseudofermion vector `ψ` now lives in the much smaller transfer-matrix
space of dimension `2 L`.  Each evaluation requires one CG solve of

```
(M M†) χ = ψ ,
```

followed by `S_pf^TDF = ψ† χ - 2 log |bulk|`.

Because `M` is `2L × 2L` while `K` is `2 L L_t × 2 L L_t`, the TDF solve is
smaller and the estimator noise is much lower.  The bulk determinant is a
product of `L_t` small `2L × 2L` determinants and is evaluated exactly.

## 4. What the comparison reports

For each lattice size the comparison script prints three determinant values:

1. **Exact determinant**
   ```
   |det K|² = exp(2 log |det K|)
   ```
   computed from the full Wilson–Dirac matrix.

2. **Standard pseudofermion estimate**
   The pseudofermion field `η` is drawn from the standard complex Gaussian
   `N(0, I)`.  For a positive-definite matrix `M` the identity (Gattringer &
   Lang, *Quantum Chromodynamics on the Lattice*, eq. (8.63))
   ```
   det M = < exp( -η† (M^{-1} - I) η ) >
   ```
   holds.  Writing the pseudofermion action as `S_pf = η† M^{-1} η`, the
   per-sample weight is therefore
   ```
   w = exp(-S_pf + |η|²) .
   ```
   The reported value is the sample mean `(1/N) Σ_i w_i`, and the error bar is
   the standard error of the mean, `σ_mean = std(w) / sqrt(N)`.

3. **TDF pseudofermion estimate**
   The same weight is used with `M = M_TDF M_TDF†`, multiplied by the exact
   bulk factor `|bulk|²`:
   ```
   w_TDF = |bulk|² exp(-S_pf^TDF + |η|²) .
   ```

The naive weight `exp(-S_pf)` is **biased**: it corresponds to the flat-measure
identity used in HMC, not to the normalized Gaussian refresh used here.  The
corrected weight `exp(-S_pf + |η|²)` removes that bias and is unbiased in
expectation.

Because the matrices involved have eigenvalues larger than 2, the variance of
the corrected estimator is formally infinite and, in practice, enormous.  A
modest sample count therefore gives honest but wide error bars, and the sample
mean often does not yet bracket the exact determinant.  The comparison reports,
for each estimator, whether `|mean − exact| ≤ σ_mean`; with modest `N` this
1-sigma check usually fails.

The script also reports the raw pseudofermion action statistics (`<S_pf>` and
`σ(S_pf)`).  Those means are dominated by the Gaussian normalisation and by the
bulk factor, so they are not compared directly.  Their widths measure the
stochastic noise of the respective estimators and show the real advantage of
the TDF formulation: the TDF action distribution is much narrower and grows
more slowly with the lattice size.

## See also

- [`algorithm.md`](algorithm.md) — derivation of the TDF reduced determinant.
- [`pseudofermion.md`](pseudofermion.md) — how the pseudofermion HMC samplers
  use these actions in a Markov chain.
- [`RESULTS.md`](../RESULTS.md) — numerical output of the comparison.
