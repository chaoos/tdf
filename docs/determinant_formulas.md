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

For each lattice size the comparison script prints:

- `S_exact = log |det K|^2` from the full matrix,
- `S_TDF = log |bulk|^2 + 2 log |det M|` from the reduced determinant,
- statistics of `S_pf^std[U, φ]` over many Gaussian samples `φ`,
- statistics of `S_pf^TDF[U, ψ]` over many Gaussian samples `ψ`.

The TDF determinant should agree with the exact determinant to machine
precision.  The pseudofermion actions have large mean values (dominated by the
Gaussian normalisation), so the means are not directly compared to
`S_exact`.  Instead, the standard deviations `σ(S_pf)` are reported as a
measure of stochastic noise.

## See also

- [`algorithm.md`](algorithm.md) — derivation of the TDF reduced determinant.
- [`pseudofermion.md`](pseudofermion.md) — how the pseudofermion HMC samplers
  use these actions in a Markov chain.
- [`RESULTS.md`](../RESULTS.md) — numerical output of the comparison.
