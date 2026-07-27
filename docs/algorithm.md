# Algorithm documentation

This page describes the lattice formulation, the Wilson–Dirac operator, and the
temporal determinant factorization used in this project.

## Lattice and gauge fields

The two-dimensional Euclidean lattice has `L` spatial sites and `L_t` temporal
sites.  A U(1) gauge field is stored as a set of link angles

```
theta[mu, t, x]   (mu = 0,1; t = 0,...,Lt-1; x = 0,...,L-1)
```

so that the link variable is `U_mu(t,x) = exp(i theta[mu,t,x])`.  Periodic
boundary conditions are used in both directions, except for the temporal
boundary of the fermion field which is anti-periodic by default.

The standard plaquette and gauge action are implemented in `tdf/lattice.py`:

```python
S_g[theta] = -beta * sum_{x,mu<nu} Re(plaquette)
```

## Wilson–Dirac operator

The massive Wilson–Dirac operator for a two-component Dirac spinor is

```
K[U, mu] = 1 - 2*kappa * sum_mu [ (1 - gamma_mu) U_mu(x) delta_{x+mu,x'}
                               + (1 + gamma_mu) U_mu^*(x-mu) delta_{x-mu,x'} ]
```

with the usual Wilson term.  In this code the operator is built in site-major
ordering in `tdf/dirac.py` and is also decomposed into time-slice blocks
`B_t`, `A^+_t`, `A^-_t`.

## Temporal determinant factorization

The time-slice decomposition allows the Wilson–Dirac matrix to be written as a
block-tridiagonal matrix in temporal index.  After eliminating the spatial
blocks, the full determinant reduces to a product of bulk determinants and a
small `2L × 2L` transfer-matrix determinant:

```
det K[U, mu] = (prod_t det R_t) * det(I - (-1)^{L_t} T)
```

where

```
T = T_0 T_1 ... T_{L_t-1},   T_t = R_t^{-1} S_t .
```

The matrices `R_t` and `S_t` are constructed in `tdf/reduced.py` from the
blocks `B_t`, `A^+_t`, `A^-_t`.  The factor `(-1)^{L_t}` accounts for the
anti-periodic temporal boundary condition.

## Canonical determinants

For a fixed quark number `k` (or, in the two-flavour context, a fixed isospin
sector) the canonical determinant is the coefficient of `z^{k+L}` in the
fugacity expansion

```
det(z I - (-1)^{L_t} T) = sum_{m=0}^{2L} c_m z^m .
```

Thus

```
det_k(K[U]) = (prod_t det R_t) * c_{k+L} .
```

The coefficients `c_m` are elementary symmetric polynomials in the eigenvalues
of `(-1)^{L_t} T` and are computed in `tdf/canonical.py`.

## Symmetries

- **γ5-hermiticity:** `gamma5 K[U, mu=0] gamma5 = K[U, mu=0]^dagger`.
- **Reality:** `det K[U, mu=0]` is real for anti-periodic temporal BCs.
- **Reflection symmetry:** `det_k^* = det_{-k}` at `mu = 0`.
- **Sum rule:** `sum_k det_k = det K[U, mu=0]`.
