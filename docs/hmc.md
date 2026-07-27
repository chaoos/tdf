# HMC samplers

The package provides two HMC samplers for the two-flavour Schwinger model.

## Standard Nf=2 HMC

The reference sampler is in `tdf/hmc.py`.  It targets the grand-canonical
2-flavour Boltzmann weight

```
exp(-S[U]) = exp(-S_g[U]) * |det K[U, mu]|^2
```

with action

```
S[U] = S_g[U] - 2 log |det K[U, mu]| .
```

The fermion force is obtained by automatic differentiation through the full
Wilson–Dirac determinant.

### Usage

```python
from jax import random
from tdf import lattice, dirac, hmc

key = random.PRNGKey(42)
theta = lattice.make_gauge_field(L=6, Lt=6, key=key)
kappa = dirac.kappa_from_mass(0.0)

def action(t):
    return hmc.action_standard(t, beta=5.0, mu=0.0, kappa=kappa)

force = hmc.make_force(action)
history, configs = hmc.run_hmc(
    key, theta, action, force,
    n_therm=50, n_measure=100, n_skip=5, dt=0.1, n_steps=10
)
```

Or from the command line:

```bash
.venv/bin/python scripts/run_hmc_standard.py --L 6 --Lt 6 --beta 5.0 \
    --mass 0.0 --n-therm 50 --n-measure 100 --n-skip 5 --n-steps 10
```

## TDF canonical HMC

The canonical sampler is in `tdf/hmc_canonical.py`.  It targets a fixed
quark-number sector `n`:

```
exp(-S_n[U]) = exp(-S_g[U]) * |det_n(K[U, mu])|^2
```

with action

```
S_n[U] = S_g[U] - 2 log |det_n(K[U, mu])| .
```

The sector index satisfies `-L <= n <= L`.  The force is obtained by
automatic differentiation through the transfer-matrix eigenvalues, so the
cost of a force evaluation is essentially independent of `L_t`.

### Usage

```python
from tdf.hmc_canonical import run_hmc_canonical

history, configs = run_hmc_canonical(
    key, theta, beta=5.0, n=3, mu=0.0, kappa=kappa,
    n_therm=50, n_measure=100, n_skip=5, dt=0.1, n_steps=10
)
```

Or from the command line:

```bash
.venv/bin/python scripts/run_hmc_canonical.py --L 6 --Lt 6 --n 3 --beta 5.0 \
    --mass 0.0 --n-therm 50 --n-measure 100 --n-skip 5 --n-steps 10
```

## Tuning guidelines

- **Trajectory length:** `dt * n_steps` should be of order 1.  A common starting
  point is `dt = 0.1`, `n_steps = 10`.
- **Acceptance:** Aim for an acceptance rate of 0.6–0.8.  Decrease `dt` or
  increase `n_steps` if acceptance is too low.
- **Thermalization:** Discard enough trajectories so that observables have
  stabilised.  For small lattices 20–50 trajectories are usually sufficient.
- **Measurements:** Separate measurements by `n_skip` trajectories.  Choose
  `n_skip` so that consecutive measurements are roughly independent.
