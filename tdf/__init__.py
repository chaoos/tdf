"""Temporal determinant factorization for the 2D Schwinger model."""

import jax

# Enable 64-bit floating point precision.  This must be done before any JAX
# arrays are materialised.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
