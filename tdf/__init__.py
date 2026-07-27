"""Temporal determinant factorization for the 2D Schwinger model."""

import logging

import jax

# Enable 64-bit floating point precision.  This must be done before any JAX
# arrays are materialised.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"


def configure_logging(level=logging.INFO):
    """Configure the package logger with a sensible default format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
