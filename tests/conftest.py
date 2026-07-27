"""Pytest configuration."""

import os

# Do not pre-allocate the entire GPU memory pool; the RTX 3050 has only 6 GB.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
