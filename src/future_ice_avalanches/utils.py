"""Utility functions for the future_ice_avalanches package."""

import random

import numpy as np


def seed_all(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy's legacy random state (used by sklearn and other libraries)
    - NumPy's default RNG (note: this affects the global default RNG,
      but new generators created with np.random.default_rng() will use
      their own state unless explicitly seeded)

    Args:
        seed (int): Seed value to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
