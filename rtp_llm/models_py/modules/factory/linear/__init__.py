"""Linear factory module

Uses strategy pattern for creating Linear layers.
"""

from .factory import LinearFactory
from .linear_base import LinearBase

__all__ = ["LinearFactory", "LinearBase"]
