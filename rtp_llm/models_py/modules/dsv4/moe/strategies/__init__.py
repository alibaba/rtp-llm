"""Hopper grouped-FP8 kernel implementation. Selection uses the MoE factory."""

from .base import MoeCfg
from .grouped_fp8 import GroupedFP8Strategy, _has_grouped_fp8_kernel
