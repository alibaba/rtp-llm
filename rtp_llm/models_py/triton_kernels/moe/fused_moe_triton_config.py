# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py
# Adapted for RTP-LLM. We currently ship only the default config heuristic
# (no per-shape JSON tuning files); ``try_get_optimal_moe_config`` falls back
# to ``get_default_config`` directly. Tuned JSONs can be added later under
# ``configs/`` and wired via ``get_moe_configs``.
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                }
        else:
            # Block-wise fp8: BLOCK_SIZE_K must be divisible by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        if M <= E:
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    E, _, N = w2_shape
    return get_default_config(
        M, E, N, w1_shape[2], top_k, dtype, block_shape=block_shape
    )


def get_config_dtype_str(
    dtype: torch.dtype,
    use_fp8_w8a8: Optional[bool] = False,
) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    if dtype == torch.float:
        return "float32"
    return None
