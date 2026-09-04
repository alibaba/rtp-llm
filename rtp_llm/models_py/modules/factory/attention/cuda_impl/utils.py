"""Shared utilities for CUDA attention implementations."""

import torch

from rtp_llm.ops import AttentionConfigs


def has_asymmetric_kv_head_dim(attn_configs: AttentionConfigs) -> bool:
    """True when V's head dimension differs from QK's (MiMo V2.5: 192 / 128).

    Implementations that assume one head dimension for both call this from ``support()``
    to exclude themselves. The predicate deliberately reads only ``v_size_per_head`` --
    a plain shape fact -- and not model markers like ``sliding_window`` or
    ``add_sink_bias``: DeepSeek-V4 also sets ``sliding_window`` while routing through MHA,
    so gating on those would change which implementation it selects.
    """
    return bool(
        attn_configs.v_size_per_head
        and attn_configs.v_size_per_head != attn_configs.size_per_head
    )


def is_cuda_12_9_or_later() -> bool:
    if not torch.version.cuda:
        return False
    try:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (12, 9)
