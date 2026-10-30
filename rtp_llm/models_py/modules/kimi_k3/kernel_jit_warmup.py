"""Directed K3 JIT preparation after the model's cache resources are bound."""

from __future__ import annotations

import torch

from rtp_llm.utils.warmup import model_warm_up_enabled


def warmup_kimi_k3_kernel_jit(model, init_resource) -> None:
    if not model_warm_up_enabled() or not torch.cuda.is_available():
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("K3 kernel JIT warmup cannot run during CUDA Graph capture")
    from rtp_llm.models_py.modules.kimi_k3.bounded_projection_jit_warmup import (
        warmup_kimi_k3_projection_jit,
    )
    from rtp_llm.models_py.modules.kimi_k3.mla_jit_warmup import warmup_kimi_k3_mla_jit

    warmup_kimi_k3_projection_jit(model, init_resource)
    warmup_kimi_k3_mla_jit(model, init_resource)
    if getattr(model, "dsa_workspace", None) is not None:
        from rtp_llm.models_py.modules.kimi_k3.dsa_jit_warmup import (
            warmup_kimi_k31_dsa_jit,
        )

        warmup_kimi_k31_dsa_jit(model, init_resource)
