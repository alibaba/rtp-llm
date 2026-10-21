"""Compatibility imports for the shared MoE CUDA-graph warmup helpers."""

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
    sync_cuda_graph_warmup_ranks,
)
