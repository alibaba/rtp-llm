"""Ascend operator implementations for Qwen3.5 (GDN/MoE kernels)."""

from rtp_llm.models_py.kernels.ascend.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.kernels.ascend.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.kernels.ascend.common import RmsNormGated, fused_gdn_gating
from rtp_llm.models_py.kernels.ascend.linear_attention import (
    chunk_fwd_o,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd_h,
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    l2norm_fwd,
    recompute_w_u_fwd,
    solve_tril,
)
from rtp_llm.models_py.kernels.ascend.recurrent import fused_recurrent_gated_delta_rule

# NOTE: w8a8_mx_layout (W8A8_MXFP8 weight & activation FP8 quantization) is
# imported directly by its consumers (model_loader / quant executors).

__all__ = [
    "CausalConv1dMetadata",
    "RmsNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_fwd_o",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt_fwd",
    "fused_gdn_gating",
    "fused_recurrent_gated_delta_rule",
    "l2norm_fwd",
    "load_initial_state_from_block_map",
    "prepare_causal_conv1d_metadata",
    "recompute_w_u_fwd",
    "solve_tril",
    "store_ssm_state_to_block_map",
]
