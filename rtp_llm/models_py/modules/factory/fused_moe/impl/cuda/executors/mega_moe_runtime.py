"""Checkpoint preparation and chunking for generic MegaMoE model callers.

Execution, graph resources and kernel launches belong to MegaMoeExecutor.
DeepSeek's chunked model layer and generic models share the same chunk runner.
"""

import copy

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    moe_decode_forward,
    resolve_moe_max_tokens_per_rank,
    run_moe_chunks,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


def setup_prepared_mega_moe(executor, weights):
    """Load a generic model's checkpoint into the common executor lifecycle."""
    cfg = executor.cfg = copy.copy(executor.cfg)
    model = cfg.model_config
    parallelism = cfg.parallelism_config
    policy = cfg.mega_moe_policy
    if policy.w1_layout not in ("up_gate", "gate_up"):
        raise ValueError(f"unsupported W1 layout: {policy.w1_layout}")
    cfg.moe_w1_layout = policy.w1_layout
    executor._is_decode_role = parallelism.role_type == RoleType.DECODE
    cp_size = 1
    if not executor._is_decode_role and parallelism.prefill_cp_config.is_enabled():
        cp_size = max(int(parallelism.tp_size), 1)
    max_seq_len = int(model.max_seq_len or 0)
    capacity = max(max_seq_len, 128) if max_seq_len > 0 else 8192
    gen_num_per_cycle = int(model.gen_num_per_cycle or 0)
    cfg.max_tokens_per_rank = resolve_moe_max_tokens_per_rank(
        max_seq_len=max_seq_len,
        current_max_tokens_per_rank=capacity,
        cp_size=cp_size,
        max_generate_batch_size=max(int(cfg.max_generate_batch_size), 1),
        is_decode_role=executor._is_decode_role,
        is_speculative=gen_num_per_cycle > 0,
        gen_num_per_cycle=gen_num_per_cycle,
    )
    if parallelism.role_type == RoleType.PDFUSION:
        # Only prefill can be chunked. Colocated decode/verify must fit the
        # complete scheduler batch even when it exceeds the prefill chunk size.
        cfg.max_tokens_per_rank = max(
            cfg.max_tokens_per_rank,
            int(cfg.moe_config.ll_num_max_token or 0),
            max(int(cfg.max_generate_batch_size), 1) * max(gen_num_per_cycle + 1, 1),
        )
    cfg.warmup_include_capacity = True

    keys = (W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2)
    if any(weights.get(key) is None for key in keys):
        raise ValueError("MegaMoE requires load-time FP4 or FP8 weights and scales")
    w1, s1, w2, s2 = (weights.pop(key) for key in keys)
    if w1.dtype == torch.int8 and w2.dtype == torch.int8:
        executor.setup_weights_from_fp4(w1, s1, w2, s2, w1_layout=policy.w1_layout)
    elif w1.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn:
        first_w, second_w = w1.chunk(2, dim=1)
        first_s, second_s = s1.chunk(2, dim=1)
        if policy.w1_layout == "up_gate":
            up, gate, up_s, gate_s = first_w, second_w, first_s, second_s
        else:
            gate, up, gate_s, up_s = first_w, second_w, first_s, second_s
        executor.setup_weights_from_fp8(
            w1_fp8=gate,
            w1_scale=gate_s,
            w2_fp8=w2,
            w2_scale=s2,
            w3_fp8=up,
            w3_scale=up_s,
        )
    else:
        raise ValueError(
            "MegaMoE requires FP4 int8 or FP8 e4m3 checkpoint weights, "
            f"got {w1.dtype} and {w2.dtype}"
        )
    del w1, s1, w2, s2
    if executor.includes_shared_expert:
        if parallelism.get_ffn_tp_size() != 1:
            raise ValueError("MegaMoE shared experts require full weights on each rank")
        if weights.get(W.shared_expert_gate) is not None:
            raise ValueError("Fused MegaMoE does not support shared_expert_gate")
        keys = (W.ffn_w13, W.ffn_s13, W.ffn_w2, W.ffn_s2)
        if any(weights.get(key) is None for key in keys):
            raise ValueError(
                "Fused MegaMoE requires FP8 shared-expert weights and scales"
            )
        w1, s1, w2, s2 = (weights.pop(key) for key in keys)
        executor.setup_shared_expert_from_fp8(w1_w=w1, w1_s=s1, w2_w=w2, w2_s=s2)
        del w1, s1, w2, s2
        executor.maybe_warmup_fused_shared_jit_once()
    torch.cuda.empty_cache()


def execute_prepared_mega_moe(executor, payload, execute, *args):
    """Apply the common EP chunk plan after a generic model has routed tokens."""

    def call_chunk(start, end, out):
        chunk = copy.copy(payload)
        chunk.expert_x = payload.expert_x[start:end]
        if payload.expert_topk_weights is not None:
            chunk.expert_topk_weights = payload.expert_topk_weights[start:end]
        if payload.expert_topk_ids is not None:
            chunk.expert_topk_ids = payload.expert_topk_ids[start:end]
        if payload.gate_payload is not None:
            raise ValueError("Prepared MegaMoE requires materialized top-k routing")
        return execute(chunk, *args).fused_expert_output

    output = run_moe_chunks(
        payload.expert_x,
        executor._mega_buf.num_max_tokens_per_rank,
        call_chunk,
        is_decode_forward=executor._is_decode_role or moe_decode_forward(),
    )
    return CombineForwardPayload(fused_expert_output=output)
