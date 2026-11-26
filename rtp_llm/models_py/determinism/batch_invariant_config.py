"""Validate and apply startup settings for batch-invariant Qwen inference."""

from typing import TYPE_CHECKING

import torch

from rtp_llm.device.device_type import is_cuda
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    KvCacheDataType,
    RoleType,
    SpeculativeType,
    TaskType,
    VitSeparation,
)
from rtp_llm.utils.weight_type import WEIGHT_TYPE

if TYPE_CHECKING:
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.config.model_config import ModelConfig


def _check_m11(engine: "EngineConfig", model: "ModelConfig", conflicts: list) -> None:
    attn = model.attn_config
    # Use the geometry already parsed by the Qwen4 loader, including layer order.
    cycle = [HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE]
    if (
        not getattr(model, "_qwen4_mxfp4_source", False)
        or model.quantization
        or not (attn.is_sparse and attn.use_sparse_gqa_fmha)
        or list(model.hybrid_attention_config.hybrid_attention_types) != cycle * 15
        or model.hc_mult != 4
        or model._qwen4_ple_layer_ids != [2]
        or (attn.indexer_head_num, attn.indexer_head_dim, model._qwen4_indexer_kv_heads)
        != (8, 128, 1)
        or (model._qwen4_indexer_budget, model._qwen4_indexer_compress_ratio)
        != (2048, 4)
    ):
        conflicts.append(
            "requires the native Qwen4-Max M11 MXFP4 GDN/QSA/PLE checkpoint"
        )
    if model.linear_attention_config.ssm_state_dtype != DataType.TYPE_FP32:
        conflicts.append("M11 requires FP32 SSM state")
    if model.vit_separation != VitSeparation.VIT_SEPARATION_REMOTE:
        conflicts.append("M11 requires a text worker with vit_separation=2")
    moe = engine.moe_config
    if moe.moe_strategy != "mega_moe" or moe.use_deepep_moe or moe.use_mori_ep:
        conflicts.append("M11 requires native mega_moe with DeepEP and MoriEP disabled")


def apply_batch_invariant_config(engine: "EngineConfig", model: "ModelConfig") -> None:
    hw = engine.hw_kernel_config
    attn = model.attn_config
    is_m11 = model.model_type == "qwen4_exp"
    is_moe = model.model_type == "qwen_3_moe" or is_m11
    # inter_size is the width of the shared expert for these MoE loaders.
    ffn = (
        (model.inter_size, model.expert_num, model.moe_k, model.moe_inter_size)
        if is_moe
        else (model.inter_size,)
    )
    shape = (
        model.model_type,
        model.num_layers,
        model.hidden_size,
        attn.head_num,
        attn.kv_head_num,
        attn.size_per_head,
        ffn,
    )
    if shape not in (
        ("qwen_3", 36, 4096, 32, 8, 128, (12288,)),
        ("qwen_3_moe", 48, 2048, 32, 4, 128, (0, 128, 8, 768)),
        ("qwen4_exp", 60, 8192, 64, 4, 256, (2048, 768, 10, 2048)),
    ):
        raise ValueError(
            "RTP_BATCH_INVARIANT=1 requires Qwen3-8B Dense, Qwen3-30B-A3B or Qwen4-Max M11"
        )

    parallel = engine.parallelism_config
    cp = parallel.prefill_cp_config
    cache = engine.kv_cache_config
    conflicts = []
    if not is_cuda() or torch.cuda.get_device_capability() != (10, 3):
        conflicts.append("requires an SM103 CUDA GPU")
    tp = parallel.tp_size
    if is_m11:
        _check_m11(engine, model, conflicts)
        valid_tp = tp == 1
        world = dp = ep = 8
        layout = "single-node PDFUSION TP1/DP8/EP8 with world_size=local_world_size=8"
    else:
        valid_tp = tp in (1, 2, 4)
        world = tp
        dp = ep = 1
        layout = (
            "single-node PDFUSION TP1, TP2 or TP4 with "
            "world_size=local_world_size=ffn_tp_size=tp_size, DP=EP=PP=1"
        )
    if (
        not valid_tp
        or (parallel.world_size, parallel.local_world_size, parallel.ffn_tp_size)
        != (world, world, tp)
        or (parallel.dp_size, parallel.ep_size, parallel.pp_size) != (dp, ep, 1)
        or cp.is_enabled()
        or cp.is_prefill_enabled()
        or parallel.role_type != RoleType.PDFUSION
    ):
        conflicts.append(f"requires {layout} and context parallelism disabled")
    try:
        checkpoint_dtype = WEIGHT_TYPE.from_str(str(model.config_dtype))
    except ValueError:
        checkpoint_dtype = None
    if (
        checkpoint_dtype is not WEIGHT_TYPE.BF16
        or model.data_type != DataType.TYPE_BF16
        or model.quant_config is not None
    ):
        conflicts.append(
            "M11 requires BF16 dense weights and activations with native MXFP4 experts"
            if is_m11
            else "requires unquantized BF16 weights and BF16 activations"
        )
    if model.task_type != TaskType.LANGUAGE_MODEL or model.lora_infos:
        conflicts.append("requires text generation with LoRA disabled")
    if getattr(model, "headwise_config", None):
        conflicts.append("does not support HeadWise attention (headwise_config)")
    if (
        hw.enable_cuda_graph
        or engine.sp_config.type != SpeculativeType.NONE
        or engine.runtime_config.use_batch_decode_scheduler
    ):
        conflicts.append(
            "requires eager execution without speculative decoding or USE_BATCH_DECODE_SCHEDULER"
        )
    page_sizes = (attn.tokens_per_block, attn.kernel_tokens_per_block)
    page_size = 128 if is_m11 else 64
    if attn.kv_cache_dtype != KvCacheDataType.BASE or page_sizes != (
        page_size,
        page_size,
    ):
        conflicts.append(f"requires BF16 KV cache with {page_size}-token pages")
    if (
        cache.reuse_cache
        or cache.enable_remote_cache
        or cache.enable_disk_cache
        or cache.enable_memory_cache
        or cache.multi_task_prompt
        or cache.multi_task_prompt_str
        or cache.multi_task_prompt_tokens
    ):
        conflicts.append(
            "requires prefix reuse, task prompts and external KV caches to be disabled"
        )
    if is_moe:
        moe = engine.moe_config
        if not is_m11 and (
            moe.moe_strategy not in ("auto", "no_quant_cpp")
            or not moe.use_all_gather
            or moe.use_deepep_moe
            or moe.use_mori_ep
        ):
            conflicts.append(
                "requires the local BF16 MoE strategy with USE_ALL_GATHER enabled "
                "and DeepEP and MoriEP disabled"
            )
        if (
            moe.fake_balance_expert
            or model.eplb_config.enable_eplb()
            or model.eplb_config.redundant_expert
            or model.phy2log_path
        ):
            conflicts.append(
                "requires the original expert ordering without EPLB or fake balancing"
            )
    if conflicts:
        raise ValueError("RTP_BATCH_INVARIANT=1:\n" + "\n".join(conflicts))

    hw.enable_decode_cascade_attention = False
    # The invariant projection accumulates BF16 products directly into FP32 logits.
    model.enable_fp32_lm_head = False
    if is_m11:
        model.mm_model_config.is_multimodal = False
    elif is_moe:
        # Select the BF16 backend that implements fixed-order expert projections.
        engine.moe_config.moe_strategy = "no_quant_cpp"
