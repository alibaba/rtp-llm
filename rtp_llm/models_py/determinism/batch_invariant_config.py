"""Validate and apply startup settings for batch-invariant Qwen3 inference."""

from typing import TYPE_CHECKING

import torch

from rtp_llm.device.device_type import is_cuda
from rtp_llm.ops import DataType, KvCacheDataType, RoleType, SpeculativeType, TaskType
from rtp_llm.utils.weight_type import WEIGHT_TYPE

if TYPE_CHECKING:
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.config.model_config import ModelConfig


def is_qwen3_8b_dense(model: "ModelConfig") -> bool:
    return (
        model.model_type,
        model.num_layers,
        model.hidden_size,
        model.inter_size,
    ) == ("qwen_3", 36, 4096, 12288)


def apply_batch_invariant_config(engine: "EngineConfig", model: "ModelConfig") -> None:
    hw = engine.hw_kernel_config
    attn = model.attn_config
    is_moe = model.model_type == "qwen_3_moe"
    heads = (attn.head_num, attn.kv_head_num, attn.size_per_head)
    if is_moe:
        # The MoE loader uses inter_size for shared experts. The 30B has none.
        ffn = (model.inter_size, model.expert_num, model.moe_k, model.moe_inter_size)
        supported = (model.num_layers, model.hidden_size, heads, ffn) == (
            48,
            2048,
            (32, 4, 128),
            (0, 128, 8, 768),
        )
    else:
        supported = is_qwen3_8b_dense(model) and heads == (32, 8, 128)
    if not supported:
        raise ValueError(
            "RTP_BATCH_INVARIANT=1 requires Qwen3-8B Dense or Qwen3-30B-A3B"
        )

    parallel = engine.parallelism_config
    cp = parallel.prefill_cp_config
    cache = engine.kv_cache_config
    conflicts = []
    if not is_cuda() or torch.cuda.get_device_capability() != (10, 3):
        conflicts.append("requires an SM103 CUDA GPU")
    tp = parallel.tp_size
    if (
        tp not in (1, 2, 4)
        or (parallel.world_size, parallel.local_world_size, parallel.ffn_tp_size)
        != (tp, tp, tp)
        or (parallel.dp_size, parallel.ep_size, parallel.pp_size) != (1, 1, 1)
        or cp.is_enabled()
        or cp.is_prefill_enabled()
        or parallel.role_type != RoleType.PDFUSION
    ):
        conflicts.append(
            "requires single-node PDFUSION TP1, TP2 or TP4 with "
            "world_size=local_world_size=ffn_tp_size=tp_size, "
            "DP=EP=PP=1 and context parallelism disabled"
        )
    try:
        checkpoint_dtype = WEIGHT_TYPE.from_str(str(model.config_dtype))
    except ValueError:
        checkpoint_dtype = None
    if (
        checkpoint_dtype is not WEIGHT_TYPE.BF16
        or model.data_type != DataType.TYPE_BF16
        or model.quant_config is not None
    ):
        conflicts.append("requires unquantized BF16 weights and BF16 activations")
    if model.task_type != TaskType.LANGUAGE_MODEL or model.lora_infos:
        conflicts.append("requires text generation with LoRA disabled")
    if getattr(model, "headwise_config", None):
        conflicts.append("does not support HeadWise attention (headwise_config)")
    if hw.enable_cuda_graph and hw.generation_prefill_capture_token_buckets:
        conflicts.append(
            "supports decode CUDA Graph only. Leave GENERATION_PREFILL_CAPTURE_CONFIG empty"
        )
    if engine.sp_config.type != SpeculativeType.NONE:
        conflicts.append("requires speculative decoding to be disabled")
    if engine.runtime_config.use_batch_decode_scheduler:
        conflicts.append("requires USE_BATCH_DECODE_SCHEDULER to be disabled")
    page_sizes = (attn.tokens_per_block, attn.kernel_tokens_per_block)
    if attn.kv_cache_dtype != KvCacheDataType.BASE or page_sizes != (64, 64):
        conflicts.append("requires BF16 KV cache with 64-token pages")
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
        if (
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

    hw.enable_dynamic_decode_backend = False
    hw.enable_decode_cascade_attention = False
    # The invariant projection accumulates BF16 products directly into FP32 logits.
    model.enable_fp32_lm_head = False
    if is_moe:
        # Select the BF16 backend that implements fixed-order expert projections.
        engine.moe_config.moe_strategy = "no_quant_cpp"
