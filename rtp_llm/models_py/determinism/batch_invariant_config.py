"""Validate and apply startup settings for batch-invariant Qwen3 inference."""

from typing import TYPE_CHECKING

import torch

from rtp_llm.device.device_type import is_cuda
from rtp_llm.ops import (
    DataType,
    KvCacheDataType,
    RoleType,
    SpeculativeType,
    TaskType,
    VitSeparation,
)
from rtp_llm.utils.util import get_config_from_path
from rtp_llm.utils.weight_type import WEIGHT_TYPE

if TYPE_CHECKING:
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.config.model_config import ModelConfig


def _supported_topologies(model: "ModelConfig") -> tuple[tuple[int, int, int], ...]:
    """Return supported (TP, DP, EP) layouts for the unsharded checkpoint."""
    attn = model.attn_config
    ffn = (
        (model.inter_size, model.expert_num, model.moe_k, model.moe_inter_size)
        if model.model_type in ("qwen_3_moe", "qwen35_moe")
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
    if shape in (
        ("qwen_3", 36, 4096, 32, 8, 128, (12288,)),
        ("qwen_3_moe", 48, 2048, 32, 4, 128, (0, 128, 8, 768)),
    ):
        return ((1, 1, 1), (2, 1, 1), (4, 1, 1))
    if shape == ("qwen35_moe", 40, 2048, 16, 2, 256, (512, 256, 8, 512)):
        return ((1, 1, 1), (1, 4, 4))
    if shape == ("qwen35_moe", 60, 4096, 32, 2, 256, (1024, 512, 10, 1024)):
        return ((4, 1, 1), (4, 1, 4))
    raise ValueError(
        "RTP_BATCH_INVARIANT=1 requires Qwen3-8B Dense, Qwen3-30B-A3B, "
        "Qwen3.6-35B-A3B or Qwen3.5-397B-A17B"
    )


def apply_batch_invariant_config(engine: "EngineConfig", model: "ModelConfig") -> None:
    hw = engine.hw_kernel_config
    attn = model.attn_config
    layouts = _supported_topologies(model)
    hybrid = model.model_type == "qwen35_moe"
    is_moe = hybrid or model.model_type == "qwen_3_moe"
    parallel = engine.parallelism_config
    cp = parallel.prefill_cp_config
    cache = engine.kv_cache_config
    conflicts = []
    if not is_cuda() or torch.cuda.get_device_capability() != (10, 3):
        conflicts.append("requires an SM103 CUDA GPU")
    tp = parallel.tp_size
    if (
        (tp, parallel.dp_size, parallel.ep_size) not in layouts
        or parallel.local_world_size != parallel.world_size
        or parallel.world_size != tp * parallel.dp_size
        or parallel.ffn_tp_size != tp
        or parallel.pp_size != 1
        or cp.is_enabled()
        or cp.is_prefill_enabled()
        or parallel.role_type != RoleType.PDFUSION
    ):
        layout_description = (
            f"TP/DP/EP in {layouts}" if hybrid else "TP1, TP2 or TP4 with DP=EP=PP=1"
        )
        conflicts.append(
            f"requires single-node PDFUSION {layout_description}, FFN TP=TP, "
            "PP=1 and context parallelism disabled"
        )
    dtype = model.config_dtype
    if hybrid:
        # The official multimodal checkpoints declare the text dtype separately.
        config = get_config_from_path(model.ckpt_path) or {}
        text_config = config.get("text_config") or {}
        dtype = text_config.get("dtype") or text_config.get("torch_dtype")
        if model.vit_separation != VitSeparation.VIT_SEPARATION_LOCAL:
            conflicts.append(
                "requires colocated text generation without a separate ViT"
            )
        if model.linear_attention_config.ssm_state_dtype != DataType.TYPE_FP32:
            conflicts.append("requires FP32 SSM state")
    try:
        checkpoint_dtype = WEIGHT_TYPE.from_str(str(dtype))
    except ValueError:
        checkpoint_dtype = None
    quant = model.quant_config
    fp8_block = (
        hybrid
        and quant is not None
        and quant.get_method() == "FP8_PER_BLOCK"
        and quant.bits == 8
        and quant.group_size() == 128
    )
    valid_dtype = (
        checkpoint_dtype is WEIGHT_TYPE.BF16 and model.data_type == DataType.TYPE_BF16
    )
    if hybrid:
        if not valid_dtype:
            conflicts.append("requires a BF16 text checkpoint and BF16 activations")
        if quant is not None and not fp8_block:
            conflicts.append(
                "only FP8_PER_BLOCK with 128x128 weight blocks is supported"
            )
    elif not valid_dtype or quant is not None:
        conflicts.append("requires unquantized BF16 weights and BF16 activations")
    if model.task_type != TaskType.LANGUAGE_MODEL or model.lora_infos:
        conflicts.append("requires text generation with LoRA disabled")
    if getattr(model, "headwise_config", None):
        conflicts.append("does not support HeadWise attention (headwise_config)")
    if hw.enable_cuda_graph and not hybrid:
        conflicts.append("requires eager execution for standard Qwen3")
    if (
        engine.sp_config.type != SpeculativeType.NONE
        or engine.runtime_config.use_batch_decode_scheduler
    ):
        conflicts.append(
            "requires speculative decoding and USE_BATCH_DECODE_SCHEDULER to be disabled"
        )
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
        if parallel.dp_size > 1 and hybrid:
            # The DP/EP router gathers rows and combines expert outputs in rank order.
            strategy, alias = (
                ("fp8_per_block_pure_dp", "fp8_per_block_ep_normal")
                if fp8_block
                else ("no_quant_pure_dp", "no_auant_dp_normal")
            )
            if moe.moe_strategy not in ("auto", strategy, alias) or moe.use_mori_ep:
                conflicts.append(f"requires the {strategy} all-gather DP/EP route")
        else:
            strategy = "fp8_per_block_no_dp" if fp8_block else "no_quant_cpp"
            if (
                moe.moe_strategy not in ("auto", strategy)
                or not moe.use_all_gather
                or moe.use_deepep_moe
                or moe.use_mori_ep
            ):
                backend = "local DeepGEMM MoE" if fp8_block else "local BF16 MoE"
                conflicts.append(
                    f"requires the {backend} strategy with USE_ALL_GATHER enabled "
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
    if hybrid:
        model.mm_model_config.is_multimodal = False
    if is_moe:
        moe.moe_strategy = strategy
        if parallel.dp_size > 1:
            moe.use_all_gather = True
            moe.use_deepep_moe = False
            moe.use_deepep_low_latency = False
            moe.use_deepep_internode = False
