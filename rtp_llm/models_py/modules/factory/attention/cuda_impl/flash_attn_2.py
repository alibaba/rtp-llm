"""FlashAttention-2 non-paged prefill for sm120 (Blackwell Pro GPUs).

flashinfer 0.6.9's fmha_v2 has no TMA / warp-specialized variant for sm120
(warp specialization is gated to sm90 in both the JIT generator and the
runtime launcher), so prefill attention on RTX PRO 5000 runs the
Ampere-generation tiled kernel. Measured on d1-3B 31K prefill (H64 GQA 32:8
causal, bf16): 52 TF on sm120 vs 138 TF on sm90's tma_ws variant and 942 TF
on sm103a's trtllm-gen fmha. flash-attn 2.8.3 (pinned dependency) ships
tuned kernels for the same varlen GQA causal shapes on sm120 since
build 2.8.4 (the pinned 2.8.3.post1 r1 wheel shipped only sm_80/sm_90
cubins with no PTX, so sm_120 raised cudaErrorNoKernelImageForDevice);
this impl reuses flash-attn 2.8.4+ for the eager non-prefix prefill
path.

The generation-prefill CUDA-graph backend keeps selecting fmha_v2 on
sm12x (its graph support contract is covered by dedicated tests), so this
impl is eager-only: support() rejects is_cuda_graph and the class exposes
no prepare_cuda_graph.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    LayerKVCache,
    PyAttentionInputs,
)

try:
    # The wheel is optional per platform lock; an unavailable binary stays a
    # clean factory miss instead of an import error.
    import flash_attn as _flash_attn_module
    from flash_attn import flash_attn_varlen_func

    _HAS_FLASH_ATTN_2 = True
except (ImportError, OSError, RuntimeError):
    _flash_attn_module = None
    flash_attn_varlen_func = None
    _HAS_FLASH_ATTN_2 = False


def _has_sm120_kernels() -> bool:
    """flash-attn builds before 2.8.4 carry no sm_120 cubins.

    Importability only proves the extension loads, not that kernels exist
    for this arch: the 2.8.3.post1 r1 wheel imports cleanly on sm_120 and
    then raises cudaErrorNoKernelImageForDevice at the first launch.
    Guard on the version so old wheels keep their original (fmha_v2)
    routing instead of newly crashing through this impl.
    """
    if not _HAS_FLASH_ATTN_2:
        return False
    try:
        version = tuple(
            int(part)
            for part in _flash_attn_module.__version__.split(".")[:2]
            if part.isdigit()
        )
    except (AttributeError, ValueError):
        return False
    return version >= (2, 8)

# Head dims with tuned FA2 kernels across Ampere..Blackwell consumer.
_SUPPORTED_HEAD_DIMS = (64, 96, 128, 192, 256)


class FlashAttn2PrefillImpl(FMHAImplBase):
    """FA2 varlen prefill (no prefix) for sm120 eager path.

    Position encoding and cache writes are owned by the shared fused RoPE
    operator, exactly as in the fmha_v2 prefill impl this replaces on
    sm120; FA2 only consumes the rope'd packed q/k/v.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope = attn_configs.rope_config.style != RopeStyle.No
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.attn_inputs = attn_inputs
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.kv_head_num = attn_configs.kv_head_num
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.softmax_scale = (
            attn_configs.softmax_extra_scale
            / attn_configs.q_scaling
            * self.head_dim**-0.5
        )
        self.causal = attn_configs.is_causal
        self.max_seqlen = max(
            int(attn_inputs.input_lengths.max().item()),
            1,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if not _has_sm120_kernels() or not is_sm12x():
            return False
        if (
            attn_configs.use_mla
            or attn_configs.is_sparse
            or attn_configs.kv_cache_dtype != KvCacheDataType.BASE
            or attn_configs.dtype not in (torch.bfloat16, torch.float16)
            or attn_configs.size_per_head not in _SUPPORTED_HEAD_DIMS
            or attn_configs.head_num <= 0
            or attn_configs.kv_head_num <= 0
            or attn_configs.head_num % attn_configs.kv_head_num != 0
        ):
            return False
        if not attn_inputs.is_prefill or getattr(attn_inputs, "is_cuda_graph", False):
            return False
        # Non-paged scope, same as the fmha_v2 prefill impl this replaces.
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.any().item()
        )
        return not has_prefix

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        assert flash_attn_varlen_func is not None

        fmha_input = (
            self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            if self.need_rope or kv_cache is not None
            else qkv
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        query = (
            fmha_input[:, : self.q_size]
            .contiguous()
            .view(-1, self.head_num, self.head_dim)
        )
        key = (
            fmha_input[:, self.q_size : self.q_size + self.kv_size]
            .contiguous()
            .view(-1, self.kv_head_num, self.head_dim)
        )
        value = (
            fmha_input[:, self.q_size + self.kv_size :]
            .contiguous()
            .view(-1, self.kv_head_num, self.head_dim)
        )

        cu_seqlens = self.attn_inputs.cu_seqlens_device
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=self.max_seqlen,
            max_seqlen_k=self.max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output.view(-1, self.head_num * self.head_dim)

    def supports_generation_prefill_cuda_graph(self) -> bool:
        # The sm12x generation-prefill graph backend is owned by fmha_v2;
        # this impl never participates in graph selection (see support()).
        return False
