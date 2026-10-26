"""M890P DeepSeek V4 construction adapter.

The public manifest owns qualification and topology selection. This adapter
binds the chosen FP8/FP4, attention, HC and MoE implementations to each model
instance; importing it alone does not select a model implementation.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any, Callable, FrozenSet

import torch

from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import Dsv4AttentionLayout, Dsv4ProviderCapability
from rtp_llm.platforms.ppu.models.dsv4.attention_contract import scheduler_key, validate_dual_pool, validate_swa_pool

FP8_INDEXER_MODE = "FP8"








class M890PDsv4Provider:
    """M890P construction adapter; supported execution modes come from the manifest.

    Construction factories preserve provider identity while delegating to the
    supplied factory. Attention uses the verified public rich-FlashMLA path;
    Local MoE uses the PPU grouped-FP4 strategy, while distributed MoE uses
    the configured DeepEP dispatch/combine path.
    """

    name = "m890p-dsv4-candidate"
    validate_swa_pool = staticmethod(validate_swa_pool)
    validate_dual_pool = staticmethod(validate_dual_pool)
    scheduler_key = staticmethod(scheduler_key)
    capabilities: FrozenSet[Dsv4ProviderCapability] = frozenset(
        {
            Dsv4ProviderCapability.BLOCK,
            Dsv4ProviderCapability.TRANSFORMER,
            Dsv4ProviderCapability.ATTENTION,
            Dsv4ProviderCapability.MOE,
            Dsv4ProviderCapability.DEEPEP_MOE,
            Dsv4ProviderCapability.FP8_LINEAR,
            Dsv4ProviderCapability.WO_A_FP8_LINEAR,
            Dsv4ProviderCapability.BF16_FP32_LINEAR,
            Dsv4ProviderCapability.FP8_MQA_LOGITS,
            Dsv4ProviderCapability.HC_PRENORM,
        }
    )
    attention_layout = Dsv4AttentionLayout.FLAT
    # The rich Attention factory currently constructs IndexerFP8. Its pool
    # entry is 132 bytes; the routed experts independently use MXFP4.
    indexer_mode = FP8_INDEXER_MODE
    # PPU DeepGEMM paged MQA supports next_n in {1, 2}.
    paged_indexer_query_chunk_size = 2

    def __init__(self, execution_options=None):
        self.execution_options = MappingProxyType(
            dict(os.environ if execution_options is None else execution_options)
        )

    def _bool(self, name, default=False):
        from rtp_llm.platforms.ppu.models.dsv4.impl.runtime_config import parse_bool

        raw = self.execution_options.get(name)
        return default if raw is None else parse_bool(raw, default)

    def build_shared_expert(self, *args, platform_provider=None, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_shared_expert import PpuSharedExpert

        if platform_provider is not None and platform_provider is not self:
            raise ValueError("Shared expert received a different model adapter")
        sglang_moe = kwargs.pop(
            "sglang_moe", self._bool("DSV4_PPU_SGLANG_MOE", False)
        )
        return PpuSharedExpert(
            *args,
            platform_provider=self,
            sglang_moe=sglang_moe,
            **kwargs,
        )

    def build_shared_expert_executor(self, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_shared_expert import PpuSharedExpertExecutor

        return PpuSharedExpertExecutor()

    def run_hc_prenorm(self, *args: Any, **kwargs: Any) -> Any:
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc_prenorm import tf32_hc_prenorm_gemm

        return tf32_hc_prenorm_gemm(*args, **kwargs)

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        from rtp_llm.platforms.ppu.models.dsv4.block import PpuBlock

        return self._build_with_provider("block", PpuBlock, *args, **kwargs)

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        from rtp_llm.platforms.ppu.models.dsv4.transformer import PpuV4Transformer

        return self._build_with_provider(
            "transformer", PpuV4Transformer, *args, **kwargs
        )

    @staticmethod
    def run_compress_kv_write(*args, **kwargs):
        from rtp_llm.platforms.ppu.kernels.compressor import run_fused_compress_kv_write

        return run_fused_compress_kv_write(*args, **kwargs)

    def _build_with_provider(
        self,
        operation: str,
        default_factory: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        existing_provider = kwargs.get("platform_provider")
        if existing_provider is not None and existing_provider is not self:
            raise TypeError(
                f"DSV4 M890P {operation} factory received duplicate "
                "platform_provider"
            )
        if existing_provider is None:
            kwargs["platform_provider"] = self
        return default_factory(*args, **kwargs)

    def build_attention(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return self._build_with_provider("attention", default_factory, *args, **kwargs)


    def build_fp8_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import PpuFp8Linear

        return PpuFp8Linear(
            *args,
            share_input_quantization=self._bool("DSV4_PPU_SHARED_QKV_QUANT", False),
            **kwargs,
        )

    def build_wo_a_fp8_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        from rtp_llm.platforms.ppu.models.dsv4.ppu_wo_a import PpuWoAFp8Linear

        return PpuWoAFp8Linear(
            *args, sglang_layout=self._bool("DSV4_PPU_SGLANG_WO_A", False), **kwargs
        )

    def run_bf16_fp32_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        if len(args) == 2:
            x, weight = args
        else:
            x = kwargs["x"]
            weight = kwargs["weight"]
        return torch.mm(x, weight.t(), out_dtype=torch.float32)

    def run_fp8_mqa_logits(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Adapt the CUDA seven-argument contract to PPU DeepGEMM's ABI."""

        del default_factory
        import deep_gemm

        if args:
            if len(args) != 7:
                raise TypeError(f"expected 7 FP8 MQA arguments, got {len(args)}")
            q, kv_s, weights, cu_start, cu_end, clean_logits, _max_seqlen_k = args
        else:
            q = kwargs["q"]
            kv_s = kwargs["kv_s"]
            weights = kwargs["weights"]
            cu_start = kwargs["cu_seq_len_k_start"]
            cu_end = kwargs["cu_seq_len_k_end"]
            clean_logits = kwargs.get("clean_logits", True)
        return deep_gemm.fp8_mqa_logits(
            q, kv_s, weights, cu_start, cu_end, clean_logits
        )






__all__ = [
    "FP8_INDEXER_MODE",
    "M890PDsv4Provider",
]
