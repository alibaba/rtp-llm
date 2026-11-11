"""M890P DeepSeek V4 construction adapter.

The public manifest owns qualification and topology selection. This adapter
binds the chosen FP8/FP4, attention, HC and MoE implementations to each model
instance; importing it alone does not select a model implementation.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any, Callable

import torch

from rtp_llm.models_py.modules.dsv4.ppu.attention.attention_contract import scheduler_key, validate_dual_pool, validate_swa_pool


class PpuModuleProvider:
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

    paged_indexer_query_chunk_size = 2

    def __init__(self, execution_options=None):
        self.execution_options = MappingProxyType(
            dict(os.environ if execution_options is None else execution_options)
        )

    def _bool(self, name, default=False):
        raw = self.execution_options.get(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in ("1", "true", "on", "yes"):
            return True
        if value in ("0", "false", "off", "no"):
            return False
        raise ValueError("expected one of ['0', '1', 'false', 'no', 'off', 'on', 'true', 'yes']")

    def build_shared_expert(self, *args, platform_provider=None, **kwargs):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_shared_expert import PpuSharedExpert

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
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_shared_expert import PpuSharedExpertExecutor

        return PpuSharedExpertExecutor()

    def run_hc_prenorm(self, *args: Any, **kwargs: Any) -> Any:
        from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc_prenorm import tf32_hc_prenorm_gemm

        return tf32_hc_prenorm_gemm(*args, **kwargs)

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        from rtp_llm.models_py.modules.dsv4.ppu.block import PpuBlock

        return self._build_with_provider("block", PpuBlock, *args, **kwargs)

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        from rtp_llm.models_py.modules.dsv4.ppu.transformer import PpuV4Transformer

        return self._build_with_provider(
            "transformer", PpuV4Transformer, *args, **kwargs
        )

    @staticmethod
    def run_compress_kv_write(*args, **kwargs):
        from rtp_llm.models_py.kernels.ppu.attention.compressor import run_fused_compress_kv_write

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

    def build_fp8_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import PpuFp8Linear

        return PpuFp8Linear(
            *args,
            share_input_quantization=self._bool("DSV4_PPU_SHARED_QKV_QUANT", False),
            pow2_scales=self._bool("DSV4_PPU_DENSE_POW2_SCALE", False),
            **kwargs,
        )

    def build_wo_a_fp8_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_wo_a import PpuWoAFp8Linear

        return PpuWoAFp8Linear(
            *args, sglang_layout=self._bool("DSV4_PPU_SGLANG_WO_A", False),
            bf16_input=self._bool("DSV4_PPU_WO_A_BF16", False), **kwargs
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


    indexer_mode = "FP4"
    owns_runtime_buffers = True

    def build_attention(self, default_factory, *args, **kwargs):
        from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_fp4_indexer import PpuFP4Attention

        attention = self._build_with_provider("attention", PpuFP4Attention, *args, **kwargs)
        attention._prefill_kv_qat = self._bool("DSV4_PPU_PREFILL_KV_QAT", False)
        return attention

    def build_moe(self, default_factory, *args, **kwargs):
        if kwargs.get("ep_size") == 1:
            from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_tp_moe import PpuTPMoE

            return PpuTPMoE(*args, platform_provider=self, **kwargs)
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_ep_moe import PpuEPMoE
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_legacy_deepep import PpuLegacyDeepEPStrategy

        return PpuEPMoE(
            *args,
            platform_provider=self,
            strategy_type=PpuLegacyDeepEPStrategy,
            strategy_kwargs={"options": self.execution_options},
            **kwargs,
        )

    def build_hc_unit(self, *args, **kwargs):
        from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc import PpuHCUnit

        norm = self.execution_options.get("DSV4_PPU_PREFILL_HC_NORM", "fused")
        if norm not in ("separate", "fused"):
            raise ValueError("PPU Prefill HC norm must be separate or fused")
        return PpuHCUnit(
            *args,
            options=self.execution_options,
            fuse_norm=norm == "fused",
            **kwargs,
        )

    def build_hc_head(self, *args, **kwargs):
        from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc_head import PpuHCHead

        tp_size, tp_rank = kwargs.pop("tp_size"), kwargs.pop("tp_rank")
        head = PpuHCHead(*args, options=self.execution_options, **kwargs)
        head.tp_size, head.tp_rank = tp_size, tp_rank
        return head

    def build_decode_metadata(self, default_factory, *args, **kwargs):
        return default_factory(*args, **kwargs)

    def build_mtp_input_fusion(self, enorm, hnorm, e_proj, h_proj):
        return None
