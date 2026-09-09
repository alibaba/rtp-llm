"""M890P DeepSeek V4 construction adapter.

The public manifest owns qualification and topology selection. This adapter
binds the chosen FP8/FP4, attention, HC and MoE implementations to each model
instance; importing it alone does not select a model implementation.
"""

from __future__ import annotations

import json
import logging
import os
from types import MappingProxyType
from typing import Any, Callable, FrozenSet

import torch

from rtp_llm.models_py.modules.dsv4.platform_provider import (
    Dsv4AttentionLayout,
    Dsv4PlatformProvider,
    Dsv4ProviderCapability,
    register_dsv4_platform_provider,
)

M890P_DEVICE_NAME = "ZW-M890P"
FP8_INDEXER_MODE = "FP8"
_MOE_OBSERVABILITY_EVENT = "dsv4_moe_provider_selection"
_LOGGER = logging.getLogger(__name__)


def _process_rank() -> Any:
    for name in ("WORLD_RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return value
    return None


def _log_moe_selection(
    *,
    provider: str,
    tp_size: Any,
    ep_size: Any,
    requested_strategy: Any,
    strategy: Any,
    forced_strategy: Any,
    decision: str,
) -> None:
    record = {
        "decision": decision,
        "ep_size": ep_size,
        "event": _MOE_OBSERVABILITY_EVENT,
        "forced_strategy": forced_strategy,
        "provider": provider,
        "requested_strategy": requested_strategy,
        "strategy": strategy,
        "tp_size": tp_size,
        "world_rank": _process_rank(),
    }
    _LOGGER.info(
        "DSV4_PROVIDER_SELECTION %s",
        json.dumps(record, sort_keys=True, separators=(",", ":"), default=repr),
    )


def _fail_closed(operation: str) -> None:
    raise RuntimeError(
        f"DSV4 candidate provider {operation} is unavailable: "
        "the requested topology is outside the verified EP1/EP8 contracts"
    )


class M890PDsv4Provider:
    """M890P construction adapter; supported execution modes come from the manifest.

    Construction factories preserve provider identity while delegating to the
    supplied factory. Attention uses the verified public rich-FlashMLA path;
    EP1/TP4 MoE forces the PPU grouped-FP4 strategy. Other EP1 topologies keep
    their existing selection, while EP8 keeps the verified DeepEP normal-mode
    dispatch/combine path. Other EP sizes remain blocked.
    """

    name = "m890p-dsv4-candidate"
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
            Dsv4ProviderCapability.FP4_LINEAR,
            Dsv4ProviderCapability.HC_PRENORM,
        }
    )
    attention_layout = Dsv4AttentionLayout.FLAT
    # The rich Attention factory currently constructs IndexerFP8. Its pool
    # entry is 132 bytes; the routed experts independently use MXFP4.
    indexer_mode = FP8_INDEXER_MODE

    def __init__(self, execution_options=None):
        self.execution_options = MappingProxyType(
            dict(os.environ if execution_options is None else execution_options)
        )

    def _bool(self, name, default=False):
        from rtp_llm.models_py.modules.dsv4.runtime_config import parse_bool

        raw = self.execution_options.get(name)
        return default if raw is None else parse_bool(raw, default)

    def build_shared_expert(self, *args, platform_provider=None, **kwargs):
        from .ppu_shared_expert import PpuSharedExpert

        if platform_provider is not None and platform_provider is not self:
            raise ValueError("Shared expert received a different model adapter")
        return PpuSharedExpert(
            *args,
            platform_provider=self,
            sglang_moe=self._bool("DSV4_PPU_SGLANG_MOE", False),
            **kwargs,
        )

    def build_prefill_topk(self, default_factory):
        from .ppu_topk import PpuPrefillTopK

        return PpuPrefillTopK(self.execution_options)

    def run_hc_prenorm(self, *args: Any, **kwargs: Any) -> Any:
        from .ppu_hc_prenorm import tf32_hc_prenorm_gemm

        return tf32_hc_prenorm_gemm(*args, **kwargs)

    @staticmethod
    def require_device_name(device_name: str) -> None:
        if device_name != M890P_DEVICE_NAME:
            raise RuntimeError(
                f"DSV4 candidate provider requires {M890P_DEVICE_NAME}, "
                f"got {device_name!r}"
            )

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return self._build_with_provider("block", default_factory, *args, **kwargs)

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return self._build_with_provider(
            "transformer", default_factory, *args, **kwargs
        )

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

    def build_moe(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        ep_size = kwargs.get("ep_size")
        tp_size = kwargs.get("tp_size")
        requested_strategy = kwargs.get("strategy")
        if type(ep_size) is not int or ep_size not in (1, 8):
            _log_moe_selection(
                provider=self.name,
                tp_size=tp_size,
                ep_size=ep_size,
                requested_strategy=requested_strategy,
                strategy=requested_strategy,
                forced_strategy=None,
                decision="fail_closed",
            )
            _fail_closed(f"MoE construction for ep_size={ep_size!r}")
        if ep_size == 8 and self._bool("DSV4_PPU_GROUPED_FP4", False):
            from .ppu_legacy_deepep import PpuLegacyDeepEPStrategy

            if kwargs.pop("strategy", None) not in (None, "deepep"):
                raise ValueError(
                    "PPU grouped DeepEP conflicts with the requested strategy"
                )
            return default_factory(
                *args,
                strategy_type=PpuLegacyDeepEPStrategy,
                strategy_kwargs={"options": self.execution_options},
                platform_provider=self,
                execution_options=self.execution_options,
                **kwargs,
            )
        forced_strategy = None
        if ep_size == 1 and type(tp_size) is int and tp_size == 4:
            forced_strategy = "ppu_grouped_fp4"
            kwargs["strategy"] = forced_strategy
        _log_moe_selection(
            provider=self.name,
            tp_size=tp_size,
            ep_size=ep_size,
            requested_strategy=requested_strategy,
            strategy=kwargs.get("strategy"),
            forced_strategy=forced_strategy,
            decision="delegate",
        )
        return default_factory(*args, **kwargs)

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
        from .ppu_wo_a import PpuWoAFp8Linear

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

    def prepare_fp4_weight_scale(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import (
            prepare_fp4_weight_scale_mxfp4,
        )

        scale = args[0] if args else kwargs["scale"]
        return prepare_fp4_weight_scale_mxfp4(scale)

    def build_fp4_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        del default_factory
        from rtp_llm.platforms.ppu.modules.linear.fp4_linear import PpuFp4Linear

        return PpuFp4Linear(*args, **kwargs)


def register_m890p_dsv4_provider(
    *, device_name: str, ep_size: int
) -> M890PDsv4Provider:
    """Explicitly register the EP1/EP8 candidate before model construction.

    This function is intentionally never called at import time. Only the EP1
    pure-TP and EP8 distributed topologies covered by the M890P gates are
    accepted.
    """

    provider = M890PDsv4Provider()
    provider.require_device_name(device_name)
    if type(ep_size) is not int or ep_size not in (1, 8):
        _fail_closed(f"provider registration for ep_size={ep_size!r}")
    register_dsv4_platform_provider(provider)
    return provider


__all__ = [
    "FP8_INDEXER_MODE",
    "M890P_DEVICE_NAME",
    "M890PDsv4Provider",
    "register_m890p_dsv4_provider",
]
