"""Instance-bound DeepSeek-V4 operators; omitted providers retain CUDA defaults."""

from __future__ import annotations
from enum import Enum
from functools import partial
from typing import Any, Callable, FrozenSet, Iterable, Optional, Protocol


class Dsv4ProviderCapability(str, Enum):
    """Construction points a provider must explicitly claim."""

    BLOCK = "block"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    MOE = "moe"
    DEEPEP_MOE = "deepep_moe"
    FP8_LINEAR = "fp8_linear"
    WO_A_FP8_LINEAR = "wo_a_fp8_linear"
    BF16_FP32_LINEAR = "bf16_fp32_linear"
    FP8_MQA_LOGITS = "fp8_mqa_logits"
    HC_PRENORM = "hc_prenorm"
    DECODE_METADATA = "decode_metadata"
    MTP_INPUT_FUSION = "mtp_input_fusion"


class Dsv4AttentionLayout(str, Enum):
    FLAT = "flat"
    PADDED = "padded"


class Dsv4PlatformProvider(Protocol):
    """Interface implemented by DeepSeek-V4 construction providers."""

    name: str
    capabilities: FrozenSet[Dsv4ProviderCapability]

    def build_mtp_input_fusion(self, enorm, hnorm, e_proj, h_proj) -> Any: ...

    def run_hc_prenorm(self, *args: Any, **kwargs: Any) -> Any: ...

    def build_decode_metadata(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_attention(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_moe(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_fp8_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_wo_a_fp8_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def run_bf16_fp32_linear(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def run_fp8_mqa_logits(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...




class DefaultDsv4PlatformProvider:
    """Provider preserving the existing constructors and runtime behavior."""

    name = "cuda"
    attention_layout = Dsv4AttentionLayout.FLAT
    capabilities = frozenset(
        {
            Dsv4ProviderCapability.BLOCK,
            Dsv4ProviderCapability.TRANSFORMER,
        }
    )

    def build_block(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return default_factory(*args, **kwargs)

    def build_transformer(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return default_factory(*args, **kwargs)


def _normalize_capabilities(
    capabilities: Iterable[Dsv4ProviderCapability],
) -> FrozenSet[Dsv4ProviderCapability]:
    try:
        return frozenset(
            Dsv4ProviderCapability(capability) for capability in capabilities
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid DSV4 provider capability: {error}") from error


def resolve_dsv4_attention_layout(
    provider: Dsv4PlatformProvider,
) -> Dsv4AttentionLayout:
    """Resolve layout only for providers that explicitly own attention."""

    capabilities = _normalize_capabilities(getattr(provider, "capabilities", ()))
    if Dsv4ProviderCapability.ATTENTION not in capabilities:
        return Dsv4AttentionLayout.FLAT
    return Dsv4AttentionLayout(getattr(provider, "attention_layout"))


_DEFAULT_PROVIDER = DefaultDsv4PlatformProvider()


def resolve_dsv4_platform_provider(
    required: Iterable[Dsv4ProviderCapability],
) -> Dsv4PlatformProvider:
    missing = _normalize_capabilities(required) - _DEFAULT_PROVIDER.capabilities
    if missing:
        raise RuntimeError(f"CUDA DSV4 provider lacks required capabilities: {missing}")
    return _DEFAULT_PROVIDER


def resolve_dsv4_operator_provider(platform_provider=None):
    """Use the model's bound adapter, or unchanged CUDA operators."""
    if platform_provider is not None:
        return platform_provider
    return _DEFAULT_PROVIDER


def build_dsv4_decode_metadata(
    default_factory, *args, platform_provider=None, **kwargs
):
    """Construct the per-graph metadata owner through the model's provider."""
    provider = resolve_dsv4_operator_provider(platform_provider)
    if Dsv4ProviderCapability.DECODE_METADATA not in provider.capabilities:
        return default_factory(*args, **kwargs)
    return provider.build_decode_metadata(default_factory, *args, **kwargs)


def build_dsv4_fp8_linear(
    default_factory: Callable[..., Any],
    *args: Any,
    platform_provider=None,
    **kwargs: Any,
) -> Any:
    """Let an active platform provider own DSV4 FP8 storage and launch.

    Providers that do not explicitly claim ``FP8_LINEAR`` preserve the
    existing public ``LinearFactory`` path.  Resolution is construction-time
    immutable, matching the block/attention/MoE provider boundary.
    """

    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.FP8_LINEAR not in capabilities:
        return default_factory(*args, **kwargs)
    return provider.build_fp8_linear(default_factory, *args, **kwargs)


def build_dsv4_wo_a_fp8_linear(
    default_factory: Callable[..., Any],
    *args: Any,
    platform_provider=None,
    **kwargs: Any,
) -> Any:
    """Let a provider own the grouped DSV4 attention output projection."""

    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.WO_A_FP8_LINEAR not in capabilities:
        return default_factory(*args, **kwargs)
    return provider.build_wo_a_fp8_linear(default_factory, *args, **kwargs)


def build_dsv4_bf16_fp32_linear(default_factory, *, platform_provider=None):
    """Bind the BF16-input/FP32-output operation once per owning module."""
    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.BF16_FP32_LINEAR not in capabilities:
        return default_factory
    return partial(provider.run_bf16_fp32_linear, default_factory)


def run_dsv4_hc_prenorm(*args: Any, platform_provider=None, **kwargs: Any) -> Any:
    """Run the explicitly selected deterministic HC projection.

    This capability is required: falling back to the vendor's atomic split-K
    implementation would silently lose the requested reduction semantics.
    """
    provider = (
        resolve_dsv4_platform_provider({Dsv4ProviderCapability.HC_PRENORM})
        if platform_provider is None
        else platform_provider
    )
    if Dsv4ProviderCapability.HC_PRENORM not in provider.capabilities:
        raise RuntimeError("DSV4 provider lacks required capability hc_prenorm")
    return provider.run_hc_prenorm(*args, **kwargs)


def run_dsv4_fp8_mqa_logits(
    default_factory: Callable[..., Any],
    *args: Any,
    platform_provider=None,
    **kwargs: Any,
) -> Any:
    """Dispatch prefill indexer FP8 MQA logits through a platform adapter."""

    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.FP8_MQA_LOGITS not in capabilities:
        return default_factory(*args, **kwargs)
    return provider.run_fp8_mqa_logits(default_factory, *args, **kwargs)






__all__ = [
    "Dsv4AttentionLayout",
    "DefaultDsv4PlatformProvider",
    "Dsv4PlatformProvider",
    "Dsv4ProviderCapability",
    "build_dsv4_fp8_linear",
    "build_dsv4_wo_a_fp8_linear",
    "resolve_dsv4_attention_layout",
    "resolve_dsv4_platform_provider",
    "build_dsv4_bf16_fp32_linear",
    "run_dsv4_hc_prenorm",
    "run_dsv4_fp8_mqa_logits",
]
