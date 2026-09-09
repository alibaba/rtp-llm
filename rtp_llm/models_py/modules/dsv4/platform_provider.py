"""Platform-neutral construction provider for DeepSeek-V4 modules.

The built-in provider delegates to the existing constructors.  An optional
provider may be registered by a platform integration before the first model
or standalone transformer is constructed.  Registration is deliberately
process-global and immutable after construction starts so one model tree
cannot contain modules from different providers.
"""

from __future__ import annotations

import threading
from enum import Enum
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
    FP4_LINEAR = "fp4_linear"
    HC_PRENORM = "hc_prenorm"
    DECODE_METADATA = "decode_metadata"


class Dsv4AttentionLayout(str, Enum):
    FLAT = "flat"
    PADDED = "padded"


class Dsv4PlatformProvider(Protocol):
    """Interface implemented by DeepSeek-V4 construction providers."""

    name: str
    capabilities: FrozenSet[Dsv4ProviderCapability]

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

    def prepare_fp4_weight_scale(
        self, default_factory: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def build_fp4_linear(
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


def _validate_provider(provider: Dsv4PlatformProvider) -> None:
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("DSV4 platform provider must have a non-empty name")
    capabilities = _normalize_capabilities(getattr(provider, "capabilities", ()))
    methods = {
        Dsv4ProviderCapability.BLOCK: "build_block",
        Dsv4ProviderCapability.TRANSFORMER: "build_transformer",
        Dsv4ProviderCapability.ATTENTION: "build_attention",
        Dsv4ProviderCapability.MOE: "build_moe",
        Dsv4ProviderCapability.DEEPEP_MOE: "build_moe",
        Dsv4ProviderCapability.FP8_LINEAR: "build_fp8_linear",
        Dsv4ProviderCapability.WO_A_FP8_LINEAR: "build_wo_a_fp8_linear",
        Dsv4ProviderCapability.BF16_FP32_LINEAR: "run_bf16_fp32_linear",
        Dsv4ProviderCapability.FP8_MQA_LOGITS: "run_fp8_mqa_logits",
        Dsv4ProviderCapability.FP4_LINEAR: "build_fp4_linear",
        Dsv4ProviderCapability.HC_PRENORM: "run_hc_prenorm",
        Dsv4ProviderCapability.DECODE_METADATA: "build_decode_metadata",
    }
    for capability in capabilities:
        method_name = methods[capability]
        if not callable(getattr(provider, method_name, None)):
            raise TypeError(
                f"DSV4 provider {name!r} declares {capability.value!r} "
                f"but has no callable {method_name}"
            )
    if Dsv4ProviderCapability.FP4_LINEAR in capabilities and not callable(
        getattr(provider, "prepare_fp4_weight_scale", None)
    ):
        raise TypeError(
            f"DSV4 provider {name!r} declares 'fp4_linear' but has no callable "
            "prepare_fp4_weight_scale"
        )
    if Dsv4ProviderCapability.ATTENTION in capabilities:
        try:
            Dsv4AttentionLayout(getattr(provider, "attention_layout"))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"DSV4 provider {name!r} must declare a valid attention_layout"
            ) from error


class Dsv4PlatformProviderRegistry:
    """Exactly-once provider registry with a construction-time freeze."""

    def __init__(self, default_provider: Optional[Dsv4PlatformProvider] = None) -> None:
        self._lock = threading.Lock()
        self._default_provider = (
            default_provider
            if default_provider is not None
            else DefaultDsv4PlatformProvider()
        )
        _validate_provider(self._default_provider)
        self._registered_provider: Optional[Dsv4PlatformProvider] = None
        self._registration_consumed = False
        self._construction_started = False

    def register(self, provider: Dsv4PlatformProvider) -> None:
        """Register once, before the first construction-time resolution."""

        _validate_provider(provider)
        with self._lock:
            if self._construction_started:
                raise RuntimeError(
                    "cannot register a DSV4 platform provider after "
                    "construction started"
                )
            if self._registration_consumed:
                raise RuntimeError("a DSV4 platform provider is already registered")
            self._registered_provider = provider
            self._registration_consumed = True

    def _active_provider(self) -> Dsv4PlatformProvider:
        if self._registered_provider is not None:
            return self._registered_provider
        return self._default_provider

    def capabilities(self) -> FrozenSet[Dsv4ProviderCapability]:
        """Query active capabilities without freezing registration."""

        with self._lock:
            return _normalize_capabilities(self._active_provider().capabilities)

    def resolve(
        self, required: Iterable[Dsv4ProviderCapability]
    ) -> Dsv4PlatformProvider:
        """Resolve for construction and permanently close registration."""

        required_set = _normalize_capabilities(required)
        with self._lock:
            provider = self._active_provider()
            _validate_provider(provider)
            available = _normalize_capabilities(provider.capabilities)
            missing = required_set - available
            if missing:
                missing_names = ", ".join(sorted(cap.value for cap in missing))
                raise RuntimeError(
                    f"DSV4 provider {provider.name!r} lacks required capabilities: "
                    f"{missing_names}"
                )
            self._construction_started = True
            return provider


_PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()


def register_dsv4_platform_provider(provider: Dsv4PlatformProvider) -> None:
    _PROVIDER_REGISTRY.register(provider)


def get_dsv4_platform_provider_capabilities() -> FrozenSet[Dsv4ProviderCapability]:
    return _PROVIDER_REGISTRY.capabilities()


def resolve_dsv4_platform_provider(
    required: Iterable[Dsv4ProviderCapability],
) -> Dsv4PlatformProvider:
    return _PROVIDER_REGISTRY.resolve(required)


def resolve_dsv4_operator_provider(platform_provider=None):
    """Use the model's bound adapter; legacy construction retains its registry."""
    if platform_provider is not None:
        return platform_provider
    return _PROVIDER_REGISTRY.resolve(())


def build_dsv4_prefill_topk(default_factory, *, platform_provider=None):
    provider = resolve_dsv4_operator_provider(platform_provider)
    builder = getattr(provider, "build_prefill_topk", None)
    return default_factory if builder is None else builder(default_factory)


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


def run_dsv4_bf16_fp32_linear(
    default_factory: Callable[..., Any],
    *args: Any,
    platform_provider=None,
    **kwargs: Any,
) -> Any:
    """Dispatch a BF16-input/weight linear with FP32 accumulation and output."""

    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.BF16_FP32_LINEAR not in capabilities:
        return default_factory(*args, **kwargs)
    return provider.run_bf16_fp32_linear(default_factory, *args, **kwargs)


def run_dsv4_hc_prenorm(*args: Any, platform_provider=None, **kwargs: Any) -> Any:
    """Run the explicitly selected deterministic HC projection.

    This capability is required: falling back to the vendor's atomic split-K
    implementation would silently lose the requested reduction semantics.
    """
    provider = (
        _PROVIDER_REGISTRY.resolve({Dsv4ProviderCapability.HC_PRENORM})
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


def prepare_dsv4_fp4_weight_scale(
    default_factory: Callable[..., Any],
    *args: Any,
    platform_provider=None,
    **kwargs: Any,
) -> Any:
    """Prepare routed FP4 scales in the active platform's native layout."""

    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.FP4_LINEAR not in capabilities:
        return default_factory(*args, **kwargs)
    prepare = getattr(provider, "prepare_fp4_weight_scale", None)
    if not callable(prepare):
        raise TypeError(
            f"DSV4 provider {provider.name!r} declares 'fp4_linear' but has no "
            "callable prepare_fp4_weight_scale"
        )
    return prepare(default_factory, *args, **kwargs)


def build_dsv4_fp4_linear(
    default_factory: Callable[..., Any],
    *args: Any,
    platform_provider=None,
    **kwargs: Any,
) -> Any:
    """Build one routed FP4 linear through the active platform provider."""

    provider = resolve_dsv4_operator_provider(platform_provider)
    capabilities = _normalize_capabilities(provider.capabilities)
    if Dsv4ProviderCapability.FP4_LINEAR not in capabilities:
        return default_factory(*args, **kwargs)
    return provider.build_fp4_linear(default_factory, *args, **kwargs)


__all__ = [
    "Dsv4AttentionLayout",
    "DefaultDsv4PlatformProvider",
    "Dsv4PlatformProvider",
    "Dsv4PlatformProviderRegistry",
    "Dsv4ProviderCapability",
    "build_dsv4_fp4_linear",
    "build_dsv4_fp8_linear",
    "build_dsv4_wo_a_fp8_linear",
    "get_dsv4_platform_provider_capabilities",
    "register_dsv4_platform_provider",
    "prepare_dsv4_fp4_weight_scale",
    "resolve_dsv4_attention_layout",
    "resolve_dsv4_platform_provider",
    "run_dsv4_bf16_fp32_linear",
    "run_dsv4_hc_prenorm",
    "run_dsv4_fp8_mqa_logits",
]
