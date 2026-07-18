import fnmatch
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type

import torch


class QuantizeMethodBase(ABC):
    def __init__(self, quant_config: Any = None):
        self.quant_config = quant_config

    def validate_runtime_device(self, device: torch.device) -> None:
        """Fail before device migration when no executable backend exists."""

    @abstractmethod
    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer) -> None:
        raise NotImplementedError


_LINEAR_METHOD_REGISTRY: Dict[str, Type[QuantizeMethodBase]] = {}

_FP8_METHOD_KEYS = frozenset(
    {
        "fp8",
        "fp8_online",
        "fp8_per_channel",
        "fp8_per_channel_online",
        "fp8_block",
        "fp8_block_online",
        "FP8_PER_TENSOR_COMPRESSED",
        "FP8_DYNAMIC_PER_TENSOR",
        "FP8_PER_CHANNEL_COMPRESSED",
        "FP8_PER_CHANNEL_QUARK",
        "FP8_PER_BLOCK",
    }
)


def register_quant_method(*keys: str):
    def decorator(cls: Type[QuantizeMethodBase]) -> Type[QuantizeMethodBase]:
        for key in keys:
            existing = _LINEAR_METHOD_REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Quant method {key!r} is already registered to "
                    f"{existing.__name__}"
                )
            _LINEAR_METHOD_REGISTRY[key] = cls
        return cls

    return decorator


class QuantizationConfig:
    """Runtime quantization dispatch for streamed newloader weights."""

    def __init__(
        self,
        quant_type: str = "none",
        source_config: Any = None,
        ignored_layers: Optional[Iterable[str]] = None,
        hw_kernel_config: Any = None,
    ):
        if not isinstance(quant_type, str):
            raise TypeError("quant_type must be a string")
        normalized = quant_type.strip()
        if normalized.lower() in ("", "none"):
            normalized = "none"
        self.quant_type = normalized
        self.source_config = source_config
        use_swizzle_a = getattr(hw_kernel_config, "use_swizzleA", False)
        if not isinstance(use_swizzle_a, bool):
            raise TypeError("hw_kernel_config.use_swizzleA must be a bool")
        self.hw_kernel_config = hw_kernel_config
        self.ignored_layers = self._normalize_ignored(
            self._source_ignored_layers(source_config)
            if ignored_layers is None
            else ignored_layers
        )
        self.activation_dynamic = self._activation_dynamic(source_config)
        self.weight_block_size = self._weight_block_size(self.quant_type, source_config)

    @staticmethod
    def _normalize_ignored(values: Iterable[str]) -> List[str]:
        if isinstance(values, str):
            values = [values]
        try:
            candidates = list(values)
        except TypeError as exc:
            raise TypeError("ignored_layers must be an iterable of strings") from exc
        result = []
        for value in candidates:
            if not isinstance(value, str):
                raise TypeError("ignored layer patterns must be strings")
            value = value.strip()
            if value and value not in result:
                result.append(value)
        return result

    @classmethod
    def _source_ignored_layers(cls, source_config: Any) -> List[str]:
        if source_config is None:
            return []
        result = []
        for name in (
            "ignore_patterns",
            "ignored_layers",
            "ignore",
            "exclude_modules",
        ):
            value = getattr(source_config, name, None)
            if callable(value):
                value = value()
            if value:
                result.extend(cls._normalize_ignored(value))
        return result

    @staticmethod
    def _activation_dynamic(source_config: Any) -> bool:
        """Return whether per-tensor activation scales are computed at runtime."""
        if source_config is None:
            return True
        value = getattr(source_config, "is_dynamic", None)
        if callable(value):
            value = value()
        elif value is None:
            value = getattr(source_config, "activation_dynamic", None)
        if value is None:
            return True
        if not isinstance(value, bool):
            raise TypeError("activation dynamic flag must be a bool")
        return value

    @staticmethod
    def _weight_block_size(quant_type: str, source_config: Any) -> List[int]:
        value = getattr(source_config, "weight_block_size", None)
        if value is None:
            group_size = getattr(source_config, "group_size", None)
            if callable(group_size):
                group_size = group_size()
            if (
                quant_type in ("fp8_block", "fp8_block_online", "FP8_PER_BLOCK")
                and isinstance(group_size, int)
                and not isinstance(group_size, bool)
                and group_size > 0
            ):
                value = [group_size, group_size]
            else:
                value = [128, 128]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in value
            )
        ):
            raise ValueError(
                "weight_block_size must contain two positive integers, "
                f"got {value!r}"
            )
        return list(value)

    @staticmethod
    def _canonical_parts(path: str) -> List[str]:
        parts = [part for part in path.split(".") if part]
        while parts and parts[0] in ("model", "language_model"):
            parts.pop(0)
        return parts

    def is_layer_ignored(self, prefix: str) -> bool:
        if not prefix or not self.ignored_layers:
            return False
        prefix_parts = self._canonical_parts(prefix)
        canonical_prefix = ".".join(prefix_parts)
        for pattern in self.ignored_layers:
            canonical_pattern = ".".join(self._canonical_parts(pattern))
            if pattern.startswith("re:"):
                if re.search(pattern[3:], prefix) or re.search(
                    pattern[3:], canonical_prefix
                ):
                    return True
            elif "{i}" in canonical_pattern:
                expression = re.escape(canonical_pattern).replace(
                    re.escape("{i}"), r"\d+"
                )
                if re.fullmatch(expression, canonical_prefix):
                    return True
            elif "*" in canonical_pattern or "?" in canonical_pattern:
                if fnmatch.fnmatch(
                    canonical_prefix, canonical_pattern
                ) or fnmatch.fnmatch(canonical_prefix, f"{canonical_pattern}.*"):
                    return True
            else:
                pattern_parts = self._canonical_parts(pattern)
                if len(pattern_parts) == 1 and pattern_parts[0] in prefix_parts:
                    return True
                if prefix_parts[: len(pattern_parts)] == pattern_parts:
                    return True
        return False

    def get_quant_method(self, layer, prefix: str = "") -> QuantizeMethodBase:
        if self.ignored_layers and not prefix:
            raise ValueError(
                "A stable module prefix is required when quantization exclusions "
                "are configured"
            )
        ignore_prefixes = [prefix]
        shard_names = getattr(layer, "shard_names", ())
        if prefix and shard_names:
            parent, separator, _ = prefix.rpartition(".")
            ignore_prefixes.extend(
                f"{parent}{separator}{name}" if separator else name
                for name in shard_names
            )
        ignored = [self.is_layer_ignored(candidate) for candidate in ignore_prefixes]
        if ignored[0] or (len(ignored) > 1 and all(ignored[1:])):
            from rtp_llm.models_py.quant_methods.unquantized import (
                UnquantizedLinearMethod,
            )

            return UnquantizedLinearMethod(self)
        if len(ignored) > 1 and any(ignored[1:]):
            raise ValueError(
                f"Quantization exclusions partially match fused layer {prefix!r}; "
                "all fused projections must use the same quantization layout"
            )

        if self.quant_type not in _LINEAR_METHOD_REGISTRY:
            if self.quant_type == "none":
                from rtp_llm.models_py.quant_methods import unquantized  # noqa: F401
            elif self.quant_type in _FP8_METHOD_KEYS:
                from rtp_llm.models_py.quant_methods import fp8  # noqa: F401

        method_cls = _LINEAR_METHOD_REGISTRY.get(self.quant_type)
        if method_cls is None:
            raise ValueError(
                f"Quantization {self.quant_type!r} is not supported by the "
                "Qwen dense newloader path"
            )
        return method_cls(self)
