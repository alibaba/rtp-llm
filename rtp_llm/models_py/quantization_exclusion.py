import fnmatch
import re
from typing import Any, Iterable, List, Sequence


def normalize_module_patterns(values: Iterable[str]) -> List[str]:
    """Normalize checkpoint exclusion patterns without runtime dependencies."""
    if isinstance(values, str):
        values = [values]
    try:
        candidates = (
            sorted(values) if isinstance(values, (set, frozenset)) else list(values)
        )
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


def collect_quantization_exclusions(source_config: Any) -> List[str]:
    """Collect all supported exclusion aliases from a checkpoint config."""
    if source_config is None:
        return []

    result = []
    for name in (
        "ignore_patterns",
        "ignored_layers",
        "ignore",
        "exclude_modules",
        "exclude",
        "modules_to_not_convert",
    ):
        value = getattr(source_config, name, None)
        if callable(value):
            value = value()
        if value:
            result.extend(normalize_module_patterns(value))
    return normalize_module_patterns(result)


def canonical_module_parts(path: str) -> List[str]:
    parts = [part for part in path.split(".") if part]
    while parts and parts[0] in ("model", "language_model"):
        parts.pop(0)
    return parts


def is_module_ignored(prefix: str, patterns: Sequence[str]) -> bool:
    """Return whether a stable module prefix matches an exclusion pattern."""
    if not prefix or not patterns:
        return False

    prefix_parts = canonical_module_parts(prefix)
    canonical_prefix = ".".join(prefix_parts)
    for pattern in patterns:
        canonical_pattern = ".".join(canonical_module_parts(pattern))
        if pattern.startswith("re:"):
            if re.search(pattern[3:], prefix) or re.search(
                pattern[3:], canonical_prefix
            ):
                return True
        elif "{i}" in canonical_pattern:
            expression = re.escape(canonical_pattern).replace(re.escape("{i}"), r"\d+")
            if re.fullmatch(rf"{expression}(?:\..+)?", canonical_prefix):
                return True
        elif "*" in canonical_pattern or "?" in canonical_pattern:
            if fnmatch.fnmatch(canonical_prefix, canonical_pattern) or fnmatch.fnmatch(
                canonical_prefix, f"{canonical_pattern}.*"
            ):
                return True
        else:
            pattern_parts = canonical_module_parts(pattern)
            if len(pattern_parts) == 1 and pattern_parts[0] in prefix_parts:
                return True
            if prefix_parts[: len(pattern_parts)] == pattern_parts:
                return True
    return False


def moe_projection_exclusion_states(prefix: str, patterns: Sequence[str]) -> List[bool]:
    """Return effective gate/up/down exclusion states for a fused MoE layer.

    Checkpoints may name the first two logical projections separately or by
    their fused runtime name ``gate_up_proj``. Treating the fused name as an
    unrelated module would silently leave those weights quantized.
    """
    gate_up_ignored = is_module_ignored(f"{prefix}.gate_up_proj", patterns)
    return [
        gate_up_ignored or is_module_ignored(f"{prefix}.gate_proj", patterns),
        gate_up_ignored or is_module_ignored(f"{prefix}.up_proj", patterns),
        is_module_ignored(f"{prefix}.down_proj", patterns),
    ]
