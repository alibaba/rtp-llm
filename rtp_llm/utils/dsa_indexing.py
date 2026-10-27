from typing import Any, Optional, Sequence


def _normalise_pattern_entry(entry: Any) -> str:
    if entry is None:
        return ""
    value = str(entry).strip().lower()
    if value in ("f", "full", "compute"):
        return "full"
    if value in ("s", "shared", "skip"):
        return "shared"
    return value


def _get_indexer_type(config: Any, layer_idx: int) -> Optional[str]:
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None and 0 <= layer_idx < len(indexer_types):
        return _normalise_pattern_entry(indexer_types[layer_idx])
    pattern = getattr(config, "index_topk_pattern", None)
    if isinstance(pattern, (str, Sequence)) and 0 <= layer_idx < len(pattern):
        return _normalise_pattern_entry(pattern[layer_idx])
    return None


def dsa_layer_skips_topk(config: Any, layer_idx: int) -> bool:
    """Use the reference GLM branch's explicit-pattern/frequency precedence."""
    if bool(getattr(config, "is_mtp", False)):
        return False
    layer_type = _get_indexer_type(config, layer_idx)
    if layer_type == "full":
        return False
    if layer_type == "shared":
        return True
    if layer_type not in (None, ""):
        raise ValueError(
            f"Unsupported DSA indexer type for layer {layer_idx}: {layer_type}"
        )
    freq = int(getattr(config, "index_topk_freq", 1) or 1)
    if freq <= 1:
        return False
    offset = getattr(config, "index_skip_topk_offset", None)
    if offset is not None:
        offset = int(offset)
        if offset <= 0:
            raise ValueError("index_skip_topk_offset must be positive")
        return max(layer_idx - offset + 1, 0) % freq != 0
    return max(layer_idx - 1, 0) % freq != 0


def dsa_layer_has_indexer(config: Any, layer_idx: int) -> bool:
    attn_config = getattr(config, "attn_config", None)
    if attn_config is not None and not bool(getattr(attn_config, "is_sparse", False)):
        return False
    return not dsa_layer_skips_topk(config, layer_idx)


def build_dsa_indexer_groups(
    config: Any, num_layers: int
) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    source = None
    for layer in range(num_layers):
        if dsa_layer_has_indexer(config, layer):
            source = layer
            groups[source] = []
        if source is None:
            raise ValueError("DSA layer 0 cannot share top-k before a full layer")
        groups[source].append(layer)
    return {source: tuple(layers) for source, layers in groups.items()}
