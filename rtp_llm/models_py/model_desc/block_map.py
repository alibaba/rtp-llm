from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, TypeVar

from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, PyModelInputs

AttentionInputs = PyAttentionInputs | Mapping[str, PyAttentionInputs]
T = TypeVar("T")


class LayeredKVCache(Protocol):
    def get_layer_cache_groups(
        self, local_layer_idx: int
    ) -> Sequence[LayerKVCache]: ...


def get_attention_inputs_value(inputs: PyModelInputs) -> AttentionInputs:
    value = inputs.attention_inputs
    if isinstance(value, PyAttentionInputs):
        return value
    if isinstance(value, Mapping) and value:
        for tag, attention_inputs in value.items():
            if not isinstance(tag, str) or not tag:
                raise RuntimeError(
                    f"attention input tags must be non-empty strings, got {tag!r}"
                )
            if not isinstance(attention_inputs, PyAttentionInputs):
                raise RuntimeError(
                    f"attention input tag {tag!r} has invalid value type "
                    f"{type(attention_inputs)!r}"
                )
        return value
    raise RuntimeError(
        "PyModelInputs.attention_inputs must be PyAttentionInputs or a non-empty tag mapping"
    )


def get_primary_attention_inputs(
    inputs: PyModelInputs, kv_cache: LayeredKVCache | None = None
) -> PyAttentionInputs:
    """Return metadata through an explicit topology tag binding."""
    value = get_attention_inputs_value(inputs)
    if isinstance(value, PyAttentionInputs):
        return value
    layer_cache = get_single_layer_cache(kv_cache, 0)
    if layer_cache is None:
        raise RuntimeError(
            "tagged attention inputs require KV-cache topology for explicit binding"
        )
    return select_attention_inputs_for_tag(value, str(layer_cache.tag))


def select_attention_inputs_for_tag(
    attention_inputs: AttentionInputs, tag: str
) -> PyAttentionInputs:
    """Select a group directly when the model already knows its business tag."""
    if isinstance(attention_inputs, PyAttentionInputs):
        return attention_inputs
    if not isinstance(attention_inputs, Mapping):
        raise RuntimeError(f"invalid attention_inputs type: {type(attention_inputs)!r}")
    try:
        selected = attention_inputs[tag]
    except KeyError as error:
        raise RuntimeError(
            f"attention input tag {tag!r} is missing; available tags={list(attention_inputs)}"
        ) from error
    if not isinstance(selected, PyAttentionInputs):
        raise RuntimeError(
            f"attention input tag {tag!r} has invalid value type {type(selected)!r}"
        )
    return selected


def get_layer_tags(kv_cache: LayeredKVCache | None, local_layer_idx: int) -> list[str]:
    if kv_cache is None:
        return []
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    tags = [cache.tag for cache in layer_caches]
    if not tags or any(not isinstance(tag, str) or not tag for tag in tags):
        raise RuntimeError(f"local layer {local_layer_idx} has no cache group tag")
    if len(set(tags)) != len(tags):
        raise RuntimeError(
            f"local layer {local_layer_idx} has duplicate cache group tags: {tags}"
        )
    return tags


def get_single_layer_cache(
    kv_cache: LayeredKVCache | None, local_layer_idx: int
) -> LayerKVCache | None:
    if kv_cache is None:
        return None
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    if len(layer_caches) != 1:
        raise RuntimeError(
            f"local layer {local_layer_idx} requires exactly one cache group, "
            f"got {len(layer_caches)}"
        )
    for layer_cache in layer_caches:
        return layer_cache
    raise RuntimeError(f"local layer {local_layer_idx} has no cache group")


def get_layer_cache_for_tag(
    kv_cache: LayeredKVCache | None, local_layer_idx: int, tag: str
) -> LayerKVCache:
    return get_layer_caches_for_tags(kv_cache, local_layer_idx, (tag,))[tag]


def get_layer_caches_for_tags(
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
    tags: Sequence[str],
) -> dict[str, LayerKVCache]:
    if kv_cache is None:
        raise RuntimeError(
            f"KV cache is required for local layer {local_layer_idx} tags {list(tags)!r}"
        )
    required_tags = set(tags)
    if len(required_tags) != len(tags):
        raise RuntimeError(f"duplicate required KV cache tags: {list(tags)!r}")
    matches: dict[str, list[LayerKVCache]] = {tag: [] for tag in tags}
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    available: list[str] = []
    seen_tags: set[str] = set()
    for cache in layer_caches:
        if not isinstance(cache, LayerKVCache):
            raise RuntimeError(
                f"local layer {local_layer_idx} has invalid cache object "
                f"{type(cache)!r}"
            )
        cache_tag = cache.tag
        if not isinstance(cache_tag, str) or not cache_tag:
            raise RuntimeError(
                f"local layer {local_layer_idx} has invalid cache tag {cache_tag!r}"
            )
        if cache_tag in seen_tags:
            raise RuntimeError(
                f"local layer {local_layer_idx} has duplicate KV cache tag "
                f"{cache_tag!r}"
            )
        seen_tags.add(cache_tag)
        available.append(cache_tag)
        if cache_tag in required_tags:
            matches[cache_tag].append(cache)
    for tag, tagged_caches in matches.items():
        if len(tagged_caches) != 1:
            raise RuntimeError(
                f"local layer {local_layer_idx} requires exactly one KV cache for "
                f"tag {tag!r}, got {len(tagged_caches)}; available tags={available}"
            )
    return {tag: tagged_caches[0] for tag, tagged_caches in matches.items()}


def select_fmha_impl_for_tag(fmha_impl: Mapping[str, T], tag: str) -> T:
    if not isinstance(fmha_impl, Mapping):
        raise RuntimeError(
            f"tagged FMHA routing requires a mapping, got {type(fmha_impl)!r}"
        )
    if tag not in fmha_impl:
        raise RuntimeError(
            f"FMHA tag {tag!r} is missing; available tags={list(fmha_impl)}"
        )
    return fmha_impl[tag]


def get_group_tags_for_layers(
    kv_cache: LayeredKVCache | None, local_layer_indices: Iterable[int]
) -> list[str]:
    """Return topology tags for model-selected layers, preserving topology order."""
    tags: list[str] = []
    seen: set[str] = set()
    for local_layer_idx in local_layer_indices:
        for tag in get_layer_tags(kv_cache, local_layer_idx):
            if tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def select_attention_inputs_for_layer(
    inputs: PyModelInputs,
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
) -> PyAttentionInputs | list[PyAttentionInputs]:
    """Return the group-local input(s) owned by a model-local layer."""
    value = get_attention_inputs_value(inputs)
    if isinstance(value, PyAttentionInputs):
        return value

    tags = get_layer_tags(kv_cache, local_layer_idx)
    selected = [select_attention_inputs_for_tag(value, tag) for tag in tags]
    return selected[0] if len(selected) == 1 else selected


def select_fmha_impl_for_layer(
    fmha_impl: T | Mapping[str, T],
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
) -> T | list[T]:
    if not isinstance(fmha_impl, Mapping):
        return fmha_impl
    tags = get_layer_tags(kv_cache, local_layer_idx)
    selected = []
    for tag in tags:
        if tag not in fmha_impl:
            raise RuntimeError(
                f"FMHA tag {tag!r} is missing; available tags={list(fmha_impl)}"
            )
        selected.append(fmha_impl[tag])
    return selected[0] if len(selected) == 1 else selected
