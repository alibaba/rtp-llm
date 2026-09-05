from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, TypeVar, cast

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
        return value
    raise RuntimeError(
        "PyModelInputs.attention_inputs must be PyAttentionInputs or a non-empty tag mapping"
    )


def get_primary_attention_inputs(
    inputs: PyModelInputs, kv_cache: LayeredKVCache | None = None
) -> PyAttentionInputs:
    """Return the common/single fast-path value without interpreting tag names."""
    value = get_attention_inputs_value(inputs)
    if isinstance(value, PyAttentionInputs):
        return value
    return next(iter(value.values()))


def select_attention_inputs_for_tag(
    attention_inputs: AttentionInputs, tag: str
) -> PyAttentionInputs:
    """Select a group directly when the model already knows its business tag."""
    if isinstance(attention_inputs, PyAttentionInputs):
        return attention_inputs
    if not isinstance(attention_inputs, Mapping):
        raise RuntimeError(f"invalid attention_inputs type: {type(attention_inputs)!r}")
    try:
        return attention_inputs[tag]
    except KeyError as error:
        raise RuntimeError(
            f"attention input tag {tag!r} is missing; available tags={list(attention_inputs)}"
        ) from error


def get_layer_tags(kv_cache: LayeredKVCache | None, local_layer_idx: int) -> list[str]:
    if kv_cache is None:
        return []
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    tags = [str(cache.tag) for cache in layer_caches]
    if not tags or any(not tag for tag in tags):
        raise RuntimeError(f"local layer {local_layer_idx} has no cache group tag")
    if len(tags) != len(set(tags)):
        raise RuntimeError(
            f"local layer {local_layer_idx} has duplicate KV cache tag; tags={tags}"
        )
    return tags


def get_layer_cache_for_tag(
    kv_cache: LayeredKVCache | None, local_layer_idx: int, tag: str
) -> LayerKVCache:
    if kv_cache is None:
        raise RuntimeError(
            f"KV cache tag {tag!r} is missing for local layer {local_layer_idx}: "
            "KV cache is not initialized"
        )
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    matches = [cache for cache in layer_caches if str(cache.tag) == tag]
    if len(matches) > 1:
        raise RuntimeError(
            f"local layer {local_layer_idx} has duplicate KV cache tag {tag!r}"
        )
    if not matches:
        available_tags = [str(cache.tag) for cache in layer_caches]
        raise RuntimeError(
            f"KV cache tag {tag!r} is missing for local layer {local_layer_idx}; "
            f"available tags={available_tags}"
        )
    return matches[0]


def get_layer_caches_for_tags(
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
    tags: Sequence[str],
) -> dict[str, LayerKVCache]:
    layer_caches = cast(LayeredKVCache, kv_cache).get_layer_cache_groups(
        local_layer_idx
    )
    by_tag = {str(cache.tag): cache for cache in layer_caches}
    return {tag: by_tag[tag] for tag in tags}


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


def select_fmha_impl_for_tag(fmha_impl: Mapping[str, T], tag: str) -> T:
    if not isinstance(fmha_impl, Mapping):
        raise RuntimeError(
            f"sparse MLA requires tagged FMHA implementations, got {type(fmha_impl)!r}"
        )
    try:
        return fmha_impl[tag]
    except KeyError as error:
        raise RuntimeError(
            f"FMHA tag {tag!r} is missing; available tags={list(fmha_impl)}"
        ) from error
