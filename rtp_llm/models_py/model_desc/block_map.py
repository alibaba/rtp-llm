from typing import Any, Sequence

from rtp_llm.ops.compute_ops import PyAttentionInputs


def get_attn_inputs_list(inputs: Any) -> list[PyAttentionInputs]:
    attn_inputs_list = getattr(inputs, "attn_inputs_list", None)
    if attn_inputs_list is not None and len(attn_inputs_list) > 0:
        return attn_inputs_list

    raise RuntimeError("PyModelInputs.attn_inputs_list must not be empty")


def _layer_group_ids(kv_cache: Any, local_layer_idx: int) -> list[int]:
    if kv_cache is None or not getattr(kv_cache, "layer_to_group_ids", None):
        return [0]
    if local_layer_idx < 0 or local_layer_idx >= len(kv_cache.layer_to_group_ids):
        raise RuntimeError(
            f"local layer {local_layer_idx} is out of layer_to_group_ids range "
            f"{len(kv_cache.layer_to_group_ids)}"
        )
    group_ids = [int(gid) for gid in kv_cache.layer_to_group_ids[local_layer_idx]]
    return group_ids or [0]


def _select_for_layer(
    items: Sequence[Any],
    kv_cache: Any,
    local_layer_idx: int,
    *,
    item_name: str = "item",
) -> Any:
    if len(items) <= 1:
        return items[0]

    selected = []
    for gid in _layer_group_ids(kv_cache, local_layer_idx):
        if gid < 0 or gid >= len(items):
            raise RuntimeError(
                f"local layer {local_layer_idx} maps to invalid {item_name} group {gid}; "
                f"available groups={len(items)}"
            )
        selected.append(items[gid])
    return selected[0] if len(selected) == 1 else selected


def select_attention_inputs_for_layer(
    inputs: Any, kv_cache: Any, local_layer_idx: int
) -> Any:
    """Return the PyAttentionInputs object(s) owned by a model-local layer."""
    return _select_for_layer(
        get_attn_inputs_list(inputs),
        kv_cache,
        local_layer_idx,
        item_name="attention_inputs",
    )


def select_fmha_impl_for_layer(
    fmha_impl: Any, kv_cache: Any, local_layer_idx: int
) -> Any:
    """Return the FMHA impl object(s) owned by a model-local layer."""
    if not isinstance(fmha_impl, list):
        return fmha_impl
    return _select_for_layer(
        fmha_impl,
        kv_cache,
        local_layer_idx,
        item_name="fmha_impl",
    )


def get_fmha_params(fmha_impl: Any) -> Any:
    if isinstance(fmha_impl, list):
        return [getattr(impl, "fmha_params", None) for impl in fmha_impl]
    return getattr(fmha_impl, "fmha_params", None)
