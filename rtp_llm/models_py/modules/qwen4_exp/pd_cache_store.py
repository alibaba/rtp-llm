# SPDX-License-Identifier: Apache-2.0
"""Publish Qwen's committed side-cache pages through the tagged store ABI."""

from collections.abc import Mapping

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    qwen4_pd_enabled,
)

SIDE_TAGS = frozenset(
    (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG, INDEXER_KV_TAG, INDEXER_STATE_TAG)
)


def validate_pd_cache_inputs(inputs) -> bool:
    """Require a complete eager prefill publication plan before any mutation.

    Decode and ordinary PDFUSION forwards have no publication plan. The native
    decode loader restores the same tagged pages into its independent pools.
    """
    inputs = tuple(inputs)
    planned = [
        getattr(value, "cache_store_inputs", None) is not None for value in inputs
    ]
    if not any(planned):
        return False
    if not qwen4_pd_enabled():
        raise RuntimeError(
            "qwen4_exp PD cache-store requires RTP_LLM_QWEN4_ENABLE_PD=1"
        )
    if not all(planned):
        raise RuntimeError(
            "qwen4_exp PD cache regions have incomplete publication plans"
        )
    for value in inputs:
        if not bool(value.is_prefill):
            raise RuntimeError("qwen4_exp PD publication requires prefill inputs")
        if bool(value.is_cuda_graph):
            raise RuntimeError("qwen4_exp PD publication requires eager prefill")
        if bool(value.is_target_verify):
            raise RuntimeError(
                "qwen4_exp PD cannot publish speculative target-verify state"
            )
        writer = getattr(value, "cache_store_writer", None)
        if writer is None or not callable(getattr(writer, "write", None)):
            raise RuntimeError("qwen4_exp PD cache-store writer is missing")
    return True


def prepare_side_cache_store_writers(attention_inputs, kv_cache, *, layer_count):
    """Pair each published side tag with its own native group-local inputs."""
    values = (
        tuple(attention_inputs.values())
        if isinstance(attention_inputs, Mapping)
        else (attention_inputs,)
    )
    if not validate_pd_cache_inputs(values):
        return {}
    if kv_cache is None or not isinstance(attention_inputs, Mapping):
        raise RuntimeError(
            "qwen4_exp PD side-cache publication needs tagged cache inputs"
        )
    required = {
        str(cache.tag)
        for layer in range(layer_count)
        for cache in kv_cache.get_layer_cache_groups(layer)
        if str(cache.tag) in SIDE_TAGS
    }
    missing = required - attention_inputs.keys()
    if missing:
        raise RuntimeError(
            f"qwen4_exp PD side-cache inputs are missing {sorted(missing)}"
        )
    # Delay the factory import until inference modules have finished loading.
    from rtp_llm.models_py.modules.factory.attention.common import (
        create_write_cache_store_impl,
    )

    writers = {}
    for tag, value in attention_inputs.items():
        if tag not in required:
            continue
        writer = create_write_cache_store_impl(value, kv_cache)
        if writer is None:
            raise RuntimeError(f"qwen4_exp PD side-cache writer is missing for {tag!r}")
        writers[tag] = writer
    return writers


def publish_layer_side_caches(kv_cache, layer_idx, writers) -> None:
    """Publish after the layer finished writing its committed page checkpoints."""
    if not writers:
        return
    for cache in kv_cache.get_layer_cache_groups(layer_idx):
        tag = str(cache.tag)
        if tag in SIDE_TAGS:
            if tag not in writers:
                raise RuntimeError(
                    f"qwen4_exp PD layer {layer_idx} has no writer for {tag!r}"
                )
            writers[tag](cache)
