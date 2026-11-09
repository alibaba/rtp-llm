"""Shared cache scheduling for decoders with different residual layouts."""

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.model_desc.module_base import GptModelBase
    from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
    from rtp_llm.ops import MoeConfig
    from rtp_llm.ops.compute_ops import PyModelInputs


@contextmanager
def decoder_cache_forward(
    model: "GptModelBase",
    inputs: "PyModelInputs",
    fmha_impl: "FMHAImplBase | None" = None,
) -> Iterator["FMHAImplBase"]:
    """Bind this model's caches and release active transfers on every exit."""
    failed = True
    try:
        if fmha_impl is None:
            fmha_impl = model.prepare_fmha_impl(inputs)
        fmha_impl.pinned_mla_groups = model.pinned_mla_groups
        fmha_impl.layered_prefill_cache = model.layered_prefill_cache
        if model.layered_prefill_cache is not None:
            model.layered_prefill_cache.begin_forward(fmha_impl)
        yield fmha_impl
        failed = False
    finally:
        if (
            model.layered_prefill_cache is not None
            and model.layered_prefill_cache.active
        ):
            model.layered_prefill_cache.end_forward(failed=failed)
        diagnostics = getattr(model, "_prefill_cache_diagnostics", None)
        if diagnostics is not None:
            diagnostics.finish(model.layered_prefill_cache, failed)


def should_drain_mla_before_moe(
    config: "ModelConfig", moe_config: "MoeConfig", layer_idx: int
) -> bool:
    """Freeze the existing transfer-exclusion policy before graph capture."""
    return (
        (
            os.environ.get("RTP_LLM_DSA_MLA_DRAIN_BEFORE_MOE", "1") == "1"
            or os.environ.get("RTP_LLM_CP_KV_CROSS_LAYER", "1") == "1"
        )
        and layer_idx in config.moe_layer_index
        and str(moe_config.moe_strategy).startswith("mega_moe")
    )


def wait_for_mla_prefetch(fmha_impl: "FMHAImplBase", layer_idx: int) -> None:
    """Let each cache apply its transfer policy before the layer's MoE."""
    layered = getattr(fmha_impl, "layered_prefill_cache", None)
    if layered is None:
        layered = getattr(
            getattr(fmha_impl, "fmha_impl", None), "layered_prefill_cache", None
        )
    if layered is not None:
        layered.wait_prefetch_complete()
    groups = getattr(fmha_impl, "pinned_mla_groups", None)
    if groups is None:
        groups = getattr(getattr(fmha_impl, "fmha_impl", None), "pinned_mla_groups", {})
    entry = groups.get(layer_idx)
    if entry is not None:
        working, _ = entry
        working.wait_prefetch_complete()
