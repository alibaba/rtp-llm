"""Worker-side binding before weight loading or implementation collectives."""

import json
import os
from pathlib import Path

from rtp_llm.device import get_cached_device_type
from rtp_llm.models_py.pluggable.bootstrap import get_module_registry
from rtp_llm.models_py.pluggable.control import verify_store_protocol
from rtp_llm.models_py.pluggable.dsv4_resources import (
    cache_geometry_snapshot,
    opaque_cache_layouts,
)
from rtp_llm.models_py.pluggable.dsv4_specs import (
    cache_description_snapshot,
    request_for,
)
from rtp_llm.models_py.pluggable.factory import (
    ModuleBuildContext,
    ModuleSelectionContext,
)
from rtp_llm.models_py.pluggable.platform import PlatformContext
from rtp_llm.models_py.pluggable.spec import canonical_json


def execution_options_snapshot(environ):
    """Compare execution choices without process ownership/output paths."""
    return {
        key: value
        for key, value in environ.items()
        if (key.startswith("DSV4_") and key != "DSV4_TASK_RUN") or key == "MOEDBG"
    }


def prepare_worker_model_context(
    model_config, engine_config, distributed_server, *, timeout_s
):
    config = engine_config.module_dispatch
    if config.mode == "legacy":
        return None
    if model_config.model_type != "deepseek_v4":
        raise ValueError("Module dispatch has no lifecycle integration for this model")
    pc = engine_config.parallelism_config
    if (
        int(pc.world_size) != int(pc.tp_size)
        or int(pc.dp_size) != 1
        or int(pc.pp_size) != 1
    ):
        raise ValueError(
            "Module startup protocol currently requires one homogeneous TP group"
        )
    platform = PlatformContext.detect(
        local_rank=int(pc.local_rank),
        requested=config.platform,
        created_device=get_cached_device_type(),
    )
    from rtp_llm.ops import KvCacheDataType, SpeculativeType

    with (Path(model_config.ckpt_path) / "config.json").open() as reader:
        checkpoint_config = json.load(reader)
    attn = model_config.attn_config
    cache_descriptions = cache_description_snapshot(model_config.kv_cache_spec_descs)
    indexer_entries = {
        desc["entry_elems"]
        for layer in cache_descriptions
        for desc in layer
        if desc["tag"] == "indexer_kv"
    }
    metadata = {
        "model_type": model_config.model_type,
        "num_layers": int(model_config.num_layers),
        "hidden_size": int(model_config.hidden_size),
        "tp_size": int(pc.tp_size),
        "ep_size": int(pc.ep_size),
        "dp_size": int(pc.dp_size),
        "pp_size": int(pc.pp_size),
        "world_size": int(pc.world_size),
        "cp_enabled": bool(pc.prefill_cp_config.is_enabled()),
        "role": pc.role_type.name,
        "speculative": engine_config.sp_config.type != SpeculativeType.NONE,
        "cuda_graph": bool(engine_config.hw_kernel_config.enable_cuda_graph),
        "reuse_cache": bool(engine_config.kv_cache_config.reuse_cache),
        "lora": bool(getattr(model_config, "lora_infos", None)),
        "eplb": bool(model_config.eplb_config.enable_eplb()),
        "indexer_cache_mode": {frozenset({132}): "fp8", frozenset({68}): "fp4"}.get(
            frozenset(indexer_entries), "unsupported"
        ),
        "cache_descriptions": cache_descriptions,
        "cache_geometry": cache_geometry_snapshot(
            model_config,
            engine_config,
            speculative=engine_config.sp_config.type != SpeculativeType.NONE,
        ),
        "fp8_kv_cache": attn.kv_cache_dtype == KvCacheDataType.FP8,
        "tokens_per_block": int(attn.tokens_per_block),
        "kernel_tokens_per_block": int(attn.kernel_tokens_per_block),
        "layer_compress_ratios": list(attn.layer_compress_ratios),
        "max_seq_len": int(model_config.max_seq_len),
        "checkpoint_config": checkpoint_config,
        "hw_kernel_config": engine_config.hw_kernel_config.to_string(),
        # Capture the current legacy options once for cross-rank comparison.
        # As each module migrates, its builder consumes this immutable snapshot.
        "execution_options": execution_options_snapshot(os.environ),
    }
    opaque_cache_layouts(metadata)
    ctx = ModuleBuildContext(
        get_module_registry(),
        ModuleSelectionContext(platform, canonical_json(metadata)),
        config,
        world_size=int(pc.world_size),
    )
    ctx.prepare([request_for("model", ctx.selection)])
    if int(pc.world_size) == 1:
        ctx.verify_protocol()
    else:
        generation = getattr(distributed_server, "_module_build_generation", 0) + 1
        distributed_server._module_build_generation = generation
        ctx.verify_protocol(
            lambda digest: verify_store_protocol(
                distributed_server.store,
                namespace=f"target-{generation}",
                rank=int(pc.world_rank),
                ranks=range(int(pc.world_size)),
                digest=digest,
                timeout_s=timeout_s,
            )
        )
    return ctx
