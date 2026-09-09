"""Model-owned V4 bridge for configuration, selection and delayed resources.

Registered through DeepSeekV4.get_module_adapter, without another model-name
registry. Descriptor import is runtime-free; tensor checks/builders are lazy.
"""

import json
import logging
import os
from pathlib import Path

from rtp_llm.models_py.pluggable.bootstrap import get_module_registry
from rtp_llm.models_py.pluggable.resources import ResourcePlan
from rtp_llm.models_py.pluggable.spec import canonical_json, load_entrypoint

from .resources import (
    cache_geometry_snapshot,
    opaque_cache_layouts,
    validate_bound_cache,
)
from .specs import (
    cache_description_snapshot,
    declared_indexer_cache_mode,
    register_modules,
    request_for,
    validate_runtime_role,
)


def get_registry():
    return get_module_registry(register_modules)


def execution_options_snapshot(environ):
    """Compare execution choices without process ownership/output paths."""
    return {
        key: value
        for key, value in environ.items()
        if (key.startswith("DSV4_") and key != "DSV4_TASK_RUN") or key == "MOEDBG"
    }


def validate_parallelism(pc):
    """The startup store covers the whole world before any DeepEP group exists."""
    tp, dp, ep, pp, world = (
        int(getattr(pc, name))
        for name in ("tp_size", "dp_size", "ep_size", "pp_size", "world_size")
    )
    tp_group = world == tp and dp == 1
    decode_ep_group = tp == 1 and dp == ep == world and pc.role_type.name == "DECODE"
    if min(tp, dp, ep, pp, world) < 1 or pp != 1 or not (tp_group or decode_ep_group):
        raise ValueError(
            "Module startup requires a homogeneous TP group or TP1 Decode DP/EP world"
        )


def metadata_snapshot(model_config, engine_config):
    pc = engine_config.parallelism_config
    validate_parallelism(pc)
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
        "moe_communication": {
            "enabled": bool(engine_config.moe_config.use_deepep_moe),
            "low_latency": bool(engine_config.moe_config.use_deepep_low_latency),
            "internode": bool(engine_config.moe_config.use_deepep_internode),
            "all_gather": bool(engine_config.moe_config.use_all_gather),
            "num_sms": int(engine_config.moe_config.deep_ep_num_sm),
            "max_generate_batch_size": int(
                engine_config.runtime_config.max_generate_batch_size
            ),
            "ffn_disaggregate": bool(
                getattr(pc.ffn_disaggregate_config, "enable_ffn_disaggregate", False)
            ),
        },
        # Capture the current legacy options once for cross-rank comparison.
        # As each module migrates, its builder consumes this immutable snapshot.
        "execution_options": execution_options_snapshot(os.environ),
    }
    return metadata


class Dsv4ModelAdapter:
    def registry(self):
        return get_registry()

    def configure_model(self, model_cls, model_config, kv_cache_config, dispatch):
        model_cls._apply_kv_cache_config(
            model_config,
            kv_cache_config,
            indexer_cache_mode=declared_indexer_cache_mode(dispatch),
        )

    def metadata(self, model_config, engine_config):
        return metadata_snapshot(model_config, engine_config)

    def root_request(self, selection):
        return request_for("model", selection)

    def plan_resources(self, selection, bindings):
        metadata = selection.model_metadata
        root = next(b for b in bindings if b.request == self.root_request(selection))
        if root.implementation.describe_resources is None:
            raise ValueError(
                "V4 model implementation must describe its allocator inputs"
            )
        if root.implementation.validate_initialized is None:
            raise ValueError("V4 model implementation must declare its readiness check")
        parameters = load_entrypoint(root.implementation.describe_resources)()
        if parameters["indexer_cache_mode"].lower() != metadata["indexer_cache_mode"]:
            raise ValueError("Selected allocator inputs differ from configured cache")
        weight_plans = {
            b.implementation.prepare_weights
            for b in bindings
            if b.request.module_id == "rtp.dsv4.moe"
        }
        if len(weight_plans) > 1:
            raise ValueError(
                "Current V4 loader requires one routed weight layout across layers"
            )
        return ResourcePlan(
            canonical_json(
                {"parameters": parameters, "layers": opaque_cache_layouts(metadata)}
            ),
            next(iter(weight_plans), root.implementation.prepare_weights),
        )

    def build_root(self, owner, context):
        return context.factory.build(
            self.root_request(context.selection),
            model_config=owner.model_config,
            parallelism_config=owner.parallelism_config,
            weights=owner.weight,
            moe_config=owner.moe_config,
            max_generate_batch_size=owner.max_generate_batch_size,
            fmha_config=owner.fmha_config,
            py_hw_kernel_config=owner.hw_kernel_config,
            device_resource_config=owner.device_resource_config,
        )

    def validate_resources(self, model, init_resource, context):
        metadata = context.selection.model_metadata
        validate_runtime_role(
            metadata,
            is_decode_role=init_resource.is_decode_role,
            is_speculative=init_resource.is_speculative,
        )
        if (
            cache_description_snapshot(model.config.kv_cache_spec_descs)
            != metadata["cache_descriptions"]
        ):
            raise ValueError("Cache descriptors changed after resource planning")
        record = validate_bound_cache(
            init_resource.kv_cache,
            metadata,
            device=context.selection.platform.device_string,
        )
        logging.info(
            "module_dispatch resources: %s",
            canonical_json(
                {
                    "model_instance_id": context.model_instance_id,
                    "protocol_digest": context.protocol_digest,
                    **record,
                }
            ),
        )

    def validate_initialized_model(self, model, init_resource, context):
        self.validate_resources(model, init_resource, context)
        if (model.kv_cache is None) != (init_resource.kv_cache is None):
            raise RuntimeError("V4 initialize did not bind its engine resources")
        if model.kv_cache is None:
            return
        validate_bound_cache(
            model.kv_cache,
            context.selection.model_metadata,
            device=context.selection.platform.device_string,
        )
        # pybind may create a fresh wrapper on each property access. Verify the
        # actual tensor allocations rather than Python wrapper identity.
        for layer in range(model.kv_cache.layer_count):
            expected = {
                v.tag: v for v in init_resource.kv_cache.get_layer_cache_groups(layer)
            }
            for view in model.kv_cache.get_layer_cache_groups(layer):
                if (
                    view.kv_cache_base.data_ptr()
                    != expected[view.tag].kv_cache_base.data_ptr()
                ):
                    raise RuntimeError(
                        "V4 initialize replaced an engine-owned cache allocation"
                    )
