"""DeepSeek-V4 module contracts and rank-invariant build descriptions.

No model implementation, Torch, device library or weight allocation is imported
here. The first contract retains the current loader and FP8 indexer cache layout.
"""

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.pluggable.spec import (
    BuildRequest,
    ModuleImplSpec,
    ModuleSpec,
    SupportResult,
)

WEIGHT_FORMAT = "rtp.dsv4.loader-fp8-mxfp4.v1"
STATE_FORMAT = "rtp.dsv4.cache-fp8-indexer-132.v1"
STATE_FORMAT_FP4 = "rtp.dsv4.cache-fp4-indexer-68-raw-c4-state.v1"
CONTRACTS = {
    "model": "rtp.dsv4.model-hidden-state.v1",
    "block": "rtp.dsv4.block-hc-hidden.v1",
    "attention": "rtp.dsv4.attention-flat-fp8kv.v1",
    "moe": "rtp.dsv4.moe-reduced-bf16.v1",
}


def declared_indexer_cache_mode(config):
    """Determine allocator geometry from the explicit root model descriptor."""
    from rtp_llm.models_py.pluggable.bootstrap import get_module_registry

    from .cache_mode import Dsv4IndexerCacheMode

    impl_id = dict(config.path_overrides).get("v4") or dict(config.impl_overrides).get(
        "rtp.dsv4.model"
    )
    if impl_id is None:
        impl_id = "baseline.dsv4.model.v1"
    impl = get_module_registry(register_modules).implementation(
        "rtp.dsv4.model", impl_id
    )
    if impl.describe_resources is None:
        raise ValueError("V4 model implementation must describe its allocator inputs")
    from rtp_llm.models_py.pluggable.spec import load_entrypoint

    parameters = load_entrypoint(impl.describe_resources)()
    return Dsv4IndexerCacheMode[parameters["indexer_cache_mode"]]


def cache_description_snapshot(layer_descriptions):
    fields = (
        "tag",
        "cache_type",
        "dtype",
        "is_state_cache",
        "entry_elems",
        "entry_dtype",
        "entry_count_mode",
        "explicit_entry_count",
        "compression_ratio",
        "state_ring_overlap",
        "state_ring_include_gen_num_per_cycle",
        "block_stride_bytes_override",
        "block_stride_bytes_alignment",
        "block_stride_alignment_min_entries",
        "group_type",
    )
    policies = {
        "reuse": ("enable_prefix_reuse", "evict_policy"),
        "capacity": ("reservable", "explicit_block_num", "charge_to_paged_budget"),
        "memory": ("placement",),
        "tail": ("active_tail_blocks", "validate_tail_blocks"),
        "cp": (
            "mapping",
            "slice",
            "scale_seq_size",
            "align_payload",
            "prefill_slice_layout",
        ),
    }

    def scalar(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "name"):
            return value.name
        raise TypeError(
            f"Cache descriptor value has no stable scalar representation: {type(value)}"
        )

    result = []
    for layer in layer_descriptions:
        descriptions = []
        for desc in layer:
            record = {name: scalar(getattr(desc, name)) for name in fields}
            for name, members in policies.items():
                policy = getattr(desc, name)
                record[name] = (
                    None
                    if policy is None
                    else {member: scalar(getattr(policy, member)) for member in members}
                )
            descriptions.append(record)
        result.append(descriptions)
    return result


def request_for(kind, selection, layer_id=None):
    path = "v4"
    metadata = {}
    if layer_id is not None:
        metadata = {"layer_id": int(layer_id)}
        path += f".layers.{layer_id}"
    if kind == "attention":
        path += ".attn"
    elif kind == "moe":
        path += ".ffn"
    return BuildRequest.create(
        module_id="rtp.dsv4." + kind,
        path=path,
        weight_format_id=WEIGHT_FORMAT,
        state_format_id=(
            STATE_FORMAT_FP4
            if selection.model_metadata.get("indexer_cache_mode") == "fp4"
            else STATE_FORMAT
        ),
        required_capabilities={
            "decode" if selection.model_metadata.get("role") == "DECODE" else "prefill"
        },
        metadata=metadata,
    )


def validate_runtime_role(metadata, *, is_decode_role, is_speculative):
    """Check actual delayed-init resources against the preflight role."""
    if bool(is_decode_role) != (metadata.get("role") == "DECODE"):
        raise ValueError("Runtime Decode role differs from the module preflight")
    if bool(is_speculative) or bool(metadata.get("speculative")):
        raise ValueError("Runtime speculation is not supported by this module contract")


def forward_capabilities(bindings):
    """Snapshot phases supported by every selected module before allocation."""
    if not bindings:
        raise ValueError("Forward capabilities require a prepared module plan")
    return frozenset.intersection(
        *(binding.implementation.capabilities for binding in bindings)
    )


def validate_forward_phase(
    capabilities, *, is_prefill, has_decode_fmha, is_target_verify
):
    if is_target_verify:
        phase = "target_verify"
    elif has_decode_fmha or not is_prefill:
        phase = "decode"
    else:
        phase = "prefill"
    if phase not in capabilities:
        raise RuntimeError(f"Selected module contract does not support {phase}")


def describe_model(selection, request):
    return [
        request_for("block", selection, index)
        for index in range(selection.model_metadata["num_layers"])
    ]


def describe_block(selection, request):
    return [
        request_for(kind, selection, request.metadata["layer_id"])
        for kind in ("attention", "moe")
    ]


def supports_baseline(selection, request):
    metadata = selection.model_metadata
    if metadata.get("model_type") != "deepseek_v4":
        return SupportResult(False, "requires DeepSeek-V4 target model")
    if metadata.get("indexer_cache_mode") != "fp8":
        return SupportResult(False, "requires the declared FP8 indexer cache layout")
    return SupportResult(True)


def register_modules(registry):
    for kind, contract in CONTRACTS.items():
        methods = (
            ("initialize", "prepare_fmha_impl", "forward", "get_execution_capabilities")
            if kind == "model"
            else ("forward",)
        )
        registry.register_module(
            ModuleSpec(
                "rtp.dsv4." + kind,
                1,
                contract,
                methods,
                "rtp_llm.models.dsv4.builders:validate_instance",
            )
        )
        describe = {"model": "describe_model", "block": "describe_block"}.get(kind)
        registry.register_implementation(
            ModuleImplSpec(
                module_id="rtp.dsv4." + kind,
                impl_id=f"baseline.dsv4.{kind}.v1",
                api_version=1,
                builder=f"rtp_llm.models.dsv4.builders:build_{kind}",
                supported_devices={DeviceType.Cuda},
                predicate="rtp_llm.models.dsv4.specs:supports_baseline",
                priority=0,
                contract_id=contract,
                weight_format_id=WEIGHT_FORMAT,
                state_format_id=STATE_FORMAT,
                collective_protocol_id="rtp.dsv4.tp-reduced.v1",
                capabilities={"prefill"},
                auto_selectable=False,
                validate_initialized=(
                    "rtp_llm.models.dsv4.builders:validate_initialized"
                    if kind == "model"
                    else None
                ),
                describe_resources=(
                    "rtp_llm.models.dsv4.resources:fp8_allocator_inputs"
                    if kind == "model"
                    else None
                ),
                describe_build_requests=(
                    ("rtp_llm.models.dsv4.specs:" + describe) if describe else None
                ),
            )
        )
