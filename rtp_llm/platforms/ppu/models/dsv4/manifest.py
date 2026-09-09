"""Public PPU module descriptions; no PPU kernel or model imports."""

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models.dsv4.specs import CONTRACTS, STATE_FORMAT_FP4, WEIGHT_FORMAT
from rtp_llm.models_py.pluggable.spec import ModuleImplSpec, SupportResult

# Explicit launch combinations; these are requirements, not rewritten defaults.
# Existing ModuleDispatchConfig snapshots and rank digests own the configuration.
PREFILL_EXECUTION_OPTIONS = {
    "DSV4_HC_IMPL": "hybrid",
    "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic",
    "DSV4_MHC_POST_BACKEND": "tilelang",
    "DSV4_MHC_POST_PDL": "0",
    "DSV4_PPU_SGLANG_MOE": "1",
    "DSV4_PPU_SGLANG_WO_A": "1",
    "DSV4_MOE_SHARED_EXPERT_OVERLAP": "0",
    "DSV4_PPU_TP_COMM_WARMUP": "1",
    "DSV4_PPU_SHARED_QKV_QUANT": "0",
    "DSV4_MOE_SCALE_GATHER_FUSED": "1",
}

DECODE_EXECUTION_OPTIONS = {
    "DSV4_HC_IMPL": "hybrid",
    "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic",
    "DSV4_MHC_POST_BACKEND": "tilelang",
    "DSV4_MHC_POST_PDL": "0",
    "DSV4_PPU_SGLANG_MOE": "1",
    "DSV4_PPU_SGLANG_WO_A": "1",
    "DSV4_MOE_SHARED_EXPERT_OVERLAP": "0",
    "DSV4_PPU_TP_COMM_WARMUP": "0",
    "DSV4_PPU_DECODE_HC_REDUCTION": "fused",
    "DSV4_PPU_DECODE_HC_NORM": "fused",
    "DSV4_PPU_DECODE_FP8_QUANT": "v2",
    "DSV4_PPU_DECODE_QKV": "merged",
    "DSV4_PPU_DECODE_INDEXER": "overlap",
    "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
    "DSV4_SHARED_EXPERT_MODE": "overlap",
    "DSV4_PPU_DECODE_SHARED_SCHEDULE": "before_route",
    "DSV4_PPU_DECODE_MOE_OUTPUT": "bf16",
    "DSV4_PPU_DECODE_METADATA": "graph_fused",
    "DSV4_PPU_DECODE_ROPE": "shared",
    "DSV4_PPU_DECODE_MOE_HINT": "capacity",
}


def _check_execution_options(options, required):
    for key, expected in required.items():
        if options.get(key) != expected:
            return SupportResult(
                False, f"requires {key}={expected}; use the PPU launch template"
            )
    return SupportResult(True)


def _supports_ppu_prefill(selection, request, cache_mode):
    metadata = selection.model_metadata
    options = metadata.get("execution_options", {})
    checks = (
        (selection.platform.device_name == "ZW-M890P", "requires ZW-M890P"),
        (
            metadata.get("model_type") == "deepseek_v4",
            "requires DeepSeek-V4 target model",
        ),
        (
            metadata.get("tp_size") == 4 and metadata.get("world_size") == 4,
            "requires TP4 world4",
        ),
        (
            metadata.get("ep_size") == 1
            and metadata.get("dp_size") == 1
            and metadata.get("pp_size") == 1,
            "requires EP1 DP1 PP1",
        ),
        (
            options.get("DSV4_HC_IMPL", "hybrid") in ("hybrid", "tilelang"),
            "requires the PPU HC implementation",
        ),
        (
            options.get("DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm_deterministic")
            in ("deepgemm", "deepgemm_deterministic"),
            "unsupported PPU HC prenorm backend",
        ),
        (
            options.get("DSV4_MHC_POST_BACKEND", "tilelang") == "tilelang"
            and options.get("DSV4_MHC_POST_PDL", "0") == "0",
            "requires TileLang POST without PDL",
        ),
        (not metadata.get("cp_enabled"), "CP is not qualified"),
        (metadata.get("role") == "PDFUSION", "PD/decode roles are not qualified"),
        (not metadata.get("speculative"), "MTP/speculation is not qualified"),
        (not metadata.get("cuda_graph"), "graph execution is not qualified"),
        (not metadata.get("reuse_cache"), "prefix reuse is not qualified"),
        (not metadata.get("lora"), "LoRA is not supported by PPU weight layouts"),
        (not metadata.get("eplb"), "EPLB is not qualified"),
        (
            metadata.get("indexer_cache_mode") == cache_mode,
            f"requires {cache_mode.upper()} indexer cache",
        ),
        (metadata.get("fp8_kv_cache") is True, "requires FP8 KV cache"),
        (
            metadata.get("hidden_size") == 4096 and metadata.get("num_layers") == 43,
            "requires Flash 43-layer/4096 model",
        ),
    )
    for supported, reason in checks:
        if not supported:
            return SupportResult(False, reason)
    return SupportResult(True)


def supports_ppu_fp4_prefill(selection, request):
    result = _supports_ppu_prefill(selection, request, "fp4")
    if not result.supported:
        return result
    return _check_execution_options(
        selection.model_metadata.get("execution_options", {}), PREFILL_EXECUTION_OPTIONS
    )


def _supports_ppu_tp_moe(selection, request, cache_mode):
    base = supports_ppu_fp4_prefill(selection, request)
    if not base.supported:
        return base
    options = selection.model_metadata.get("execution_options", {})
    if options.get("DSV4_PPU_SGLANG_MOE", "0") != "1":
        return SupportResult(
            False, "TP MoE v2 requires the SG fused activation boundary"
        )
    if (
        options.get("DSV4_MOE_SHARED_EXPERT_OVERLAP", "0") != "0"
        or options.get("DSV4_SHARED_EXPERT_MODE", "sequential") != "sequential"
    ):
        return SupportResult(
            False, "TP MoE v2 currently uses sequential shared compute"
        )
    if options.get("DSV4_MOE_GATHER_FUSED", "1") != "1":
        return SupportResult(False, "TP MoE v2 requires fused route gathering")
    return SupportResult(True)


def supports_ppu_fp4_tp_moe(selection, request):
    return _supports_ppu_tp_moe(selection, request, "fp4")


def supports_ppu_fp4_decode(selection, request):
    """Explicit integration candidate; whole-model qualification is external."""
    metadata = selection.model_metadata
    options = metadata.get("execution_options", {})
    comm = metadata.get("moe_communication", {})
    checks = (
        (selection.platform.device_name == "ZW-M890P", "requires ZW-M890P"),
        (metadata.get("model_type") == "deepseek_v4", "requires DeepSeek-V4"),
        (
            tuple(
                metadata.get(k)
                for k in ("tp_size", "dp_size", "ep_size", "pp_size", "world_size")
            )
            == (1, 8, 8, 1, 8),
            "requires TP1 DP8 EP8 PP1 world8",
        ),
        (metadata.get("role") == "DECODE", "requires Decode resources"),
        (
            metadata.get("cache_geometry", {}).get("kernel_tokens_per_block") == 256,
            "requires 256 raw tokens per kernel block (CSA/Indexer 64, HCA 2)",
        ),
        (
            metadata.get("hidden_size") == 4096 and metadata.get("num_layers") == 43,
            "requires Flash 43-layer/4096 model",
        ),
        (metadata.get("indexer_cache_mode") == "fp4", "requires FP4 indexer cache"),
        (metadata.get("fp8_kv_cache") is True, "requires FP8 KV cache"),
        (not metadata.get("cp_enabled"), "CP is not supported by this candidate"),
        (not metadata.get("speculative"), "requires MTP/speculation off"),
        (not metadata.get("reuse_cache"), "prefix reuse is not qualified"),
        (not metadata.get("lora"), "LoRA is not supported by PPU weight layouts"),
        (not metadata.get("eplb"), "EPLB is not qualified"),
        (
            comm.get("enabled") is True
            and comm.get("low_latency") is True
            and comm.get("all_gather") is False
            and comm.get("ffn_disaggregate") is False,
            "requires engine-owned DeepEP low-latency without FFN disaggregation",
        ),
        (
            type(comm.get("max_generate_batch_size")) is int
            and 0 < comm["max_generate_batch_size"] <= 128,
            "requires a Decode batch capacity in 1..128",
        ),
        (
            metadata.get("cuda_graph") is True,
            "requires the measured Decode Graph configuration",
        ),
        (not comm.get("internode", False), "requires single-node DeepEP"),
    )
    for supported, reason in checks:
        if not supported:
            return SupportResult(False, reason)
    return _check_execution_options(options, DECODE_EXECUTION_OPTIONS)


def register_modules(registry):
    for kind, contract in CONTRACTS.items():
        describe = {"model": "describe_model", "block": "describe_block"}.get(kind)
        builder = {
            "model": "build_decode_model",
            "attention": "build_attention_fp4",
            "moe": "build_decode_moe",
        }.get(kind, "build_" + kind)
        registry.register_implementation(
            ModuleImplSpec(
                module_id="rtp.dsv4." + kind,
                impl_id=f"ppu.dsv4.{kind}.fp4_decode.v1",
                api_version=1,
                builder="rtp_llm.platforms.ppu.models.dsv4.pluggable_builders:"
                + builder,
                supported_devices={DeviceType.Ppu},
                predicate="rtp_llm.platforms.ppu.models.dsv4.manifest:supports_ppu_fp4_decode",
                priority=100,
                contract_id=contract,
                weight_format_id=WEIGHT_FORMAT,
                state_format_id=STATE_FORMAT_FP4,
                validate_initialized=(
                    "rtp_llm.models.dsv4.builders:validate_initialized"
                    if kind == "model"
                    else None
                ),
                describe_resources=(
                    "rtp_llm.models.dsv4.resources:fp4_allocator_inputs"
                    if kind == "model"
                    else None
                ),
                collective_protocol_id="ppu.dsv4.decode.tp1-dp8-ep8-mxfp4-ll.v1",
                capabilities={"decode"},
                auto_selectable=False,
                describe_build_requests=(
                    "rtp_llm.models.dsv4.specs:" + describe if describe else None
                ),
            )
        )
    for kind, contract in CONTRACTS.items():
        describe = {"model": "describe_model", "block": "describe_block"}.get(kind)
        builder = {"attention": "build_attention_fp4", "moe": "build_moe_tp"}.get(
            kind, "build_" + kind
        )
        registry.register_implementation(
            ModuleImplSpec(
                module_id="rtp.dsv4." + kind,
                impl_id=f"ppu.dsv4.{kind}.fp4_indexer.v1",
                api_version=1,
                builder="rtp_llm.platforms.ppu.models.dsv4.pluggable_builders:"
                + builder,
                supported_devices={DeviceType.Ppu},
                predicate="rtp_llm.platforms.ppu.models.dsv4.manifest:"
                + (
                    "supports_ppu_fp4_tp_moe"
                    if kind == "moe"
                    else "supports_ppu_fp4_prefill"
                ),
                priority=100,
                contract_id=contract,
                weight_format_id=WEIGHT_FORMAT,
                state_format_id=STATE_FORMAT_FP4,
                validate_initialized=(
                    "rtp_llm.models.dsv4.builders:validate_initialized"
                    if kind == "model"
                    else None
                ),
                describe_resources=(
                    "rtp_llm.models.dsv4.resources:fp4_allocator_inputs"
                    if kind == "model"
                    else None
                ),
                prepare_weights=(
                    "rtp_llm.platforms.ppu.models.dsv4.resources:routed_tp_preparation"
                    if kind == "moe"
                    else None
                ),
                collective_protocol_id=(
                    "ppu.dsv4.moe.tp4-shared-sharded-bf16-reduce.v2"
                    if kind == "moe"
                    else "ppu.dsv4.fp4-indexer.tp4-bf16.v1"
                ),
                capabilities={"prefill"},
                auto_selectable=False,
                describe_build_requests=(
                    "rtp_llm.models.dsv4.specs:" + describe if describe else None
                ),
            )
        )
