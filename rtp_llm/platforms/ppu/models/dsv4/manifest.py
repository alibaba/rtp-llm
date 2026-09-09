"""Public PPU module descriptions; no PPU kernel or model imports."""

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models.dsv4.specs import (
    CONTRACTS,
    STATE_FORMAT,
    STATE_FORMAT_FP4,
    WEIGHT_FORMAT,
)
from rtp_llm.models_py.pluggable.spec import ModuleImplSpec, SupportResult


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


def supports_ppu_prefill(selection, request):
    return _supports_ppu_prefill(selection, request, "fp8")


def supports_ppu_fp4_prefill(selection, request):
    return _supports_ppu_prefill(selection, request, "fp4")


def _supports_ppu_tp_moe(selection, request, cache_mode):
    base = _supports_ppu_prefill(selection, request, cache_mode)
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


def supports_ppu_tp_moe(selection, request):
    return _supports_ppu_tp_moe(selection, request, "fp8")


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
            options.get("DSV4_HC_IMPL", "hybrid") in ("hybrid", "tilelang")
            and options.get("DSV4_MHC_PRE_GEMM_BACKEND", "deepgemm_deterministic")
            == "deepgemm_deterministic"
            and options.get("DSV4_MHC_POST_BACKEND", "tilelang") == "tilelang"
            and options.get("DSV4_MHC_POST_PDL", "0") == "0",
            "requires deterministic PPU HC with TileLang POST and PDL off",
        ),
        (
            options.get("DSV4_PPU_DECODE_HC_REDUCTION", "torch") in ("torch", "fused"),
            "requires torch or fused HC reduction",
        ),
        (
            options.get("DSV4_PPU_DECODE_HC_NORM", "separate") in ("separate", "fused"),
            "requires separate or fused HC norm",
        ),
        (
            options.get("DSV4_PPU_DECODE_FP8_QUANT", "auto") in ("auto", "v2"),
            "requires auto or v2 FP8 quantization",
        ),
        (
            options.get("DSV4_PPU_DECODE_QKV", "separate") in ("separate", "merged")
            and (
                options.get("DSV4_PPU_DECODE_QKV", "separate") != "merged"
                or options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential") == "overlap"
            ),
            "requires separate QKV or merged QKV with Attention overlap",
        ),
        (
            options.get("DSV4_PPU_DECODE_INDEXER", "sequential")
            in ("sequential", "overlap")
            and (
                options.get("DSV4_PPU_DECODE_INDEXER", "sequential") != "overlap"
                or options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential") == "overlap"
            ),
            "requires sequential Indexer or Indexer overlap with Attention overlap",
        ),
        (
            options.get("DSV4_PPU_DECODE_METADATA", "eager")
            in ("eager", "graph", "graph_fused"),
            "requires eager, graph or graph_fused Decode metadata",
        ),
        (
            options.get("DSV4_PPU_DECODE_ROPE", "layer") in ("layer", "shared")
            and (
                options.get("DSV4_PPU_DECODE_ROPE", "layer") != "shared"
                or (
                    options.get("DSV4_PPU_DECODE_METADATA", "eager")
                    in ("graph", "graph_fused")
                    and options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
                    == "overlap"
                )
            ),
            "requires layer RoPE or shared RoPE with metadata Graph and Attention overlap",
        ),
        (
            options.get("DSV4_PPU_DECODE_SHARED_SCHEDULE", "after_route")
            in ("after_route", "before_route")
            and (
                options.get("DSV4_PPU_DECODE_SHARED_SCHEDULE", "after_route")
                != "before_route"
                or options.get("DSV4_SHARED_EXPERT_MODE", "sequential") == "overlap"
            ),
            "requires after_route or an overlapped before_route shared schedule",
        ),
        (
            options.get("DSV4_PPU_DECODE_MOE_HINT", "capacity")
            in ("capacity", "batch"),
            "requires capacity or batch MoE launch hint",
        ),
        (
            options.get("DSV4_PPU_DECODE_MOE_OUTPUT", "fp32") in ("fp32", "bf16"),
            "requires fp32 or bf16 MoE output storage",
        ),
        (
            options.get("DSV4_PPU_SGLANG_MOE", "0") == "1"
            and options.get("DSV4_MOE_SHARED_EXPERT_OVERLAP", "0") == "0"
            and options.get("DSV4_SHARED_EXPERT_MODE", "sequential")
            in ("sequential", "overlap")
            and options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
            in ("sequential", "overlap"),
            "requires SG activation and a supported PPU shared expert executor",
        ),
    )
    for supported, reason in checks:
        if not supported:
            return SupportResult(False, reason)
    return SupportResult(True)


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
    for kind, contract in CONTRACTS.items():
        describe = {"model": "describe_model", "block": "describe_block"}.get(kind)
        registry.register_implementation(
            ModuleImplSpec(
                module_id="rtp.dsv4." + kind,
                impl_id=f"ppu.dsv4.{kind}.v1",
                api_version=1,
                builder=f"rtp_llm.platforms.ppu.models.dsv4.pluggable_builders:build_{kind}",
                supported_devices={DeviceType.Ppu},
                predicate="rtp_llm.platforms.ppu.models.dsv4.manifest:supports_ppu_prefill",
                priority=100,
                contract_id=contract,
                weight_format_id=WEIGHT_FORMAT,
                state_format_id=STATE_FORMAT,
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
                prepare_weights=(
                    "rtp_llm.platforms.ppu.models.dsv4.resources:routed_tp_preparation"
                    if kind == "moe"
                    else None
                ),
                collective_protocol_id="ppu.dsv4.tp4-fp32-routed-replicated-shared.v1",
                capabilities={"prefill"},
                auto_selectable=False,
                describe_build_requests=(
                    ("rtp_llm.models.dsv4.specs:" + describe) if describe else None
                ),
            )
        )
    registry.register_implementation(
        ModuleImplSpec(
            module_id="rtp.dsv4.moe",
            impl_id="ppu.dsv4.moe.tp_shared_bf16.v2",
            api_version=1,
            builder="rtp_llm.platforms.ppu.models.dsv4.pluggable_builders:build_moe_tp",
            supported_devices={DeviceType.Ppu},
            predicate="rtp_llm.platforms.ppu.models.dsv4.manifest:supports_ppu_tp_moe",
            priority=100,
            contract_id=CONTRACTS["moe"],
            weight_format_id=WEIGHT_FORMAT,
            state_format_id=STATE_FORMAT,
            prepare_weights="rtp_llm.platforms.ppu.models.dsv4.resources:routed_tp_preparation",
            collective_protocol_id="ppu.dsv4.moe.tp4-shared-sharded-bf16-reduce.v2",
            capabilities={"prefill"},
            auto_selectable=False,
        )
    )
    registry.register_implementation(
        ModuleImplSpec(
            module_id="rtp.dsv4.attention",
            impl_id="ppu.dsv4.attention.inverse_rope.v2",
            api_version=1,
            builder="rtp_llm.platforms.ppu.models.dsv4.pluggable_builders:build_attention_inverse_rope",
            supported_devices={DeviceType.Ppu},
            predicate="rtp_llm.platforms.ppu.models.dsv4.manifest:supports_ppu_prefill",
            priority=100,
            contract_id=CONTRACTS["attention"],
            weight_format_id=WEIGHT_FORMAT,
            state_format_id=STATE_FORMAT,
            collective_protocol_id="ppu.dsv4.attention.tp4-bf16.v1",
            capabilities={"prefill"},
            auto_selectable=False,
        )
    )
