"""Public PPU module descriptions; no PPU kernel or model imports."""

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.pluggable.dsv4_specs import (
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


def register_modules(registry):
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
                collective_protocol_id=(
                    "ppu.dsv4.moe.tp4-shared-sharded-bf16-reduce.v2"
                    if kind == "moe"
                    else "ppu.dsv4.fp4-indexer.tp4-bf16.v1"
                ),
                capabilities={"prefill"},
                auto_selectable=False,
                describe_build_requests=(
                    "rtp_llm.models_py.pluggable.dsv4_specs:" + describe
                    if describe
                    else None
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
                collective_protocol_id="ppu.dsv4.tp4-fp32-routed-replicated-shared.v1",
                capabilities={"prefill"},
                auto_selectable=False,
                describe_build_requests=(
                    ("rtp_llm.models_py.pluggable.dsv4_specs:" + describe)
                    if describe
                    else None
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
