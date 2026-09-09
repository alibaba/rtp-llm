"""PPU construction entrypoints; selected after the worker protocol barrier."""

from rtp_llm.models.dsv4 import builders as baseline


def build_model(*, build_ctx, request, **kwargs):
    from .ppu_module_provider import PpuModuleProvider

    provider = PpuModuleProvider(
        build_ctx.selection.model_metadata["execution_options"]
    )
    provider.require_device_name(build_ctx.selection.platform.device_name)
    return baseline.build_model(
        build_ctx=build_ctx, request=request, platform_provider=provider, **kwargs
    )


def build_block(*, build_ctx, request, **kwargs):
    return baseline.build_block(build_ctx=build_ctx, request=request, **kwargs)


def build_decode_model(*, build_ctx, request, **kwargs):
    from .ppu_decode_provider import PpuDecodeProvider

    provider = PpuDecodeProvider(
        build_ctx.selection.model_metadata["execution_options"]
    )
    provider.require_device_name(build_ctx.selection.platform.device_name)
    return baseline.build_model(
        build_ctx=build_ctx, request=request, platform_provider=provider, **kwargs
    )


def build_decode_moe(*, build_ctx, request, platform_provider, **kwargs):
    from .ppu_deepep_fp4 import PpuDeepEPFP4Strategy

    if (
        kwargs.get("tp_size") != 1
        or kwargs.get("ep_size") != 8
        or not kwargs.get("is_decode_role")
    ):
        raise ValueError("PPU Decode MoE requires TP1/EP8 Decode resources")
    return baseline.build_moe(
        build_ctx=build_ctx,
        request=request,
        platform_provider=platform_provider,
        execution_options=build_ctx.selection.model_metadata["execution_options"],
        strategy_type=PpuDeepEPFP4Strategy,
        strategy_kwargs={
            "expected_m_policy": platform_provider._moe_hint,
            "output_dtype": platform_provider._moe_output_dtype,
        },
        **kwargs,
    )


def build_moe_tp(*, build_ctx, request, tp_rank, **kwargs):
    from .ppu_tp_moe import PpuTPMoE

    baseline.validate_arguments(build_ctx, request, kwargs)
    return PpuTPMoE(tp_rank=tp_rank, **kwargs)


def build_attention_fp4(*, build_ctx, request, platform_provider, **kwargs):
    from .ppu_fp4_indexer import PpuFP4Attention

    baseline.validate_arguments(build_ctx, request, kwargs)
    if platform_provider is None:
        raise ValueError(
            "PPU FP4 attention requires an instance-owned operator adapter"
        )
    return platform_provider.build_attention(PpuFP4Attention, **kwargs)
