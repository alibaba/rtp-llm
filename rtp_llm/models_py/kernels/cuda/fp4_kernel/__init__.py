from rtp_llm.models_py.utils.arch import is_cuda


def _fp4_unavailable(*args, **kwargs):
    # A CUDA build without the registered op exports a callable failure by
    # design so accidental FP4 selection reports the missing ENABLE_FP4 flag.
    # The CPU-only fallback unit test treats this message as part of the API.
    raise RuntimeError(
        "FP4 kernels are unavailable because this build did not enable "
        "ENABLE_FP4; use a CUDA 12.9+ or CUDA 13 build configuration."
    )


if is_cuda():
    from rtp_llm.ops import compute_ops as _compute_ops

    # The staged capability contract is intentional for compatibility:
    # registration detects ENABLE_FP4 here, grouped CUDA ops reject unsupported
    # SMs at use, legacy wrappers rely on their existing caller-side SM gates,
    # and non-CUDA imports export None for feature probes. Do not use an export's
    # None-ness as a new API; audit callers before adding one capability predicate.
    if hasattr(_compute_ops, "silu_and_mul_scaled_fp4_experts_quant"):
        from .fp4_kernel import cutlass_scaled_fp4_mm_wrapper
        from .fp4_kernel import scaled_fp4_quant_wrapper
        from .flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
        from .flashinfer_cutedsl_moe import scaled_fp4_grouped_quant
        from .flashinfer_cutedsl_moe import silu_and_mul_scaled_fp4_grouped_quant
    else:
        cutlass_scaled_fp4_mm_wrapper = _fp4_unavailable
        scaled_fp4_quant_wrapper = _fp4_unavailable
        scaled_fp4_grouped_quant = _fp4_unavailable
        silu_and_mul_scaled_fp4_grouped_quant = _fp4_unavailable
        flashinfer_cutedsl_moe_masked = _fp4_unavailable
else:
    # Preserve the legacy non-CUDA None contract. This intentionally differs
    # from a CUDA build that omitted ENABLE_FP4, which must fail if called.
    cutlass_scaled_fp4_mm_wrapper = None
    scaled_fp4_quant_wrapper = None
    scaled_fp4_grouped_quant = None
    silu_and_mul_scaled_fp4_grouped_quant = None
    flashinfer_cutedsl_moe_masked = None

__all__ = [
    "cutlass_scaled_fp4_mm_wrapper",
    "scaled_fp4_quant_wrapper",
    "scaled_fp4_grouped_quant",
    "silu_and_mul_scaled_fp4_grouped_quant",
    "flashinfer_cutedsl_moe_masked",
]
