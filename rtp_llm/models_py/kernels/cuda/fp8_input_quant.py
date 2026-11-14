"""Shared per-tensor projection input quantization dispatch."""


def _quantize_projection_input(rows, *, role, scale=None, output=None):
    if scale is None and role in ("DECODE", "PDFUSION"):
        from rtp_llm.models_py.triton_kernels.common.fp8_small_quant import (
            try_small_fp8_quant,
        )

        result = try_small_fp8_quant(rows, output=output)
        if result is not None:
            return result
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant

    return scaled_fp8_per_tensor_quant(rows, scale=scale, output=output)
