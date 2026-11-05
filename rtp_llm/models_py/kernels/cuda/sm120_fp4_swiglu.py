"""SM120 dense up GEMM, SwiGLU and NVFP4 output under the existing fusion gate."""

import functools

import torch


def prepare_weights(weight, scales):
    """Interleave gate/up rows once, preserving the physical 128x4 SF layout."""
    n, packed_k = weight.shape
    indices = torch.arange(n, device=weight.device).reshape(2, n // 2).T.flatten()
    raw_scales = (
        scales.reshape(n // 128, packed_k // 32, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(n, packed_k // 8)
    )
    interleaved_scales = (
        raw_scales[indices]
        .reshape(n // 128, 4, 32, packed_k // 32, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(scales.shape)
    )
    return weight[indices].contiguous(), interleaved_scales


@functools.lru_cache(maxsize=None)
def _compile(device, small):
    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import get_max_active_clusters, make_ptr

    from .sm120_fp4_swiglu_kernel import FusedGemmKernel

    with torch.cuda.device(device):
        m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()

        def fake(dtype, shape, alignment=16):
            return cute.runtime.make_fake_compact_tensor(
                dtype,
                shape,
                stride_order=tuple(reversed(range(len(shape)))),
                assumed_align=alignment,
            )

        gemm = FusedGemmKernel(16, (16, 64) if small else (64, 64), (1, 1), False)
        # Keep these dimensions symbolic, matching FlashInfer's production
        # compiler interface. Static M=1 produces an invalid TMA descriptor.
        return cute.compile(
            gemm.wrapper,
            fake(cutlass.Uint8, (m, k), 32),
            fake(cutlass.Uint8, (n, k), 32),
            fake(cutlass.Uint8, (m, cute.sym_int())),
            fake(cutlass.Uint8, (cute.sym_int(),)),
            fake(cutlass.Float32, (1,), 4),
            1,
            136,
            80,
            1,
            make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16),
            make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16),
            fake(cutlass.Float32, (1,), 4),
            get_max_active_clusters(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            False,
            options="--opt-level 2 --enable-tvm-ffi",
        )


def fused_swiglu_fp4(layer, input_fp4, input_sf, output_scale):
    """Return packed SwiGLU output; the caller checks shape, dtype and gate."""
    m = input_fp4.shape[0]
    weight, weight_sf = layer._swiglu_weights
    output = torch.empty((m, 4352), device=input_fp4.device, dtype=torch.uint8)
    scales = torch.empty((128 * 544,), device=input_fp4.device, dtype=torch.uint8)
    _compile(input_fp4.device, m <= 32)(
        input_fp4,
        weight,
        output,
        scales,
        output_scale.reshape(1),
        1,
        136,
        80,
        input_sf.data_ptr(),
        weight_sf.data_ptr(),
        layer.alpha.reshape(1),
    )
    return output, scales.reshape(128, 544)
