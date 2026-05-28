#pragma once

#include <torch/all.h>

// FP8 PER_BLOCK GEMM for sm_120 family (consumer Blackwell: RTX 5090 / RTX PRO
// 5000 / 6000). Based on vLLM's scaled_mm_blockwise_sm120_fp8 (Apache 2.0,
// commit 6f955986e). DeepGEMM 2.1.x ships no sm_120 cubin; FlashInfer's
// blockwise FP8 path stalls at <200 TFLOPs on sm_120 (see deep-dives/
// fp8-gemm-sm120.md). This kernel hits ~442 TFLOPs at 4096^3 (1.78x BF16).
//
// Inputs:
//   D: (M, N) bf16/fp16 row-major
//   A: (M, K) float8_e4m3fn row-major
//   B: (N, K) float8_e4m3fn row-major contiguous weight.  CUTLASS sm120
//      blockwise uses RowMajor-style packed strides for B (K-stride=1,
//      N-stride=K) despite LayoutB=ColumnMajor in the template.
//   A_sf: per-token-group scale, MN-major layout (output of
//         sgl_per_token_group_quant_fp8 with column_major_scales=True,
//         scale_tma_aligned=False)
//   B_sf: per-block weight scale, K-major layout (shape (N/128, K/128) ->
//         flattened consistently with cutlass Sm120BlockwiseScaleConfig)
void cutlass_scaled_mm_blockwise_sm120_fp8(torch::Tensor&       D,
                                           torch::Tensor const& A,
                                           torch::Tensor const& B,
                                           torch::Tensor const& A_sf,
                                           torch::Tensor const& B_sf);
