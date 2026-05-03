#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_ieee_rmsnorm.h"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

#include <cuda_bf16.h>

namespace rtp_llm {

// ============================================================================
// Vendored from: flashinfer ``norm.cuh::RMSNormKernel<VEC_SIZE=8, T=bf16>``
// (flashinfer-python 0.6.9, in this repo at
//  3rdparty/flashinfer/include/flashinfer/norm.cuh:36-111, host launcher
//  ``RMSNorm<__nv_bfloat16>`` at lines 113-146).
//
// MODIFIED RANGE: ONLY the rsqrt computation. Flashinfer uses
//   ``float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);``   (line 87)
// where ``math::rsqrt`` (math.cuh:117) is
//   ``asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : ...);``      (~2 ULP, FTZ)
// We replace it with IEEE-rounded:
//   ``s_rms_rcp = __frcp_rn(__fsqrt_rn(sum_sq / dim + eps));``   (0.5 ULP, no FTZ)
//
// Everything else (tile shape, sum-of-squares reduce, vec_t<bf16,8> loads,
// per-row block-per-row launch) follows flashinfer 1:1, expressed here in
// plain ``float4`` reinterpret + reduce_kernel_utils.cuh::blockReduceSum
// because we don't want to drag in the rest of flashinfer's vec_dtypes.cuh
// /utils.cuh / trtllm dependency tree just for one kernel.
//
// Why only the rsqrt diff matters: see dsv4_rmsnorm.py for the full bisect
// writeup — at S=64K + 60-layer DSv4 + tiny attn_norm.weight (~0.03), the
// approx PTX rsqrt's ~2 ULP per-call accumulates into a directional ~0.5
// logit shift on a specific token, flipping argmax and producing
// context-leakage gibberish.
// ============================================================================
template<int BLOCK_SIZE>
__global__ void dsv4IeeeRmsNormKernel(__nv_bfloat16* __restrict__ output,
                                      const __nv_bfloat16* __restrict__ input,
                                      const __nv_bfloat16* __restrict__ weight,
                                      const int   dim,
                                      const float eps) {
    constexpr int VEC_SIZE = 8;  // 8 * 2B = 16B = float4
    const int     bx       = blockIdx.x;
    const int     tx       = threadIdx.x;
    const int     n_vec    = dim / VEC_SIZE;  // ints per row, vector units
    const int     stride   = BLOCK_SIZE;

    const float4* row_in_v4  = reinterpret_cast<const float4*>(input + bx * dim);
    const float4* w_v4       = reinterpret_cast<const float4*>(weight);
    float4*       row_out_v4 = reinterpret_cast<float4*>(output + bx * dim);

    // Pass 1: accumulate sum(x^2) in fp32.
    float sum_sq = 0.f;
    for (int i = tx; i < n_vec; i += stride) {
        float4         packed = row_in_v4[i];
        __nv_bfloat16* x_bf16 = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float v = __bfloat162float(x_bf16[j]);
            sum_sq += v * v;
        }
    }

    // Block-wide reduce sum_sq.
    sum_sq = blockReduceSum<float>(sum_sq);

    __shared__ float s_rms_rcp;
    if (tx == 0) {
        // === THE PATCH (only deviation from flashinfer norm.cuh) ===
        s_rms_rcp = __frcp_rn(__fsqrt_rn(sum_sq / static_cast<float>(dim) + eps));
    }
    __syncthreads();
    const float rms_rcp = s_rms_rcp;

    // Pass 2: normalize and weight.
    for (int i = tx; i < n_vec; i += stride) {
        float4         packed_x = row_in_v4[i];
        float4         packed_w = w_v4[i];
        float4         packed_o;
        __nv_bfloat16* x_bf16 = reinterpret_cast<__nv_bfloat16*>(&packed_x);
        __nv_bfloat16* w_bf16 = reinterpret_cast<__nv_bfloat16*>(&packed_w);
        __nv_bfloat16* o_bf16 = reinterpret_cast<__nv_bfloat16*>(&packed_o);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float xv  = __bfloat162float(x_bf16[j]);
            float wv  = __bfloat162float(w_bf16[j]);
            o_bf16[j] = __float2bfloat16(xv * rms_rcp * wv);
        }
        row_out_v4[i] = packed_o;
    }
}

void invokeDsv4IeeeRmsNorm(__nv_bfloat16*       output,
                           const __nv_bfloat16* input,
                           const __nv_bfloat16* weight,
                           int                  num_tokens,
                           int                  dim,
                           float                eps,
                           cudaStream_t         stream) {
    // dim must be a multiple of 8 (16-byte vector loads).  Holds for V4-Flash
    // (dim=4096, head_dim=512, etc).
    constexpr int BLOCK_SIZE = 256;
    dim3          grid(num_tokens);
    dim3          block(BLOCK_SIZE);
    dsv4IeeeRmsNormKernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(output, input, weight, dim, eps);
}

}  // namespace rtp_llm

namespace torch_ext {

void dsv4_ieee_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, int64_t cuda_stream) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bf16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bf16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    const int64_t N = input.size(0);
    const int64_t D = input.size(1);
    TORCH_CHECK(output.size(0) == N && output.size(1) == D, "output shape mismatch");
    TORCH_CHECK(weight.size(0) == D, "weight shape mismatch");
    TORCH_CHECK(D % 8 == 0, "D must be multiple of 8 (vec_size=8)");
    if (N == 0) {
        return;
    }
    rtp_llm::invokeDsv4IeeeRmsNorm(reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                                   reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                                   reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                                   static_cast<int>(N),
                                   static_cast<int>(D),
                                   static_cast<float>(eps),
                                   reinterpret_cast<cudaStream_t>(cuda_stream));
}

}  // namespace torch_ext
