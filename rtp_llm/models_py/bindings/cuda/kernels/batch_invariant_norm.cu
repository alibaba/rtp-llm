// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM 0.29.0 csrc/libtorch_stable/layernorm_kernels.cu.
// See models_py/modules/batch_invariant/UPSTREAM.md for provenance.

#include "rtp_llm/models_py/bindings/cuda/kernels/batch_invariant_norm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda/std/functional>
#include <algorithm>
#include <limits>

namespace torch_ext {
namespace {
#if CUB_VERSION >= 200800
using Add = cuda::std::plus<>;
#else
using Add = cub::Sum;
#endif

__global__ void post_residual_norm(const __nv_bfloat16* __restrict__ input,
                                   const __nv_bfloat16* __restrict__ weight,
                                   __nv_bfloat16* __restrict__ output,
                                   const float epsilon,
                                   const int   hidden_size) {
    const int64_t    offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
    __shared__ float inverse_rms;
    float            variance = 0.0f;
    // vLLM's batch-invariant residual path deliberately uses scalar lanes,
    // including when vector loads would be aligned. Do not retune by row count.
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = __bfloat162float(input[offset + idx]);
        variance += x * x;
    }
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage storage;
    variance = BlockReduce(storage).Reduce(variance, Add{}, blockDim.x);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x              = __bfloat162float(input[offset + idx]);
        float w              = __bfloat162float(weight[idx]);
        output[offset + idx] = __float2bfloat16(x * inverse_rms * w);
    }
}
}  // namespace

at::Tensor batch_invariant_post_residual_rmsnorm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && input.device() == weight.device());
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16);
    TORCH_CHECK(input.dim() == 2 && input.size(1) > 0 && input.is_contiguous());
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1) && weight.is_contiguous());
    TORCH_CHECK(input.size(1) <= std::numeric_limits<int>::max());
    TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max());
    const c10::cuda::CUDAGuard guard(input.device());
    auto                       output = at::empty_like(input);
    if (input.size(0)) {
        const int width = static_cast<int>(input.size(1));
        post_residual_norm<<<input.size(0), std::min(width, 1024), 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(epsilon),
            width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
}  // namespace torch_ext
