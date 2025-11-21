#pragma once

#include "rtp_llm/cpp/kernels/atex/common.cuh"

namespace atex {
namespace warp {

__device__ __forceinline__ fp32_t reduce_max(fp32_t local_max) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
    asm("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(local_sum) : "f"(local_sum));
#else
#pragma unroll
    for (int offset = atex::warpSize / 2; offset > 0; offset /= 2) {
        fp32_t other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max        = fmaxf(local_max, other_max);
    }
#endif
    return local_max;
}

__device__ __forceinline__ fp32_t reduce_sum(fp32_t local_sum) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
    asm("redux.sync.sum.f32 %0, %1, 0xffffffff;\n" : "=f"(local_sum) : "f"(local_sum));
#else
#pragma unroll
    for (int offset = atex::warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
#endif
    return local_sum;
}

}  // namespace warp
}  // namespace atex