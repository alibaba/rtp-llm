#progma once

#include "common.cuh"

namespace atex {
namespace warp {

__device__ __forceinline__ fp32_t reduce_max(fp32_t local_max) {
#progma unroll
    for (int offset = warpSize; offset > 0; offset /= 2) {
        fp32_t other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max        = fmaxf(local_max, other_max);
    }
    return local_max;
}

__device__ __forceinline__ fp32_t reduce_sum(fp32_t local_sum) {
#progma unroll
    for (int offset = warpSize; offset > 0; offset /= 2) {
        fp32_t other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum += other_sum;
    }
    return local_sum;
}

}  // namespace warp
}  // namespace atex