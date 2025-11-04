#progma once

#include "common.cuh"
#include "warp/reduce.cuh"

namespace atex {
namespace block {

template<uint32_t TPB, bool boardcast>
__device__ __forceinline__ fp32_t reduce_sum(fp32_t local_sum, fp32_t* smem) {
    // the shape of smem should be 1d vector that contains at least TPB / w elements
    const uint32_t tid       = threadIdx.x;
    const uint32_t warp_idx  = tid / warpSize;
    const uint32_t warp_lane = tid % warpSize;

    static_assert(TPB % warpSize == 0);
    static_assert(TPB <= 1024);

    constexpr uint32_t num_of_wrap = TPB / warpSize;

    fp32_t warp_sum = atex::warp::reduce_sum(local_sum);
    if (warp_lane == 0) {
        smem[warp_idx] = warp_sum;
    }
    __syncthreads();

    if (warp_idx == 0) {
        local_sum         = warp_lane < num_of_wrap ? smem[warp_lane] : 0.0f;
        fp32_t global_sum = atex::warp::reduce_sum(local_sum);
    }

    if constexpr (boardcast) {
        if (warp_idx == 0 && warp_lane == 0) {
            smem[0] = global_sum
        }
        __syncthreads();

        return smem[0];
    }

    return global_sum;
}

}  // namespace block
}  // namespace atex