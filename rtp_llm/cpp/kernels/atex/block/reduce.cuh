#pragma once

#include "rtp_llm/cpp/kernels/atex/common.cuh"
#include "rtp_llm/cpp/kernels/atex/warp/reduce.cuh"

namespace atex {
namespace block {

template<uint32_t TPB, bool boardcast>
__device__ __forceinline__ fp32_t reduce_sum(fp32_t local_sum, fp32_t* smem) {
    // 使用此函数时，smem 的大小至少应该为 TPB / warpSize
    // 线程数必须是 warpSize 的整倍数

    const uint32_t tid       = threadIdx.x;
    const uint32_t warp_idx  = tid / warpSize;
    const uint32_t warp_lane = tid % warpSize;

    static_assert(TPB % warpSize == 0);
    static_assert(TPB <= 1024);

    constexpr uint32_t num_of_warp = TPB / warpSize;

    fp32_t warp_sum = atex::warp::reduce_sum(local_sum);
    if (warp_lane == 0) {
        smem[warp_idx] = warp_sum;
    }
    __syncthreads();

    fp32_t global_sum;
    if (warp_idx == 0) {
        local_sum  = warp_lane < num_of_warp ? smem[warp_lane] : 0.0f;
        global_sum = atex::warp::reduce_sum(local_sum);
    }

    if constexpr (boardcast) {
        if (warp_idx == 0 && warp_lane == 0) {
            smem[0] = global_sum;
        }

        __syncthreads();
        return smem[0];
    }

    // if not boardcast, only thread 0 have the valid result.
    return global_sum;
}

template<uint32_t TPB, bool boardcast>
__device__ __forceinline__ fp32_t reduce_max(fp32_t local_max, fp32_t* smem) {
    // 使用此函数时，smem 的大小至少应该为 TPB / warpSize
    // 线程数必须是 warpSize 的整倍数

    const uint32_t tid       = threadIdx.x;
    const uint32_t warp_idx  = tid / warpSize;
    const uint32_t warp_lane = tid % warpSize;

    static_assert(TPB % warpSize == 0);
    static_assert(TPB <= 1024);

    constexpr uint32_t num_of_warp = TPB / warpSize;

    fp32_t warp_max = atex::warp::reduce_max(local_max);
    if (warp_lane == 0) {
        smem[warp_idx] = warp_max;
    }
    __syncthreads();

    fp32_t global_max;
    if (warp_idx == 0) {
        local_max  = warp_lane < num_of_warp ? smem[warp_lane] : std::numeric_limits<fp32_t>::lowest();
        global_max = atex::warp::reduce_max(local_max);
    }

    if constexpr (boardcast) {
        if (warp_idx == 0 && warp_lane == 0) {
            smem[0] = global_max;
        }

        __syncthreads();
        return smem[0];
    }

    // if not boardcast, only thread 0 have the valid result.
    return global_max;
}

}  // namespace block
}  // namespace atex