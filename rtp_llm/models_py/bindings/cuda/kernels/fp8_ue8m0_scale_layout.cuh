#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rtp_llm {

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    val = tid < (blockDim.x + 31) / 32 ? shared[lane] : 0.0f;
    if (warp == 0) {
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[256];
    const int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ __forceinline__ uint8_t scaleToUe8m0(float scale) {
    scale = fmaxf(scale, 1e-10f);
    const uint32_t bits = __float_as_uint(scale);
    int exp = static_cast<int>((bits >> 23) & 0xff);
    exp += (bits & 0x7fffff) != 0;
    exp = max(1, min(254, exp));
    return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ float ue8m0ToScale(uint8_t packed_scale) {
    return __uint_as_float(static_cast<uint32_t>(packed_scale) << 23);
}

__device__ __forceinline__ void writeColumnMajorUe8m0Scale(uint32_t* __restrict__ output_s,
                                                           int row_idx,
                                                           int group_idx,
                                                           int scale_stride,
                                                           uint8_t packed_scale) {
    constexpr int elems_per_pack = static_cast<int>(sizeof(uint32_t) / sizeof(uint8_t));
    const int     col_idx        = group_idx / elems_per_pack;
    const int     pack_idx       = group_idx % elems_per_pack;
    auto*         scale_u8       = reinterpret_cast<uint8_t*>(output_s);
    scale_u8[col_idx * scale_stride * elems_per_pack + row_idx * elems_per_pack + pack_idx] = packed_scale;
}

}  // namespace rtp_llm
