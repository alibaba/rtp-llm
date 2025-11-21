#pragma once

#include "rtp_llm/cpp/kernels/atex/common.cuh"

namespace atex {
namespace thread {

template<uint32_t VPT>
__device__ __forceinline__ fp32_t reduce_sum(const fp32_t* const values) {
    fp32_t local_sum = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < VPT; i++) {
        local_sum += values[i];
    }
    return local_sum;
}

template<uint32_t VPT>
__device__ __forceinline__ fp16_t reduce_sum(const fp16x2_t* const values) {
    fp16x2_t local_sum = {0.0f, 0.0f};

#pragma unroll
    for (uint32_t i = 0; i < VPT; i++) {
        local_sum += values[i];
    }
    return local_sum.x + local_sum.y;
}

template<uint32_t VPT>
__device__ __forceinline__ bf16_t reduce_sum(const bf16x2_t* const values) {
    bf16x2_t local_sum = {0.0f, 0.0f};

#pragma unroll
    for (uint32_t i = 0; i < VPT; i++) {
        local_sum += values[i];
    }
    return local_sum.x + local_sum.y;
}

template<uint32_t VPT>
__device__ __forceinline__ fp16x2_t reduce_absmax(fp16x2_t& init_max, const fp16x2_t* const values) {
#pragma unroll
    for (uint32_t i = 0; i < VPT; i++) {
        init_max = __hmax2(__habs2(values[i]), init_max);
    }
    return init_max;
}

template<uint32_t VPT>
__device__ __forceinline__ bf16x2_t reduce_absmax(bf16x2_t& init_max, const bf16x2_t* const values) {
#pragma unroll
    for (uint32_t i = 0; i < VPT; i++) {
        init_max = __hmax2(__habs2(values[i]), init_max);
    }
    return init_max;
}

}  // namespace thread
}  // namespace atex