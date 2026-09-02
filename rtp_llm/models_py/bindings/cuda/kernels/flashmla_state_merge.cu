#include "rtp_llm/models_py/bindings/cuda/kernels/flashmla_state_merge.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace rtp_llm {
namespace {

constexpr int kWarpSize      = 32;
constexpr int kHeadsPerBlock = 8;
constexpr int kThreads       = kWarpSize * kHeadsPerBlock;
constexpr int kHeadSize      = 128;
constexpr int kValuesPerLane = kHeadSize / kWarpSize;

template<typename T, int Elements>
struct alignas(sizeof(T) * Elements) AlignedVector {
    T values[Elements];
};

template<typename T, int Elements>
union VectorBits {
    AlignedVector<T, Elements> typed;
    uint2                      raw64;
    uint4                      raw128;
};

__device__ __forceinline__ uint2 loadGlobal64(const void* ptr) {
    uint2 value;
    asm volatile("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(value.x), "=r"(value.y) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ uint4 loadGlobal128(const void* ptr) {
    uint4 value;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr)
                 : "memory");
    return value;
}

__device__ __forceinline__ void storeGlobal64(void* ptr, uint2 value) {
    asm volatile("st.global.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(value.x), "r"(value.y) : "memory");
}

__device__ __forceinline__ void storeGlobal128(void* ptr, uint4 value) {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
                 : "memory");
}

template<typename T, int Elements>
__device__ __forceinline__ AlignedVector<T, Elements> loadVector(const T* ptr) {
    VectorBits<T, Elements> value;
    if constexpr (sizeof(T) * Elements == sizeof(uint4)) {
        value.raw128 = loadGlobal128(ptr);
    } else {
        static_assert(sizeof(T) * Elements == sizeof(uint2));
        value.raw64 = loadGlobal64(ptr);
    }
    return value.typed;
}

template<typename T, int Elements>
__device__ __forceinline__ void storeVector(T* ptr, const AlignedVector<T, Elements>& value) {
    VectorBits<T, Elements> bits;
    bits.typed = value;
    if constexpr (sizeof(T) * Elements == sizeof(uint4)) {
        storeGlobal128(ptr, bits.raw128);
    } else {
        static_assert(sizeof(T) * Elements == sizeof(uint2));
        storeGlobal64(ptr, bits.raw64);
    }
}

template<typename T>
__device__ __forceinline__ float loadAsFloat(const T* ptr) {
    return static_cast<float>(*ptr);
}

template<>
__device__ __forceinline__ float loadAsFloat(const __nv_bfloat16* ptr) {
    return __bfloat162float(*ptr);
}

template<typename T>
__device__ __forceinline__ void storeFromFloat(T* ptr, float value) {
    *ptr = static_cast<T>(value);
}

template<>
__device__ __forceinline__ void storeFromFloat(__nv_bfloat16* ptr, float value) {
    *ptr = __float2bfloat16_rn(value);
}

// Metadata is built from the immutable forward plan.
__device__ __forceinline__ int32_t resolveSegmentedToken(int32_t        partial_token,
                                                         const int32_t* partial_q_indptr,
                                                         const int32_t* destination_starts,
                                                         int32_t        num_partial_segments) {
    int32_t low  = 0;
    int32_t high = num_partial_segments;
    while (low < high) {
        const int32_t middle = low + (high - low) / 2;
        if (partial_token < partial_q_indptr[middle + 1]) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }

    const int32_t partial_begin = partial_q_indptr[low];
    return destination_starts[low] + partial_token - partial_begin;
}

template<typename OutputT>
__global__ void flashMlaStateMergeSegmented128Kernel(OutputT* __restrict__ output,
                                                     float* __restrict__ output_lse,
                                                     const __nv_bfloat16* __restrict__ partial_output,
                                                     const float* __restrict__ partial_lse,
                                                     const int32_t* __restrict__ partial_q_indptr,
                                                     const int32_t* __restrict__ destination_starts,
                                                     int32_t num_partial_segments,
                                                     int32_t num_heads,
                                                     int64_t output_lse_stride_token,
                                                     int64_t output_lse_stride_head,
                                                     int64_t partial_lse_stride_token,
                                                     int64_t partial_lse_stride_head) {
    __shared__ int32_t destination;
    const int32_t      partial_token = static_cast<int32_t>(blockIdx.x);
    if (threadIdx.x == 0) {
        destination = resolveSegmentedToken(partial_token, partial_q_indptr, destination_starts, num_partial_segments);
    }
    __syncthreads();

    using OutputVector  = AlignedVector<OutputT, kValuesPerLane>;
    using PartialVector = AlignedVector<__nv_bfloat16, kValuesPerLane>;

    const int32_t warp_id = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    const int32_t lane_id = static_cast<int32_t>(threadIdx.x) % kWarpSize;
    const int32_t head    = static_cast<int32_t>(blockIdx.y) * kHeadsPerBlock + warp_id;
    if (head >= num_heads) {
        return;
    }

    const int64_t output_lse_offset = static_cast<int64_t>(destination) * output_lse_stride_token
                                      + static_cast<int64_t>(head) * output_lse_stride_head;
    const int64_t partial_lse_offset = static_cast<int64_t>(partial_token) * partial_lse_stride_token
                                       + static_cast<int64_t>(head) * partial_lse_stride_head;
    const float accumulator_lse = output_lse[output_lse_offset];
    const float state_lse       = partial_lse[partial_lse_offset];

    const int64_t token_stride = static_cast<int64_t>(num_heads) * kHeadSize;
    const int32_t value        = lane_id * kValuesPerLane;
    const int64_t output_offset =
        static_cast<int64_t>(destination) * token_stride + static_cast<int64_t>(head) * kHeadSize + value;
    const int64_t partial_offset =
        static_cast<int64_t>(partial_token) * token_stride + static_cast<int64_t>(head) * kHeadSize + value;

    // NaN poisons both results. With +inf, normalized scales are undefined,
    // while logaddexp remains +inf unless the other LSE is NaN.
    const bool has_nan               = isnan(accumulator_lse) || isnan(state_lse);
    const bool has_positive_infinity = accumulator_lse == CUDART_INF_F || state_lse == CUDART_INF_F;
    if (has_nan || has_positive_infinity) {
        OutputVector result;
#pragma unroll
        for (int index = 0; index < kValuesPerLane; ++index) {
            storeFromFloat(&result.values[index], CUDART_NAN_F);
        }
        storeVector(output + output_offset, result);
        if (lane_id == 0) {
            output_lse[output_lse_offset] = has_nan ? CUDART_NAN_F : CUDART_INF_F;
        }
        return;
    }

    const bool accumulator_empty = accumulator_lse == -CUDART_INF_F;
    const bool state_empty       = state_lse == -CUDART_INF_F;
    if (accumulator_empty || state_empty) {
        if (accumulator_empty) {
            OutputVector result;
            if (state_empty) {
#pragma unroll
                for (int index = 0; index < kValuesPerLane; ++index) {
                    storeFromFloat(&result.values[index], 0.0f);
                }
            } else {
                const PartialVector state = loadVector<__nv_bfloat16, kValuesPerLane>(partial_output + partial_offset);
#pragma unroll
                for (int index = 0; index < kValuesPerLane; ++index) {
                    storeFromFloat(&result.values[index], loadAsFloat(&state.values[index]));
                }
            }
            storeVector(output + output_offset, result);
        }
        if (lane_id == 0) {
            output_lse[output_lse_offset] = accumulator_empty ? state_lse : accumulator_lse;
        }
        return;
    }

    const float         max_lse           = fmaxf(accumulator_lse, state_lse);
    const float         accumulator_exp   = __expf(accumulator_lse - max_lse);
    const float         state_exp         = __expf(state_lse - max_lse);
    const float         denominator       = accumulator_exp + state_exp;
    const float         accumulator_scale = accumulator_exp / denominator;
    const float         state_scale       = state_exp / denominator;
    const OutputVector  accumulator       = loadVector<OutputT, kValuesPerLane>(output + output_offset);
    const PartialVector state             = loadVector<__nv_bfloat16, kValuesPerLane>(partial_output + partial_offset);
    OutputVector        result;
#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
        const float merged = loadAsFloat(&accumulator.values[index]) * accumulator_scale
                             + loadAsFloat(&state.values[index]) * state_scale;
        storeFromFloat(&result.values[index], merged);
    }
    storeVector(output + output_offset, result);
    if (lane_id == 0) {
        output_lse[output_lse_offset] = __logf(denominator) + max_lse;
    }
}

template<typename OutputT>
void launchSegmented(const at::Tensor& output,
                     const at::Tensor& output_lse,
                     const at::Tensor& partial_output,
                     const at::Tensor& partial_lse,
                     const at::Tensor& partial_q_indptr,
                     const at::Tensor& destination_starts,
                     cudaStream_t      stream) {
    const dim3 grid(static_cast<uint32_t>(partial_output.size(0)),
                    static_cast<uint32_t>((output.size(1) + kHeadsPerBlock - 1) / kHeadsPerBlock));
    flashMlaStateMergeSegmented128Kernel<OutputT>
        <<<grid, kThreads, 0, stream>>>(reinterpret_cast<OutputT*>(output.data_ptr()),
                                        output_lse.data_ptr<float>(),
                                        reinterpret_cast<const __nv_bfloat16*>(partial_output.data_ptr()),
                                        partial_lse.data_ptr<float>(),
                                        partial_q_indptr.data_ptr<int32_t>(),
                                        destination_starts.data_ptr<int32_t>(),
                                        static_cast<int32_t>(destination_starts.numel()),
                                        static_cast<int32_t>(output.size(1)),
                                        output_lse.stride(0),
                                        output_lse.stride(1),
                                        partial_lse.stride(0),
                                        partial_lse.stride(1));
}

}  // namespace

void invokeFlashMlaStateMergeSegmented(const at::Tensor& output,
                                       const at::Tensor& output_lse,
                                       const at::Tensor& partial_output,
                                       const at::Tensor& partial_lse,
                                       const at::Tensor& partial_q_indptr,
                                       const at::Tensor& destination_starts,
                                       cudaStream_t      stream) {
    if (output.scalar_type() == at::kFloat) {
        launchSegmented<float>(
            output, output_lse, partial_output, partial_lse, partial_q_indptr, destination_starts, stream);
    } else {
        launchSegmented<__nv_bfloat16>(
            output, output_lse, partial_output, partial_lse, partial_q_indptr, destination_starts, stream);
    }
}

}  // namespace rtp_llm
