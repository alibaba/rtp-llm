#include "rtp_llm/models_py/bindings/cuda/kernels/contiguous_silu_fp8.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rtp_llm {
namespace {

template<bool FAST_RECIPROCAL>
__device__ __forceinline__ float silu_rtp(const float gate) {
    float exp_neg, reciprocal;
    const float scaled = -gate * 0x1.715476p+0f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(exp_neg) : "f"(scaled));
    const float denom = 1.f + exp_neg;
    if constexpr (FAST_RECIPROCAL) {
        // H20 matches the reference over finite BF16 gates above this cutoff.
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(denom));
        if (gate < -80.f) {
            asm("div.full.f32 %0, 1.0, %1;" : "=f"(reciprocal) : "f"(denom));
        }
    } else {
        asm("div.full.f32 %0, 1.0, %1;" : "=f"(reciprocal) : "f"(denom));
    }
    return gate * reciprocal;
}

template<int N>
struct alignas(16) PackedBF16 { uint4 v[N]; };

template<int N>
struct alignas(N >= 8 ? 16 : 8) PackedFP8 { uint32_t v[N / 2]; };

__device__ __forceinline__ uint32_t pair_bits(const __nv_bfloat162 x) {
    return *reinterpret_cast<const uint32_t*>(&x);
}

__device__ __forceinline__ __nv_bfloat162 bits_pair(const uint32_t x) {
    return *reinterpret_cast<const __nv_bfloat162*>(&x);
}

template<int GROUPS, int LANES, int THREADS, bool FLAT, bool HAS_EXPERTS, bool FAST_SCALE, bool FAST_RECIPROCAL, int MIN_BLOCKS, bool TILE_PADDING_CHECK>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS) void silu_groups(
    const __nv_bfloat16* __restrict__ input, const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ expert_end, uint8_t* __restrict__ output,
    float* __restrict__ scales, int32_t rows, int32_t scale_ld, int32_t groups_runtime, int32_t pad_first, int32_t pad_last) {
    constexpr int values_per_lane = 128 / LANES;
    constexpr int pairs = values_per_lane / 2;
    const int lane = threadIdx.x % LANES;
    const int warp_lane = threadIdx.x % 32;
    const int groups = GROUPS ? GROUPS : groups_runtime;
    const int width = 128 * groups;
    const unsigned task = blockIdx.x * (THREADS / LANES) + threadIdx.x / LANES;
    const int row = FLAT ? task / static_cast<unsigned>(groups) : task;
    const int group = FLAT ? task % static_cast<unsigned>(groups) : blockIdx.y;
    if (row >= rows) {
        return;
    }
    bool live = true;
    if constexpr (HAS_EXPERTS) {
        bool check_padding;
        if constexpr (TILE_PADDING_CHECK) {
            const unsigned first_task = blockIdx.x * (THREADS / LANES);
            const unsigned last_task = first_task + THREADS / LANES - 1;
            check_padding = first_task < static_cast<unsigned>(pad_last * groups) &&
                last_task >= static_cast<unsigned>(pad_first * groups);
        } else {
            check_padding = row >= pad_first && row < pad_last;
        }
        if (check_padding) {
            live = row < expert_end[expert_ids[row]];
        }
    }
    const int column = 128 * group + lane * values_per_lane;
    PackedFP8<pairs> quant{};
    float scale = __fdiv_rn(1.e-4f, 448.f);
    if (live) {
        const auto* in = input + static_cast<int64_t>(row) * (2 * width) + column;
        const auto up = *reinterpret_cast<const PackedBF16<values_per_lane / 8>*>(in);
        const auto gate = *reinterpret_cast<const PackedBF16<values_per_lane / 8>*>(in + width);
        __nv_bfloat162 activations[pairs];
        auto maximum = __float2bfloat162_rn(0.f);
        #pragma unroll
        for (int i = 0; i < pairs; ++i) {
            const float2 u = __bfloat1622float2(bits_pair(reinterpret_cast<const uint32_t*>(&up)[i]));
            const float2 g = __bfloat1622float2(bits_pair(reinterpret_cast<const uint32_t*>(&gate)[i]));
            activations[i] = __floats2bfloat162_rn(silu_rtp<FAST_RECIPROCAL>(g.x) * u.x, silu_rtp<FAST_RECIPROCAL>(g.y) * u.y);
            maximum = __hmax2(maximum, __habs2(activations[i]));
        }
        // All lanes in a subwarp handle exactly one row/group and share the live predicate.
        const uint32_t mask = ((1u << LANES) - 1) << (warp_lane / LANES * LANES);
        #pragma unroll
        for (int offset = LANES / 2; offset; offset /= 2) {
            maximum = __hmax2(maximum, bits_pair(__shfl_xor_sync(mask, pair_bits(maximum), offset, LANES)));
        }
        const auto max_pair = __bfloat1622float2(maximum);
        const float amax = fmaxf(fmaxf(max_pair.x, max_pair.y), 1.e-4f);
        float inverse;
        if constexpr (FAST_SCALE) {
            // Split the reciprocal so rounding matches division on the BF16 absmax domain.
            constexpr float reciprocal = 1.f / 448.f;
            constexpr float residual = static_cast<float>(1.0 / 448.0 - reciprocal);
            scale = isinf(amax) ? amax : __fmaf_rn(amax, reciprocal, amax * residual);
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(scale));
        } else {
            scale = __fdiv_rn(amax, 448.f);
            inverse = __fdiv_rn(1.f, scale);
        }
        const auto inv = __float2bfloat162_rn(inverse);
        #pragma unroll
        for (int i = 0; i < pairs / 2; ++i) {
            const auto a = __hmul2(activations[2 * i], inv);
            const auto b = __hmul2(activations[2 * i + 1], inv);
            const auto lo = __nv_cvt_float2_to_fp8x2(__bfloat1622float2(a), __NV_SATFINITE, __NV_E4M3);
            const auto hi = __nv_cvt_float2_to_fp8x2(__bfloat1622float2(b), __NV_SATFINITE, __NV_E4M3);
            quant.v[i] = static_cast<uint32_t>(lo) | static_cast<uint32_t>(hi) << 16;
        }
    }
    auto* out = output + static_cast<int64_t>(row) * width + column;
    *reinterpret_cast<PackedFP8<pairs>*>(out) = quant;
    if (lane == 0) {
        scales[static_cast<int64_t>(group) * scale_ld + row] = scale;
    }
}

template<int GROUPS, int LANES, int THREADS, bool FLAT, bool HAS_EXPERTS, bool FAST_SCALE, bool FAST_RECIPROCAL, int MIN_BLOCKS, bool TILE_PADDING_CHECK>
void launch(const void* input, const int32_t* ids, const int32_t* end, uint8_t* output,
            float* scales, int rows, int scale_ld, int groups, cudaStream_t stream, int pad_first, int pad_last) {
    const int64_t tasks = FLAT ? static_cast<int64_t>(rows) * groups : rows;
    const dim3 grid((tasks + THREADS / LANES - 1) / (THREADS / LANES), FLAT ? 1 : groups);
    silu_groups<GROUPS, LANES, THREADS, FLAT, HAS_EXPERTS, FAST_SCALE, FAST_RECIPROCAL, MIN_BLOCKS, TILE_PADDING_CHECK><<<grid, THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), ids, end, output, scales, rows, scale_ld, groups, pad_first, pad_last);
}

template<int GROUPS, bool HAS_EXPERTS>
void dispatch(const void* input, const int32_t* ids, const int32_t* end, uint8_t* output,
              float* scales, int rows, int scale_ld, int groups, cudaStream_t stream, int pad_first, int pad_last) {
    if (rows <= 8192) {
        launch<GROUPS, 8, 128, false, HAS_EXPERTS, false, false, 1, false>(
            input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
    } else if (GROUPS == 5 && rows >= 65536) {
        launch<GROUPS, 4, 160, true, HAS_EXPERTS, false, true, 8, true>(
            input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
    } else {
        launch<GROUPS, 4, 128, true, HAS_EXPERTS, true, false, 1, false>(
            input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
    }
}

}  // namespace

cudaError_t invokeContiguousSiluFp8(
    const void* input, const int32_t* ids, const int32_t* end, uint8_t* output,
    float* scales, int rows, int width, int scale_ld, cudaStream_t stream, int pad_first, int pad_last) {
    const int groups = width / 128;
    if (!rows) return cudaSuccess;
    if (groups == 5) {
        if (ids) dispatch<5, true>(input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
        else dispatch<5, false>(input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
    } else {
        if (ids) dispatch<0, true>(input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
        else dispatch<0, false>(input, ids, end, output, scales, rows, scale_ld, groups, stream, pad_first, pad_last);
    }
    return cudaPeekAtLastError();
}

}  // namespace rtp_llm
