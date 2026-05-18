#include "dsv4_indexed_inv_rope_fp8_quant.h"

#include "fp8_ue8m0_scale_layout.cuh"
#include "util.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rtp_llm {
namespace {

constexpr int kQuantGroupSize = 128;
constexpr int kWarpSize       = 32;
constexpr int kWarpsPerCta    = 8;
constexpr int kBlockSize      = kWarpSize * kWarpsPerCta;
constexpr int kKernelModeGeneric     = 0;
constexpr int kKernelModeD512Token4  = 1;
constexpr int kKernelModeD512Group8  = 2;
constexpr int kKernelModeD512Token64 = 3;
constexpr int kKernelModeD512Tile16  = 4;
constexpr int kKernelModeD512Tile32  = 5;
constexpr int kKernelModeD512Token64Stream = 6;
constexpr int kKernelModeD512Head1Small = 7;
constexpr int kKernelModeD512Token64W16 = 8;
constexpr int kKernelModeD512Token64W32 = 9;

__device__ __forceinline__ float bf16ToFloat(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ uint8_t scaleToUe8m0Byte(float scale) {
    scale = fmaxf(scale, 1e-10f);
    const uint32_t bits = __float_as_uint(scale);
    int exp = static_cast<int>((bits >> 23) & 0xff);
    exp += (bits & 0x7fffff) != 0;
    exp = max(1, min(254, exp));
    return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float clampFp8(float v, float fp8_max) {
    return fminf(fmaxf(v, -fp8_max), fp8_max);
}

__device__ __forceinline__ uint32_t packFp8x4(float v0, float v1, float v2, float v3, float scale, float fp8_max) {
    uint32_t packed_u32 = 0;
    auto* packed = reinterpret_cast<__nv_fp8x2_storage_t*>(&packed_u32);
    const float2 q01 = make_float2(clampFp8(v0 / scale, fp8_max), clampFp8(v1 / scale, fp8_max));
    const float2 q23 = make_float2(clampFp8(v2 / scale, fp8_max), clampFp8(v3 / scale, fp8_max));
    packed[0] = __nv_cvt_float2_to_fp8x2(q01, __NV_SATFINITE, __NV_E4M3);
    packed[1] = __nv_cvt_float2_to_fp8x2(q23, __NV_SATFINITE, __NV_E4M3);
    return packed_u32;
}

template<int CHUNK>
__device__ __forceinline__ uint8_t processD512NopeChunkStream(const __nv_bfloat16* __restrict__ row_in,
                                                              __nv_fp8_e4m3* __restrict__ row_out,
                                                              int lane,
                                                              float eps,
                                                              float fp8_max) {
    static_assert(CHUNK >= 0 && CHUNK < 3, "pure nope stream chunk must be 0, 1, or 2");
    constexpr int kColBase = CHUNK * kQuantGroupSize;
    const int col = kColBase + lane * 4;
    const auto* row_in2 = reinterpret_cast<const __nv_bfloat162*>(row_in);
    const int pair_base = col >> 1;
    const float2 x01 = __bfloat1622float2(row_in2[pair_base]);
    const float2 x23 = __bfloat1622float2(row_in2[pair_base + 1]);
    const float local_absmax = fmaxf(eps, fmaxf(fmaxf(fabsf(x01.x), fabsf(x01.y)), fmaxf(fabsf(x23.x), fabsf(x23.y))));
    const float absmax = __shfl_sync(0xffffffff, warpReduceMax(local_absmax), 0);
    const uint8_t scale_byte = scaleToUe8m0Byte(absmax / fp8_max);
    const float scale = __uint_as_float(static_cast<uint32_t>(scale_byte) << 23);
    const uint32_t packed = packFp8x4(x01.x, x01.y, x23.x, x23.y, scale, fp8_max);
    *reinterpret_cast<uint32_t*>(row_out + col) = packed;
    return scale_byte;
}

__device__ __forceinline__ uint8_t processD512Chunk3Stream(const __nv_bfloat16* __restrict__ row_in,
                                                           const float2* __restrict__ shared_freq,
                                                           __nv_fp8_e4m3* __restrict__ row_out,
                                                           int lane,
                                                           float eps,
                                                           float fp8_max) {
    constexpr int kChunk       = 3;
    constexpr int kColBase     = kChunk * kQuantGroupSize;
    constexpr int kRopeOffset  = 448;
    constexpr int kRopePairStart = kRopeOffset / 2;
    const int col = kColBase + lane * 4;
    const auto* row_in2 = reinterpret_cast<const __nv_bfloat162*>(row_in);
    const int pair_base = col >> 1;
    const float2 x01 = __bfloat1622float2(row_in2[pair_base]);
    const float2 x23 = __bfloat1622float2(row_in2[pair_base + 1]);
    float v0 = x01.x;
    float v1 = x01.y;
    float v2 = x23.x;
    float v3 = x23.y;

    if (lane >= 16) {
        const int rope_pair0 = pair_base - kRopePairStart;
        const float2 freq0 = shared_freq[rope_pair0];
        const float2 freq1 = shared_freq[rope_pair0 + 1];
        v0 = x01.x * freq0.x + x01.y * freq0.y;
        v1 = x01.y * freq0.x - x01.x * freq0.y;
        v2 = x23.x * freq1.x + x23.y * freq1.y;
        v3 = x23.y * freq1.x - x23.x * freq1.y;
    }

    const float local_absmax = fmaxf(eps, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
    const float absmax = __shfl_sync(0xffffffff, warpReduceMax(local_absmax), 0);
    const uint8_t scale_byte = scaleToUe8m0Byte(absmax / fp8_max);
    const float scale = __uint_as_float(static_cast<uint32_t>(scale_byte) << 23);
    const uint32_t packed = packFp8x4(v0, v1, v2, v3, scale, fp8_max);
    *reinterpret_cast<uint32_t*>(row_out + col) = packed;
    return scale_byte;
}

__device__ __forceinline__ void processD512HeadStream(const __nv_bfloat16* __restrict__ input,
                                                       const float2* __restrict__ shared_freq,
                                                       __nv_fp8_e4m3* __restrict__ output_q,
                                                       int32_t* __restrict__ output_s,
                                                       int64_t token,
                                                       int gh,
                                                       int group,
                                                       int head_in_group,
                                                       int64_t input_stride_m,
                                                       int64_t input_stride_h,
                                                       int64_t output_q_stride_m,
                                                       int64_t output_q_stride_g,
                                                       int64_t output_s_stride_m,
                                                       int64_t output_s_stride_g,
                                                       int64_t output_s_stride_k,
                                                       float eps,
                                                       float fp8_max) {
    constexpr int kHeadDim = 512;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* row_in = input + token * input_stride_m + static_cast<int64_t>(gh) * input_stride_h;
    __nv_fp8_e4m3* row_out = output_q + token * output_q_stride_m + group * output_q_stride_g
                             + static_cast<int64_t>(head_in_group) * kHeadDim;

    uint32_t packed_scale = 0;
    packed_scale |= static_cast<uint32_t>(processD512NopeChunkStream<0>(row_in, row_out, lane, eps, fp8_max));
    packed_scale |= static_cast<uint32_t>(processD512NopeChunkStream<1>(row_in, row_out, lane, eps, fp8_max)) << 8;
    packed_scale |= static_cast<uint32_t>(processD512NopeChunkStream<2>(row_in, row_out, lane, eps, fp8_max)) << 16;
    packed_scale |= static_cast<uint32_t>(processD512Chunk3Stream(row_in, shared_freq, row_out, lane, eps, fp8_max))
                    << 24;

    if (lane == 0) {
        output_s[token * output_s_stride_m + group * output_s_stride_g + head_in_group * output_s_stride_k] =
            static_cast<int32_t>(packed_scale);
    }
}

template<bool USE_SHARED_FREQ, typename index_t>
__device__ __forceinline__ void processD512HeadVec(const __nv_bfloat16* __restrict__ input,
                                                    const float2* __restrict__ freqs_cis,
                                                    const index_t* __restrict__ position_ids,
                                                    const float2* __restrict__ shared_freq,
                                                    __nv_fp8_e4m3* __restrict__ output_q,
                                                    int32_t* __restrict__ output_s,
                                                    int64_t token,
                                                    int gh,
                                                    int group,
                                                    int head_in_group,
                                                    int64_t input_stride_m,
                                                    int64_t input_stride_h,
                                                    int64_t freqs_stride,
                                                    int64_t output_q_stride_m,
                                                    int64_t output_q_stride_g,
                                                    int64_t output_s_stride_m,
                                                    int64_t output_s_stride_g,
                                                    int64_t output_s_stride_k,
                                                    float eps,
                                                    float fp8_max) {
    constexpr int kHeadDim       = 512;
    constexpr int kChunks        = 4;
    constexpr int kRopeOffset    = 448;
    constexpr int kColsPerLane   = 4;
    constexpr int kRopePairStart = kRopeOffset / 2;

    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* row_in = input + token * input_stride_m + static_cast<int64_t>(gh) * input_stride_h;
    __nv_fp8_e4m3* row_out = output_q + token * output_q_stride_m + group * output_q_stride_g
                             + static_cast<int64_t>(head_in_group) * kHeadDim;

    const auto* row_in2 = reinterpret_cast<const __nv_bfloat162*>(row_in);

    float vals[kChunks][kColsPerLane];
    float local_absmax[kChunks] = {eps, eps, eps, eps};
#pragma unroll
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        const int col_base = chunk * kQuantGroupSize + lane * kColsPerLane;
        const int pair_base = col_base >> 1;
        const float2 x01 = __bfloat1622float2(row_in2[pair_base]);
        const float2 x23 = __bfloat1622float2(row_in2[pair_base + 1]);
        float v0 = x01.x;
        float v1 = x01.y;
        float v2 = x23.x;
        float v3 = x23.y;
        if (chunk == 3 && col_base >= kRopeOffset) {
            const int rope_pair0 = pair_base - kRopePairStart;
            const int rope_pair1 = rope_pair0 + 1;
            float2 freq0;
            float2 freq1;
            if constexpr (USE_SHARED_FREQ) {
                freq0 = shared_freq[rope_pair0];
                freq1 = shared_freq[rope_pair1];
            } else {
                const int64_t pos = static_cast<int64_t>(position_ids[token]);
                freq0 = freqs_cis[pos * freqs_stride + rope_pair0];
                freq1 = freqs_cis[pos * freqs_stride + rope_pair1];
            }
            const float real0 = v0;
            const float imag0 = v1;
            const float real1 = v2;
            const float imag1 = v3;
            v0 = real0 * freq0.x + imag0 * freq0.y;
            v1 = imag0 * freq0.x - real0 * freq0.y;
            v2 = real1 * freq1.x + imag1 * freq1.y;
            v3 = imag1 * freq1.x - real1 * freq1.y;
        }
        vals[chunk][0] = v0;
        vals[chunk][1] = v1;
        vals[chunk][2] = v2;
        vals[chunk][3] = v3;
        local_absmax[chunk] = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    }

    uint32_t packed_scale = 0;
    float scales[kChunks];
#pragma unroll
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        const float absmax = __shfl_sync(0xffffffff, warpReduceMax(local_absmax[chunk]), 0);
        const uint8_t scale_byte = scaleToUe8m0Byte(absmax / fp8_max);
        scales[chunk] = __uint_as_float(static_cast<uint32_t>(scale_byte) << 23);
        if (lane == 0) {
            packed_scale |= static_cast<uint32_t>(scale_byte) << (chunk * 8);
        }
    }

#pragma unroll
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        const int col_base = chunk * kQuantGroupSize + lane * kColsPerLane;
        const uint32_t packed = packFp8x4(
            vals[chunk][0], vals[chunk][1], vals[chunk][2], vals[chunk][3], scales[chunk], fp8_max);
        *reinterpret_cast<uint32_t*>(row_out + col_base) = packed;
    }

    if (lane == 0) {
        output_s[token * output_s_stride_m + group * output_s_stride_g + head_in_group * output_s_stride_k] =
            static_cast<int32_t>(packed_scale);
    }
}

template<typename index_t, int TOKENS_PER_CTA>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_inv_rope_fp8_quant_d512_token4_vec_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float eps,
    float fp8_max) {
    const int warp_id = threadIdx.x >> 5;
    const int group = static_cast<int>(blockIdx.y);
    const int head_in_group = warp_id;
    const int gh = group * kWarpsPerCta + head_in_group;
    const int64_t token_base = static_cast<int64_t>(blockIdx.x) * TOKENS_PER_CTA;
#pragma unroll
    for (int token_iter = 0; token_iter < TOKENS_PER_CTA; ++token_iter) {
        const int64_t token = token_base + token_iter;
        if (token >= m) {
            return;
        }
        processD512HeadVec<false>(input,
                                  freqs_cis,
                                  position_ids,
                                  nullptr,
                                  output_q,
                                  output_s,
                                  token,
                                  gh,
                                  group,
                                  head_in_group,
                                  input_stride_m,
                                  input_stride_h,
                                  freqs_stride,
                                  output_q_stride_m,
                                  output_q_stride_g,
                                  output_s_stride_m,
                                  output_s_stride_g,
                                  output_s_stride_k,
                                  eps,
                                  fp8_max);
    }
}

template<typename index_t>
__global__ __launch_bounds__(kWarpSize) void dsv4_indexed_inv_rope_fp8_quant_d512_head1_small_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float eps,
    float fp8_max) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= m) {
        return;
    }
    const int gh = static_cast<int>(blockIdx.y);
    const int group = gh >> 3;
    const int head_in_group = gh & 7;

    processD512HeadVec<false>(input,
                              freqs_cis,
                              position_ids,
                              nullptr,
                              output_q,
                              output_s,
                              token,
                              gh,
                              group,
                              head_in_group,
                              input_stride_m,
                              input_stride_h,
                              freqs_stride,
                              output_q_stride_m,
                              output_q_stride_g,
                              output_s_stride_m,
                              output_s_stride_g,
                              output_s_stride_k,
                              eps,
                              fp8_max);
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_inv_rope_fp8_quant_d512_group8_vec_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float eps,
    float fp8_max) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= m) {
        return;
    }
    const int group = static_cast<int>(blockIdx.y);
    const int warp_id = threadIdx.x >> 5;
    const int head_in_group = warp_id;
    const int gh = group * kWarpsPerCta + head_in_group;

    __shared__ float2 shared_freq[32];
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    for (int pair = threadIdx.x; pair < 32; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

    processD512HeadVec<true>(input,
                             freqs_cis,
                             position_ids,
                             shared_freq,
                             output_q,
                             output_s,
                             token,
                             gh,
                             group,
                             head_in_group,
                             input_stride_m,
                             input_stride_h,
                             freqs_stride,
                             output_q_stride_m,
                             output_q_stride_g,
                             output_s_stride_m,
                             output_s_stride_g,
                             output_s_stride_k,
                             eps,
                             fp8_max);
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_inv_rope_fp8_quant_d512_headtile_vec_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float eps,
    float fp8_max,
    int   heads_per_tile) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= m) {
        return;
    }
    const int warp_id = threadIdx.x >> 5;
    const int head_base = static_cast<int>(blockIdx.y) * heads_per_tile;

    __shared__ float2 shared_freq[32];
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    for (int pair = threadIdx.x; pair < 32; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

    for (int local_head = warp_id; local_head < heads_per_tile; local_head += kWarpsPerCta) {
        const int gh = head_base + local_head;
        const int group = gh / kWarpsPerCta;
        const int head_in_group = gh - group * kWarpsPerCta;
        processD512HeadVec<true>(input,
                                 freqs_cis,
                                 position_ids,
                                 shared_freq,
                                 output_q,
                                 output_s,
                                 token,
                                 gh,
                                 group,
                                 head_in_group,
                                 input_stride_m,
                                 input_stride_h,
                                 freqs_stride,
                                 output_q_stride_m,
                                 output_q_stride_g,
                                 output_s_stride_m,
                                 output_s_stride_g,
                                 output_s_stride_k,
                                 eps,
                                 fp8_max);
    }
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_inv_rope_fp8_quant_d512_token64_vec_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float eps,
    float fp8_max) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= m) {
        return;
    }
    const int warp_id = threadIdx.x >> 5;

    __shared__ float2 shared_freq[32];
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    for (int pair = threadIdx.x; pair < 32; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

    for (int gh = warp_id; gh < 64; gh += kWarpsPerCta) {
        const int group = gh / kWarpsPerCta;
        const int head_in_group = gh - group * kWarpsPerCta;
        processD512HeadVec<true>(input,
                                 freqs_cis,
                                 position_ids,
                                 shared_freq,
                                 output_q,
                                 output_s,
                                 token,
                                 gh,
                                 group,
                                 head_in_group,
                                 input_stride_m,
                                 input_stride_h,
                                 freqs_stride,
                                 output_q_stride_m,
                                 output_q_stride_g,
                                 output_s_stride_m,
                                 output_s_stride_g,
                                 output_s_stride_k,
                                 eps,
                                 fp8_max);
    }
}

template<typename index_t, int WARPS_PER_CTA>
__global__ __launch_bounds__(WARPS_PER_CTA * kWarpSize)
    void dsv4_indexed_inv_rope_fp8_quant_d512_token64_warpvec_kernel(
        const __nv_bfloat16* __restrict__ input,
        const float2* __restrict__ freqs_cis,
        const index_t* __restrict__ position_ids,
        __nv_fp8_e4m3* __restrict__ output_q,
        int32_t* __restrict__ output_s,
        int64_t m,
        int64_t input_stride_m,
        int64_t input_stride_h,
        int64_t freqs_stride,
        int64_t output_q_stride_m,
        int64_t output_q_stride_g,
        int64_t output_s_stride_m,
        int64_t output_s_stride_g,
        int64_t output_s_stride_k,
        float eps,
        float fp8_max) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= m) {
        return;
    }
    const int warp_id = threadIdx.x >> 5;

    __shared__ float2 shared_freq[32];
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    for (int pair = threadIdx.x; pair < 32; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

    for (int gh = warp_id; gh < 64; gh += WARPS_PER_CTA) {
        const int group = gh >> 3;
        const int head_in_group = gh & 7;
        processD512HeadVec<true>(input,
                                 freqs_cis,
                                 position_ids,
                                 shared_freq,
                                 output_q,
                                 output_s,
                                 token,
                                 gh,
                                 group,
                                 head_in_group,
                                 input_stride_m,
                                 input_stride_h,
                                 freqs_stride,
                                 output_q_stride_m,
                                 output_q_stride_g,
                                 output_s_stride_m,
                                 output_s_stride_g,
                                 output_s_stride_k,
                                 eps,
                                 fp8_max);
    }
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_inv_rope_fp8_quant_d512_token64_stream_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float eps,
    float fp8_max) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= m) {
        return;
    }
    const int warp_id = threadIdx.x >> 5;

    __shared__ float2 shared_freq[32];
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    for (int pair = threadIdx.x; pair < 32; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

#pragma unroll
    for (int head_iter = 0; head_iter < 8; ++head_iter) {
        const int gh = warp_id + head_iter * kWarpsPerCta;
        const int group = gh >> 3;
        const int head_in_group = gh & 7;
        processD512HeadStream(input,
                              shared_freq,
                              output_q,
                              output_s,
                              token,
                              gh,
                              group,
                              head_in_group,
                              input_stride_m,
                              input_stride_h,
                              output_q_stride_m,
                              output_q_stride_g,
                              output_s_stride_m,
                              output_s_stride_g,
                              output_s_stride_k,
                              eps,
                              fp8_max);
    }
}

template<typename index_t, int CHUNKS_PER_HEAD, int ROPE_HEAD_DIM, int WARPS_PER_CTA>
__global__ __launch_bounds__(WARPS_PER_CTA * kWarpSize) void dsv4_indexed_inv_rope_fp8_quant_warp_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int     h,
    int     n_groups,
    int     heads_per_group,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float   eps,
    float   fp8_max) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int     gh    = static_cast<int>(blockIdx.y) * WARPS_PER_CTA + warp_id;
    if (token >= m) {
        return;
    }

    constexpr int kHeadDim = CHUNKS_PER_HEAD * kQuantGroupSize;
    constexpr int kRopeOffset = kHeadDim - ROPE_HEAD_DIM;
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    __shared__ float2 shared_freq[ROPE_HEAD_DIM / 2];
    for (int pair = threadIdx.x; pair < ROPE_HEAD_DIM / 2; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

    if (gh >= h) {
        return;
    }

    const int group         = gh / heads_per_group;
    const int head_in_group = gh - group * heads_per_group;
    const __nv_bfloat16* row_in = input + token * input_stride_m + gh * input_stride_h;
    __nv_fp8_e4m3* row_out = output_q + token * output_q_stride_m + group * output_q_stride_g
                             + static_cast<int64_t>(head_in_group) * kHeadDim;

    float vals[16];
    float local_absmax[4] = {eps, eps, eps, eps};
    constexpr int kElemsPerLane = CHUNKS_PER_HEAD * (kQuantGroupSize / kWarpSize);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = 0.0f;
        if (i < kElemsPerLane) {
            const int col = lane + i * kWarpSize;
            float x = bf16ToFloat(row_in[col]);
            if (col >= kRopeOffset) {
                const int rope_local = col - kRopeOffset;
                const int pair = rope_local >> 1;
                const float partner = __shfl_xor_sync(0xffffffff, x, 1);
                const float2 freq = shared_freq[pair];
                if ((rope_local & 1) == 0) {
                    x = x * freq.x + partner * freq.y;
                } else {
                    x = x * freq.x - partner * freq.y;
                }
            }
            vals[i] = x;
            const int chunk = col / kQuantGroupSize;
            local_absmax[chunk] = fmaxf(local_absmax[chunk], fabsf(x));
        }
    }

    uint32_t packed_scale = 0;
    float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        if (chunk < CHUNKS_PER_HEAD) {
            const float absmax = __shfl_sync(0xffffffff, warpReduceMax(local_absmax[chunk]), 0);
            const uint8_t scale_byte = scaleToUe8m0Byte(absmax / fp8_max);
            scales[chunk] = __uint_as_float(static_cast<uint32_t>(scale_byte) << 23);
            if (lane == 0) {
                packed_scale |= static_cast<uint32_t>(scale_byte) << (chunk * 8);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i < kElemsPerLane) {
            const int col = lane + i * kWarpSize;
            const int chunk = col / kQuantGroupSize;
            const float q = fminf(fmaxf(vals[i] / scales[chunk], -fp8_max), fp8_max);
            row_out[col] = __nv_fp8_e4m3(q);
        }
    }

    if (lane == 0) {
        output_s[token * output_s_stride_m + group * output_s_stride_g + head_in_group * output_s_stride_k] =
            static_cast<int32_t>(packed_scale);
    }
}

template<typename index_t, int CHUNKS_PER_HEAD, int ROPE_HEAD_DIM, int WARPS_PER_CTA, int TOKENS_PER_CTA>
__global__ __launch_bounds__(WARPS_PER_CTA * kWarpSize) void dsv4_indexed_inv_rope_fp8_quant_no_shared_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t m,
    int     h,
    int     n_groups,
    int     heads_per_group,
    int64_t input_stride_m,
    int64_t input_stride_h,
    int64_t freqs_stride,
    int64_t output_q_stride_m,
    int64_t output_q_stride_g,
    int64_t output_s_stride_m,
    int64_t output_s_stride_g,
    int64_t output_s_stride_k,
    float   eps,
    float   fp8_max) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int     gh    = static_cast<int>(blockIdx.y) * WARPS_PER_CTA + warp_id;
    if (gh >= h) {
        return;
    }

    constexpr int kHeadDim = CHUNKS_PER_HEAD * kQuantGroupSize;
    constexpr int kRopeOffset = kHeadDim - ROPE_HEAD_DIM;
    const int group = gh / heads_per_group;
    const int head_in_group = gh - group * heads_per_group;
    constexpr int kElemsPerLane = CHUNKS_PER_HEAD * (kQuantGroupSize / kWarpSize);
    const int64_t token_base = static_cast<int64_t>(blockIdx.x) * TOKENS_PER_CTA;
#pragma unroll
    for (int token_iter = 0; token_iter < TOKENS_PER_CTA; ++token_iter) {
        const int64_t token = token_base + token_iter;
        if (token >= m) {
            return;
        }
        const int64_t pos = static_cast<int64_t>(position_ids[token]);
        const __nv_bfloat16* row_in = input + token * input_stride_m + gh * input_stride_h;
        __nv_fp8_e4m3* row_out = output_q + token * output_q_stride_m + group * output_q_stride_g
                                 + static_cast<int64_t>(head_in_group) * kHeadDim;

        float vals[16];
        float local_absmax[4] = {eps, eps, eps, eps};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = 0.0f;
            if (i < kElemsPerLane) {
                const int col = lane + i * kWarpSize;
                float x = bf16ToFloat(row_in[col]);
                if (col >= kRopeOffset) {
                    const int rope_local = col - kRopeOffset;
                    const int pair = rope_local >> 1;
                    const float partner = __shfl_xor_sync(0xffffffff, x, 1);
                    const float2 freq = freqs_cis[pos * freqs_stride + pair];
                    if ((rope_local & 1) == 0) {
                        x = x * freq.x + partner * freq.y;
                    } else {
                        x = x * freq.x - partner * freq.y;
                    }
                }
                vals[i] = x;
                const int chunk = col / kQuantGroupSize;
                local_absmax[chunk] = fmaxf(local_absmax[chunk], fabsf(x));
            }
        }

        uint32_t packed_scale = 0;
        float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            if (chunk < CHUNKS_PER_HEAD) {
                const float absmax = __shfl_sync(0xffffffff, warpReduceMax(local_absmax[chunk]), 0);
                const uint8_t scale_byte = scaleToUe8m0Byte(absmax / fp8_max);
                scales[chunk] = __uint_as_float(static_cast<uint32_t>(scale_byte) << 23);
                if (lane == 0) {
                    packed_scale |= static_cast<uint32_t>(scale_byte) << (chunk * 8);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < kElemsPerLane) {
                const int col = lane + i * kWarpSize;
                const int chunk = col / kQuantGroupSize;
                const float q = fminf(fmaxf(vals[i] / scales[chunk], -fp8_max), fp8_max);
                row_out[col] = __nv_fp8_e4m3(q);
            }
        }

        if (lane == 0) {
            output_s[token * output_s_stride_m + group * output_s_stride_g + head_in_group * output_s_stride_k] =
                static_cast<int32_t>(packed_scale);
        }
    }
}

void checkLaunchRange(int64_t value, const char* name) {
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
                name,
                " is too large for dsv4 indexed inverse-RoPE+FP8 quant launch: ",
                value);
}

template<typename index_t>
void launchD512Vec(torch::Tensor input,
                   torch::Tensor freqs_cis,
                   torch::Tensor position_ids,
                   torch::Tensor output_q,
                   torch::Tensor output_s,
                   int64_t       m,
                   double        eps,
                   double        fp8_max,
                   int64_t       kernel_mode) {
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int kTokensPerCta = 4;
    const auto input_stride_m = input.stride(input.dim() - 3);
    const auto input_stride_h = input.stride(input.dim() - 2);
    const auto freqs_stride = freqs_cis.stride(0);
    const auto output_q_stride_m = output_q.stride(0);
    const auto output_q_stride_g = output_q.stride(1);
    const auto output_s_stride_m = output_s.stride(0);
    const auto output_s_stride_g = output_s.stride(1);
    const auto output_s_stride_k = output_s.stride(2);

    if (kernel_mode == kKernelModeD512Head1Small) {
        const dim3 grid(static_cast<uint32_t>(m), 64);
        dsv4_indexed_inv_rope_fp8_quant_d512_head1_small_kernel<index_t><<<grid, kWarpSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            reinterpret_cast<int32_t*>(output_s.data_ptr()),
            m,
            input_stride_m,
            input_stride_h,
            freqs_stride,
            output_q_stride_m,
            output_q_stride_g,
            output_s_stride_m,
            output_s_stride_g,
            output_s_stride_k,
            static_cast<float>(eps),
            static_cast<float>(fp8_max));
        return;
    }

    if (kernel_mode == kKernelModeD512Token64 || kernel_mode == kKernelModeD512Token64Stream) {
        if (kernel_mode == kKernelModeD512Token64Stream) {
            dsv4_indexed_inv_rope_fp8_quant_d512_token64_stream_kernel<index_t>
                <<<static_cast<uint32_t>(m), kBlockSize, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
                    reinterpret_cast<const index_t*>(position_ids.data_ptr()),
                    reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                    reinterpret_cast<int32_t*>(output_s.data_ptr()),
                    m,
                    input_stride_m,
                    input_stride_h,
                    freqs_stride,
                    output_q_stride_m,
                    output_q_stride_g,
                    output_s_stride_m,
                    output_s_stride_g,
                    output_s_stride_k,
                    static_cast<float>(eps),
                    static_cast<float>(fp8_max));
            return;
        }
        dsv4_indexed_inv_rope_fp8_quant_d512_token64_vec_kernel<index_t>
            <<<static_cast<uint32_t>(m), kBlockSize, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
                reinterpret_cast<const index_t*>(position_ids.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                reinterpret_cast<int32_t*>(output_s.data_ptr()),
                m,
                input_stride_m,
                input_stride_h,
                freqs_stride,
                output_q_stride_m,
                output_q_stride_g,
                output_s_stride_m,
                output_s_stride_g,
                output_s_stride_k,
                static_cast<float>(eps),
                static_cast<float>(fp8_max));
        return;
    }

    if (kernel_mode == kKernelModeD512Token64W16 || kernel_mode == kKernelModeD512Token64W32) {
        if (kernel_mode == kKernelModeD512Token64W16) {
            constexpr int kWarpsPerCtaW16 = 16;
            dsv4_indexed_inv_rope_fp8_quant_d512_token64_warpvec_kernel<index_t, kWarpsPerCtaW16>
                <<<static_cast<uint32_t>(m), kWarpsPerCtaW16 * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
                    reinterpret_cast<const index_t*>(position_ids.data_ptr()),
                    reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                    reinterpret_cast<int32_t*>(output_s.data_ptr()),
                    m,
                    input_stride_m,
                    input_stride_h,
                    freqs_stride,
                    output_q_stride_m,
                    output_q_stride_g,
                    output_s_stride_m,
                    output_s_stride_g,
                    output_s_stride_k,
                    static_cast<float>(eps),
                    static_cast<float>(fp8_max));
            return;
        }
        constexpr int kWarpsPerCtaW32 = 32;
        dsv4_indexed_inv_rope_fp8_quant_d512_token64_warpvec_kernel<index_t, kWarpsPerCtaW32>
            <<<static_cast<uint32_t>(m), kWarpsPerCtaW32 * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
                reinterpret_cast<const index_t*>(position_ids.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                reinterpret_cast<int32_t*>(output_s.data_ptr()),
                m,
                input_stride_m,
                input_stride_h,
                freqs_stride,
                output_q_stride_m,
                output_q_stride_g,
                output_s_stride_m,
                output_s_stride_g,
                output_s_stride_k,
                static_cast<float>(eps),
                static_cast<float>(fp8_max));
        return;
    }

    if (kernel_mode == kKernelModeD512Token4) {
        const dim3 grid(static_cast<uint32_t>((m + kTokensPerCta - 1) / kTokensPerCta), 8);
        dsv4_indexed_inv_rope_fp8_quant_d512_token4_vec_kernel<index_t, kTokensPerCta>
            <<<grid, kBlockSize, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
                reinterpret_cast<const index_t*>(position_ids.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                reinterpret_cast<int32_t*>(output_s.data_ptr()),
                m,
                input_stride_m,
                input_stride_h,
                freqs_stride,
                output_q_stride_m,
                output_q_stride_g,
                output_s_stride_m,
                output_s_stride_g,
                output_s_stride_k,
                static_cast<float>(eps),
                static_cast<float>(fp8_max));
        return;
    }

    if (kernel_mode == kKernelModeD512Tile16 || kernel_mode == kKernelModeD512Tile32) {
        const int heads_per_tile = kernel_mode == kKernelModeD512Tile16 ? 16 : 32;
        const dim3 grid(static_cast<uint32_t>(m), static_cast<uint32_t>(64 / heads_per_tile));
        dsv4_indexed_inv_rope_fp8_quant_d512_headtile_vec_kernel<index_t><<<grid, kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            reinterpret_cast<int32_t*>(output_s.data_ptr()),
            m,
            input_stride_m,
            input_stride_h,
            freqs_stride,
            output_q_stride_m,
            output_q_stride_g,
            output_s_stride_m,
            output_s_stride_g,
            output_s_stride_k,
            static_cast<float>(eps),
            static_cast<float>(fp8_max),
            heads_per_tile);
        return;
    }

    TORCH_CHECK(kernel_mode == kKernelModeD512Group8,
                "invalid D512 inverse-RoPE+FP8 quant kernel_mode: ",
                kernel_mode);
    const dim3 grid(static_cast<uint32_t>(m), 8);
    dsv4_indexed_inv_rope_fp8_quant_d512_group8_vec_kernel<index_t><<<grid, kBlockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
        reinterpret_cast<const index_t*>(position_ids.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        reinterpret_cast<int32_t*>(output_s.data_ptr()),
        m,
        input_stride_m,
        input_stride_h,
        freqs_stride,
        output_q_stride_m,
        output_q_stride_g,
        output_s_stride_m,
        output_s_stride_g,
        output_s_stride_k,
        static_cast<float>(eps),
        static_cast<float>(fp8_max));
}

template<typename index_t, int CHUNKS_PER_HEAD, int ROPE_HEAD_DIM>
void launch(torch::Tensor input,
            torch::Tensor freqs_cis,
            torch::Tensor position_ids,
            torch::Tensor output_q,
            torch::Tensor output_s,
            int64_t       m,
            int64_t       h,
            int64_t       n_groups,
            int64_t       heads_per_group,
            double        eps,
            double        fp8_max) {
    auto stream = at::cuda::getCurrentCUDAStream();
    if (m < 512 && CHUNKS_PER_HEAD == 4 && ROPE_HEAD_DIM == 64) {
        constexpr int kTokensPerCta = 4;
        const dim3 grid(static_cast<uint32_t>((m + kTokensPerCta - 1) / kTokensPerCta),
                        static_cast<uint32_t>((h + kWarpsPerCta - 1) / kWarpsPerCta));
        dsv4_indexed_inv_rope_fp8_quant_no_shared_kernel<
            index_t,
            CHUNKS_PER_HEAD,
            ROPE_HEAD_DIM,
            kWarpsPerCta,
            kTokensPerCta>
            <<<grid, kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            reinterpret_cast<int32_t*>(output_s.data_ptr()),
            m,
            static_cast<int>(h),
            static_cast<int>(n_groups),
            static_cast<int>(heads_per_group),
            input.stride(input.dim() - 3),
            input.stride(input.dim() - 2),
            freqs_cis.stride(0),
            output_q.stride(0),
            output_q.stride(1),
            output_s.stride(0),
            output_s.stride(1),
            output_s.stride(2),
            static_cast<float>(eps),
            static_cast<float>(fp8_max));
        return;
    }
    const dim3 grid(static_cast<uint32_t>(m), static_cast<uint32_t>((h + kWarpsPerCta - 1) / kWarpsPerCta));
    dsv4_indexed_inv_rope_fp8_quant_warp_kernel<index_t, CHUNKS_PER_HEAD, ROPE_HEAD_DIM, kWarpsPerCta>
        <<<grid, kBlockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
        reinterpret_cast<const index_t*>(position_ids.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        reinterpret_cast<int32_t*>(output_s.data_ptr()),
        m,
        static_cast<int>(h),
        static_cast<int>(n_groups),
        static_cast<int>(heads_per_group),
        input.stride(input.dim() - 3),
        input.stride(input.dim() - 2),
        freqs_cis.stride(0),
        output_q.stride(0),
        output_q.stride(1),
        output_s.stride(0),
        output_s.stride(1),
        output_s.stride(2),
        static_cast<float>(eps),
        static_cast<float>(fp8_max));
}

template<typename index_t>
void dispatchLaunch(torch::Tensor input,
                    torch::Tensor freqs_cis,
                    torch::Tensor position_ids,
                    torch::Tensor output_q,
                    torch::Tensor output_s,
                    int64_t       m,
                    int64_t       h,
                    int64_t       d,
                    int64_t       n_groups,
                    int64_t       heads_per_group,
                    int64_t       rope_head_dim,
                    int64_t       chunks_per_head,
                    double        eps,
                    double        fp8_max,
                    int64_t       kernel_mode) {
    if (kernel_mode != kKernelModeGeneric) {
        TORCH_CHECK(h == 64 && d == 512 && n_groups == 8 && heads_per_group == 8 && chunks_per_head == 4
                        && rope_head_dim == 64,
                    "D512 optimized inverse-RoPE+FP8 quant kernel_mode requires H=64,G=8,heads_per_group=8,D=512,"
                    "rope_head_dim=64");
        launchD512Vec<index_t>(input, freqs_cis, position_ids, output_q, output_s, m, eps, fp8_max, kernel_mode);
        return;
    }
    if (chunks_per_head == 4 && rope_head_dim == 64) {
        launch<index_t, 4, 64>(
            input, freqs_cis, position_ids, output_q, output_s, m, h, n_groups, heads_per_group, eps, fp8_max);
    } else if (chunks_per_head == 1 && rope_head_dim == 64) {
        launch<index_t, 1, 64>(
            input, freqs_cis, position_ids, output_q, output_s, m, h, n_groups, heads_per_group, eps, fp8_max);
    } else {
        TORCH_CHECK(false,
                    "unsupported specialized inverse-RoPE+FP8 quant shape: D=",
                    d,
                    " rope_head_dim=",
                    rope_head_dim);
    }
}

}  // namespace

void dsv4_indexed_inv_rope_fp8_quant(torch::Tensor input,
                                     torch::Tensor freqs_cis,
                                     torch::Tensor position_ids,
                                     torch::Tensor output_q,
                                     torch::Tensor output_s,
                                     int64_t       n_groups,
                                     int64_t       heads_per_group,
                                     int64_t       nope_dim,
                                     int64_t       rope_head_dim,
                                     double        eps,
                                     double        fp8_max,
                                     int64_t       kernel_mode) {
    CHECK_INPUT(input);
    CHECK_INPUT(freqs_cis);
    CHECK_INPUT(position_ids);
    CHECK_CUDA(output_q);
    CHECK_CUDA(output_s);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input must be bfloat16");
    TORCH_CHECK(freqs_cis.scalar_type() == at::ScalarType::ComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(position_ids.scalar_type() == at::ScalarType::Int || position_ids.scalar_type() == at::ScalarType::Long,
                "position_ids must be int32 or int64");
    TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output_q must be float8_e4m3fn");
    TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Int, "output_s must be int32 UE8M0 scale");
    TORCH_CHECK(input.device() == freqs_cis.device(), "freqs_cis must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == position_ids.device(), "position_ids must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == output_q.device(), "output_q must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == output_s.device(), "output_s must be on the same CUDA device as input");
    TORCH_CHECK(input.dim() == 3 || input.dim() == 4, "input must be [M,H,D] or [B,S,H,D]");
    TORCH_CHECK(input.stride(input.dim() - 1) == 1, "input last-dim stride must be 1");
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must be [max_pos, rope_head_dim/2]");
    TORCH_CHECK(position_ids.dim() == 1, "position_ids must be 1D [M]");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and > 0, got ", eps);
    TORCH_CHECK(std::isfinite(fp8_max) && fp8_max > 0.0, "fp8_max must be finite and > 0, got ", fp8_max);

    const int64_t m = input.dim() == 4 ? input.size(0) * input.size(1) : input.size(0);
    const int64_t h = input.size(input.dim() - 2);
    const int64_t d = input.size(input.dim() - 1);
    TORCH_CHECK(position_ids.size(0) == m, "position_ids length must match M, got ", position_ids.size(0), " vs ", m);
    TORCH_CHECK(n_groups > 0 && heads_per_group > 0, "n_groups and heads_per_group must be positive");
    TORCH_CHECK(h == n_groups * heads_per_group,
                "H must equal n_groups * heads_per_group, got H=",
                h,
                " n_groups=",
                n_groups,
                " heads_per_group=",
                heads_per_group);
    TORCH_CHECK(d == nope_dim + rope_head_dim,
                "head_dim must equal nope_dim + rope_head_dim, got D=",
                d,
                " nope_dim=",
                nope_dim,
                " rope_head_dim=",
                rope_head_dim);
    TORCH_CHECK(d % kQuantGroupSize == 0 && d <= 512,
                "head_dim must be a positive multiple of 128 and <=512, got ",
                d);
    TORCH_CHECK(rope_head_dim > 0 && rope_head_dim % 2 == 0 && rope_head_dim <= d,
                "rope_head_dim must be even and within head_dim");
    TORCH_CHECK(nope_dim % kQuantGroupSize == kQuantGroupSize - rope_head_dim,
                "RoPE must occupy the tail of the last 128-wide quant block");
    TORCH_CHECK(freqs_cis.size(1) == rope_head_dim / 2,
                "freqs_cis dim1 must equal rope_head_dim/2, got ",
                freqs_cis.size(1),
                " vs ",
                rope_head_dim / 2);

    const int64_t d_per_group     = heads_per_group * d;
    const int64_t chunks_per_head = d / kQuantGroupSize;
    TORCH_CHECK(chunks_per_head <= 4, "chunks_per_head must be <=4 for int32 UE8M0 packing");
    TORCH_CHECK(output_q.dim() == 3, "output_q must be [M, n_groups, heads_per_group * D]");
    TORCH_CHECK(output_q.size(0) == m && output_q.size(1) == n_groups && output_q.size(2) == d_per_group,
                "output_q shape mismatch");
    TORCH_CHECK(output_q.stride(2) == 1, "output_q inner stride must be 1");
    TORCH_CHECK(output_s.dim() == 3, "output_s must be [M, n_groups, heads_per_group]");
    TORCH_CHECK(output_s.size(0) == m && output_s.size(1) == n_groups && output_s.size(2) == heads_per_group,
                "output_s shape mismatch");
    TORCH_CHECK(output_s.stride(0) == 1, "output_s stride(0) must be 1 for TMA-aligned scale layout");

    checkLaunchRange(m, "M");
    checkLaunchRange(h, "H");
    checkLaunchRange(d, "D");
    checkLaunchRange(n_groups, "n_groups");
    checkLaunchRange(heads_per_group, "heads_per_group");
    checkLaunchRange(freqs_cis.stride(0), "freqs_cis stride(0)");
    checkLaunchRange(output_q.stride(0), "output_q stride(0)");
    checkLaunchRange(output_q.stride(1), "output_q stride(1)");
    checkLaunchRange(output_s.stride(0), "output_s stride(0)");
    checkLaunchRange(output_s.stride(1), "output_s stride(1)");
    checkLaunchRange(output_s.stride(2), "output_s stride(2)");

    if (m == 0) {
        return;
    }
    if (position_ids.scalar_type() == at::ScalarType::Int) {
        dispatchLaunch<int32_t>(
            input,
            freqs_cis,
            position_ids,
            output_q,
            output_s,
            m,
            h,
            d,
            n_groups,
            heads_per_group,
            rope_head_dim,
            chunks_per_head,
            eps,
            fp8_max,
            kernel_mode);
    } else {
        dispatchLaunch<int64_t>(
            input,
            freqs_cis,
            position_ids,
            output_q,
            output_s,
            m,
            h,
            d,
            n_groups,
            heads_per_group,
            rope_head_dim,
            chunks_per_head,
            eps,
            fp8_max,
            kernel_mode);
    }
}

}  // namespace rtp_llm
