#include "dsv4_indexed_rmsnorm_rope.h"

#include "fp8_ue8m0_scale_layout.cuh"
#include "util.h"
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>

namespace rtp_llm {
namespace {

constexpr int kWarpSize    = 32;
constexpr int kWarpsPerCta = 8;
constexpr int kBlockSize   = kWarpSize * kWarpsPerCta;
constexpr int64_t kQD128LargeMinM = 2049;
constexpr int64_t kQD512LargeMinM = 97;

__device__ __forceinline__ float bf16ToFloat(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_rmsnorm_rope_warp_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int     d,
    int     rope_dim,
    int     freq_stride_n,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps,
    bool    has_weight) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * kWarpsPerCta) + warp_id;
    if (row >= rows) {
        return;
    }

    const __nv_bfloat16* row_input = input + row * input_stride;
    __nv_bfloat16*       row_out   = output + row * output_stride;

    float local_sumsq = 0.0f;
    for (int col = lane; col < d; col += kWarpSize) {
        const float x = bf16ToFloat(row_input[col]);
        local_sumsq += x * x;
    }
    const float sumsq   = __shfl_sync(0xffffffff, warpReduceSum(local_sumsq), 0);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(d) + eps);

    const int64_t token_idx   = row / static_cast<int64_t>(freq_stride_n);
    const int64_t pos         = static_cast<int64_t>(position_ids[token_idx]);
    const int     nope_offset = d - rope_dim;

    for (int col = lane; col < d; col += kWarpSize) {
        float y = bf16ToFloat(row_input[col]) * inv_rms;
        if (has_weight) {
            y *= bf16ToFloat(weight[col]);
        }

        if (col >= nope_offset) {
            const int rope_local  = col - nope_offset;
            const int pair        = rope_local >> 1;
            const float partner = __shfl_xor_sync(0xffffffff, y, 1);
            const float2 freq = freqs_cis[pos * freqs_stride + pair];
            if ((rope_local & 1) == 0) {
                y = y * freq.x - partner * freq.y;
            } else {
                y = y * freq.x + partner * freq.y;
            }
        }
        row_out[col] = __float2bfloat16(y);
    }
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_rmsnorm_rope_block_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int     d,
    int     rope_dim,
    int     freq_stride_n,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps,
    bool    has_weight) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const __nv_bfloat16* row_input = input + row * input_stride;
    __nv_bfloat16* row_out = output + row * output_stride;
    float local_sumsq = 0.0f;
    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        const float x = bf16ToFloat(row_input[col]);
        local_sumsq += x * x;
    }
    const float sumsq = blockReduceSum(local_sumsq);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(d) + eps);
    const int64_t token_idx = row / static_cast<int64_t>(freq_stride_n);
    const int64_t pos = static_cast<int64_t>(position_ids[token_idx]);
    const int nope_offset = d - rope_dim;

    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        float y = bf16ToFloat(row_input[col]) * inv_rms;
        if (has_weight) {
            y *= bf16ToFloat(weight[col]);
        }
        if (col >= nope_offset) {
            const int rope_local = col - nope_offset;
            const int pair = rope_local >> 1;
            float partner = bf16ToFloat(row_input[col ^ 1]) * inv_rms;
            if (has_weight) {
                partner *= bf16ToFloat(weight[col ^ 1]);
            }
            const float2 freq = freqs_cis[pos * freqs_stride + pair];
            if ((rope_local & 1) == 0) {
                y = y * freq.x - partner * freq.y;
            } else {
                y = y * freq.x + partner * freq.y;
            }
        }
        row_out[col] = __float2bfloat16(y);
    }
}

template<typename index_t, int GROUP_HEADS>
__global__ __launch_bounds__(GROUP_HEADS * kWarpSize) void dsv4_indexed_rmsnorm_rope_group_warp_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t token_rows,
    int     d,
    int     rope_dim,
    int     h,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int head_base = static_cast<int>(blockIdx.y) * GROUP_HEADS;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (token >= token_rows || warp_id >= GROUP_HEADS) {
        return;
    }

    __shared__ float2 shared_freq[256];
    const int half_rope = rope_dim / 2;
    const int64_t pos = static_cast<int64_t>(position_ids[token]);
    for (int pair = threadIdx.x; pair < half_rope; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[pos * freqs_stride + pair];
    }
    __syncthreads();

    const int head = head_base + warp_id;
    if (head >= h) {
        return;
    }
    const int64_t row = token * static_cast<int64_t>(h) + head;
    const __nv_bfloat16* row_input = input + row * input_stride;
    __nv_bfloat16* row_out = output + row * output_stride;

    float local_sumsq = 0.0f;
    for (int col = lane; col < d; col += kWarpSize) {
        const float x = bf16ToFloat(row_input[col]);
        local_sumsq += x * x;
    }
    const float sumsq = __shfl_sync(0xffffffff, warpReduceSum(local_sumsq), 0);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(d) + eps);
    const int nope_offset = d - rope_dim;

    for (int col = lane; col < d; col += kWarpSize) {
        float y = bf16ToFloat(row_input[col]) * inv_rms;
        if (col >= nope_offset) {
            const int rope_local = col - nope_offset;
            const int pair = rope_local >> 1;
            const float partner = __shfl_xor_sync(0xffffffff, y, 1);
            const float2 freq = shared_freq[pair];
            if ((rope_local & 1) == 0) {
                y = y * freq.x - partner * freq.y;
            } else {
                y = y * freq.x + partner * freq.y;
            }
        }
        row_out[col] = __float2bfloat16(y);
    }
}

template<typename index_t, int ROWS_PER_WARP>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_rmsnorm_rope_q_d128_cached_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int     h,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    constexpr int kD = 128;
    constexpr int kRopeDim = 64;
    constexpr int kNopeOffset = kD - kRopeDim;
    const int64_t row_base =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerCta + warp_id) * ROWS_PER_WARP;
    const int col_base = lane << 2;
#pragma unroll
    for (int row_iter = 0; row_iter < ROWS_PER_WARP; ++row_iter) {
        const int64_t row = row_base + row_iter;
        if (row >= rows) {
            return;
        }
        const int64_t token = row / static_cast<int64_t>(h);
        const int64_t pos = static_cast<int64_t>(position_ids[token]);
        const __nv_bfloat16* row_input = input + row * input_stride;
        __nv_bfloat16* row_out = output + row * output_stride;

        float vals[4];
        float local_sumsq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = bf16ToFloat(row_input[col_base + i]);
            vals[i] = x;
            local_sumsq += x * x;
        }
        const float sumsq = __shfl_sync(0xffffffff, warpReduceSum(local_sumsq), 0);
        const float inv_rms = rsqrtf(sumsq / static_cast<float>(kD) + eps);

        float y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = vals[i] * inv_rms;
        }
        if (col_base + 3 >= kNopeOffset) {
            const float rope_y[4] = {y[0], y[1], y[2], y[3]};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = col_base + i;
                if (col >= kNopeOffset) {
                    const int rope_local = col - kNopeOffset;
                    const int pair = rope_local >> 1;
                    const float partner = rope_y[i ^ 1];
                    const float2 freq = freqs_cis[pos * freqs_stride + pair];
                    if ((rope_local & 1) == 0) {
                        y[i] = rope_y[i] * freq.x - partner * freq.y;
                    } else {
                        y[i] = rope_y[i] * freq.x + partner * freq.y;
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            row_out[col_base + i] = __float2bfloat16(y[i]);
        }
    }
}

template<typename index_t, int GROUP_HEADS>
__global__ __launch_bounds__(GROUP_HEADS * kWarpSize) void
dsv4_indexed_rmsnorm_rope_q_d128_cached_group8_large_m_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t token_rows,
    int     h,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps) {
    const int64_t token   = static_cast<int64_t>(blockIdx.x);
    const int     warp_id = threadIdx.x >> 5;
    const int     lane    = threadIdx.x & 31;
    const int     head    = static_cast<int>(blockIdx.y) * GROUP_HEADS + warp_id;
    if (token >= token_rows || warp_id >= GROUP_HEADS) {
        return;
    }

    constexpr int kD          = 128;
    constexpr int kRopeDim    = 64;
    constexpr int kNopeOffset = kD - kRopeDim;
    __shared__ int64_t shared_pos;
    __shared__ float2  shared_freq[kRopeDim / 2];
    if (threadIdx.x == 0) {
        shared_pos = static_cast<int64_t>(position_ids[token]);
    }
    __syncthreads();
    for (int pair = threadIdx.x; pair < kRopeDim / 2; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[shared_pos * freqs_stride + pair];
    }
    __syncthreads();

    if (head >= h) {
        return;
    }
    const int64_t row = token * static_cast<int64_t>(h) + head;
    const __nv_bfloat16* row_input = input + row * input_stride;
    __nv_bfloat16*       row_out   = output + row * output_stride;
    const int            col_base  = lane << 2;

    float vals[4];
    float local_sumsq = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x = bf16ToFloat(row_input[col_base + i]);
        vals[i] = x;
        local_sumsq += x * x;
    }
    const float sumsq   = __shfl_sync(0xffffffff, warpReduceSum(local_sumsq), 0);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(kD) + eps);

    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[i] = vals[i] * inv_rms;
    }
    if (col_base + 3 >= kNopeOffset) {
        const float rope_y[4] = {y[0], y[1], y[2], y[3]};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = col_base + i;
            if (col >= kNopeOffset) {
                const int    rope_local = col - kNopeOffset;
                const int    pair       = rope_local >> 1;
                const float  partner    = rope_y[i ^ 1];
                const float2 freq       = shared_freq[pair];
                if ((rope_local & 1) == 0) {
                    y[i] = rope_y[i] * freq.x - partner * freq.y;
                } else {
                    y[i] = rope_y[i] * freq.x + partner * freq.y;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        row_out[col_base + i] = __float2bfloat16(y[i]);
    }
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_rmsnorm_rope_q_d128_cached_token64_large_m_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t token_rows,
    int     h,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps) {
    const int64_t token   = static_cast<int64_t>(blockIdx.x);
    const int     warp_id = threadIdx.x >> 5;
    const int     lane    = threadIdx.x & 31;
    if (token >= token_rows) {
        return;
    }

    constexpr int kD          = 128;
    constexpr int kRopeDim    = 64;
    constexpr int kNopeOffset = kD - kRopeDim;
    __shared__ int64_t shared_pos;
    __shared__ float2  shared_freq[kRopeDim / 2];
    if (threadIdx.x == 0) {
        shared_pos = static_cast<int64_t>(position_ids[token]);
    }
    __syncthreads();
    for (int pair = threadIdx.x; pair < kRopeDim / 2; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[shared_pos * freqs_stride + pair];
    }
    __syncthreads();

    const int col_base = lane << 2;
    for (int head = warp_id; head < h; head += kWarpsPerCta) {
        const int64_t row = token * static_cast<int64_t>(h) + head;
        const __nv_bfloat16* row_input = input + row * input_stride;
        __nv_bfloat16*       row_out   = output + row * output_stride;

        float vals[4];
        float local_sumsq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = bf16ToFloat(row_input[col_base + i]);
            vals[i] = x;
            local_sumsq += x * x;
        }
        const float sumsq   = __shfl_sync(0xffffffff, warpReduceSum(local_sumsq), 0);
        const float inv_rms = rsqrtf(sumsq / static_cast<float>(kD) + eps);

        float y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = vals[i] * inv_rms;
        }
        if (col_base + 3 >= kNopeOffset) {
            const float rope_y[4] = {y[0], y[1], y[2], y[3]};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = col_base + i;
                if (col >= kNopeOffset) {
                    const int    rope_local = col - kNopeOffset;
                    const int    pair       = rope_local >> 1;
                    const float  partner    = rope_y[i ^ 1];
                    const float2 freq       = shared_freq[pair];
                    if ((rope_local & 1) == 0) {
                        y[i] = rope_y[i] * freq.x - partner * freq.y;
                    } else {
                        y[i] = rope_y[i] * freq.x + partner * freq.y;
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            row_out[col_base + i] = __float2bfloat16(y[i]);
        }
    }
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_rmsnorm_rope_q_d512_cached_token64_large_m_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t token_rows,
    int     h,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float   eps) {
    const int64_t token   = static_cast<int64_t>(blockIdx.x);
    const int     warp_id = threadIdx.x >> 5;
    const int     lane    = threadIdx.x & 31;
    if (token >= token_rows) {
        return;
    }

    constexpr int kD            = 512;
    constexpr int kRopeDim      = 64;
    constexpr int kNopeOffset   = kD - kRopeDim;
    constexpr int kPairsPerLane = kD / (2 * kWarpSize);
    constexpr int kNopePairs    = kNopeOffset / 2;
    __shared__ int64_t shared_pos;
    __shared__ float2  shared_freq[kRopeDim / 2];
    if (threadIdx.x == 0) {
        shared_pos = static_cast<int64_t>(position_ids[token]);
    }
    __syncthreads();
    for (int pair = threadIdx.x; pair < kRopeDim / 2; pair += blockDim.x) {
        shared_freq[pair] = freqs_cis[shared_pos * freqs_stride + pair];
    }
    __syncthreads();

    float2 vals[kPairsPerLane];
    for (int head = warp_id; head < h; head += kWarpsPerCta) {
        const int64_t row = token * static_cast<int64_t>(h) + head;
        const __nv_bfloat16* row_input = input + row * input_stride;
        __nv_bfloat16*       row_out   = output + row * output_stride;
        const __nv_bfloat162* row_input2 = reinterpret_cast<const __nv_bfloat162*>(row_input);
        __nv_bfloat162*       row_out2   = reinterpret_cast<__nv_bfloat162*>(row_out);

        float local_sumsq = 0.0f;
#pragma unroll
        for (int i = 0; i < kPairsPerLane; ++i) {
            const int pair_idx = lane + i * kWarpSize;
            const float2 x = __bfloat1622float2(row_input2[pair_idx]);
            vals[i] = x;
            local_sumsq += x.x * x.x + x.y * x.y;
        }
        const float sumsq = __shfl_sync(0xffffffff, warpReduceSum(local_sumsq), 0);
        const float inv_rms = rsqrtf(sumsq / static_cast<float>(kD) + eps);

#pragma unroll
        for (int i = 0; i < kPairsPerLane; ++i) {
            const int pair_idx = lane + i * kWarpSize;
            float2 y = {vals[i].x * inv_rms, vals[i].y * inv_rms};
            if (pair_idx >= kNopePairs) {
                const float2 freq = shared_freq[pair_idx - kNopePairs];
                const float real = y.x;
                const float imag = y.y;
                y.x = real * freq.x - imag * freq.y;
                y.y = real * freq.y + imag * freq.x;
            }
            row_out2[pair_idx] = __floats2bfloat162_rn(y.x, y.y);
        }
    }
}

template<typename index_t>
__global__ __launch_bounds__(kBlockSize) void dsv4_indexed_rmsnorm_rope_kv_d512_cached_block_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float2* __restrict__ freqs_cis,
    const index_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int64_t input_stride,
    int64_t output_stride,
    int64_t freqs_stride,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    constexpr int kD = 512;
    constexpr int kRopeDim = 64;
    constexpr int kNopeOffset = kD - kRopeDim;

    const int tid = threadIdx.x;
    const int high_col = tid + kBlockSize;
    const __nv_bfloat16* row_input = input + row * input_stride;
    __nv_bfloat16* row_out = output + row * output_stride;

    const float x0 = bf16ToFloat(row_input[tid]);
    const float x1 = bf16ToFloat(row_input[high_col]);
    const float sumsq = blockReduceSum(x0 * x0 + x1 * x1);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(kD) + eps);

    float y0 = x0 * inv_rms * bf16ToFloat(weight[tid]);
    float y1 = x1 * inv_rms * bf16ToFloat(weight[high_col]);

    if (high_col >= kNopeOffset) {
        const int rope_local = high_col - kNopeOffset;
        const int pair = rope_local >> 1;
        const int64_t pos = static_cast<int64_t>(position_ids[row]);
        const float partner = __shfl_xor_sync(0xffffffff, y1, 1);
        const float2 freq = freqs_cis[pos * freqs_stride + pair];
        if ((rope_local & 1) == 0) {
            y1 = y1 * freq.x - partner * freq.y;
        } else {
            y1 = y1 * freq.x + partner * freq.y;
        }
    }

    row_out[tid] = __float2bfloat16(y0);
    row_out[high_col] = __float2bfloat16(y1);
}

void checkLaunchRange(int64_t value, const char* name) {
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
                name,
                " is too large for dsv4 indexed RMSNorm+RoPE launch: ",
                value);
}

void checkAligned16(const torch::Tensor& tensor, const char* name) {
    if (tensor.numel() == 0) {
        return;
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                name,
                " data_ptr must be 16-byte aligned for vectorized/bf16 CUDA access");
}

template<typename index_t>
void launchQD128Cached(torch::Tensor input,
                       torch::Tensor freqs_cis,
                       torch::Tensor position_ids,
                       torch::Tensor output,
                       int64_t       rows,
                       int64_t       h,
                       int           rows_per_warp,
                       double        eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_Q_D128_CACHED_TILED(ROWS_PER_WARP)                                                                      \
    do {                                                                                                               \
        const int64_t grid_x =                                                                                         \
            (rows + kWarpsPerCta * static_cast<int64_t>(ROWS_PER_WARP) - 1)                                            \
            / (kWarpsPerCta * static_cast<int64_t>(ROWS_PER_WARP));                                                    \
        dsv4_indexed_rmsnorm_rope_q_d128_cached_tiled_kernel<index_t, ROWS_PER_WARP>                                   \
            <<<static_cast<int>(grid_x), kBlockSize, 0, stream>>>(                                                     \
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),                                              \
                reinterpret_cast<const float2*>(freqs_cis.data_ptr()),                                                 \
                reinterpret_cast<const index_t*>(position_ids.data_ptr()),                                             \
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),                                                   \
                rows,                                                                                                  \
                static_cast<int>(h),                                                                                   \
                input.stride(input.dim() - 2),                                                                         \
                output.stride(output.dim() - 2),                                                                       \
                freqs_cis.stride(0),                                                                                   \
                static_cast<float>(eps));                                                                              \
    } while (false)
    if (rows_per_warp == 4) {
        LAUNCH_Q_D128_CACHED_TILED(4);
    } else if (rows_per_warp == 2) {
        LAUNCH_Q_D128_CACHED_TILED(2);
    } else {
        LAUNCH_Q_D128_CACHED_TILED(1);
    }
#undef LAUNCH_Q_D128_CACHED_TILED
}

template<typename index_t>
void launchKVD512Cached(torch::Tensor input,
                        torch::Tensor weight,
                        torch::Tensor freqs_cis,
                        torch::Tensor position_ids,
                        torch::Tensor output,
                        int64_t rows,
                        double eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_indexed_rmsnorm_rope_kv_d512_cached_block_kernel<index_t>
        <<<static_cast<uint32_t>(rows), kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            rows,
            input.stride(input.dim() - 2),
            output.stride(output.dim() - 2),
            freqs_cis.stride(0),
            static_cast<float>(eps));
}

template<typename index_t>
void launchQD128CachedTokenLarge(torch::Tensor input,
                                 torch::Tensor freqs_cis,
                                 torch::Tensor position_ids,
                                 torch::Tensor output,
                                 int64_t       token_rows,
                                 int64_t       h,
                                 double        eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_indexed_rmsnorm_rope_q_d128_cached_token64_large_m_kernel<index_t>
        <<<static_cast<uint32_t>(token_rows), kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            token_rows,
            static_cast<int>(h),
            input.stride(input.dim() - 2),
            output.stride(output.dim() - 2),
            freqs_cis.stride(0),
            static_cast<float>(eps));
}

template<typename index_t>
void launchQD512CachedTokenLarge(torch::Tensor input,
                                 torch::Tensor freqs_cis,
                                 torch::Tensor position_ids,
                                 torch::Tensor output,
                                 int64_t       token_rows,
                                 int64_t       h,
                                 double        eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_indexed_rmsnorm_rope_q_d512_cached_token64_large_m_kernel<index_t>
        <<<static_cast<uint32_t>(token_rows), kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            token_rows,
            static_cast<int>(h),
            input.stride(input.dim() - 2),
            output.stride(output.dim() - 2),
            freqs_cis.stride(0),
            static_cast<float>(eps));
}

template<typename index_t, int GROUP_HEADS>
void launchQD128CachedGroupLarge(torch::Tensor input,
                                 torch::Tensor freqs_cis,
                                 torch::Tensor position_ids,
                                 torch::Tensor output,
                                 int64_t       token_rows,
                                 int64_t       h,
                                 double        eps) {
    auto       stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<uint32_t>(token_rows), static_cast<uint32_t>((h + GROUP_HEADS - 1) / GROUP_HEADS));
    dsv4_indexed_rmsnorm_rope_q_d128_cached_group8_large_m_kernel<index_t, GROUP_HEADS>
        <<<grid, GROUP_HEADS * kWarpSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
            reinterpret_cast<const index_t*>(position_ids.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            token_rows,
            static_cast<int>(h),
            input.stride(input.dim() - 2),
            output.stride(output.dim() - 2),
            freqs_cis.stride(0),
            static_cast<float>(eps));
}

int selectQD128RowsPerWarp(int64_t token_rows) {
    if (token_rows <= 128) {
        return 4;
    }
    return 2;
}

bool useQD128LargeM(int64_t token_rows) {
    return token_rows >= kQD128LargeMinM;
}

bool useQD512LargeM(int64_t token_rows) {
    return token_rows >= kQD512LargeMinM;
}

template<typename index_t>
void launch(torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor freqs_cis,
            torch::Tensor position_ids,
            torch::Tensor output,
            int64_t       rows,
            int64_t       d,
            int64_t       rope_head_dim,
            int64_t       freq_stride_n,
            bool          has_weight,
            double        eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t grid_x = (rows + kWarpsPerCta - 1) / kWarpsPerCta;
    dsv4_indexed_rmsnorm_rope_warp_kernel<index_t><<<static_cast<int>(grid_x), kBlockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        has_weight ? reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()) : nullptr,
        reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
        reinterpret_cast<const index_t*>(position_ids.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows,
        static_cast<int>(d),
        static_cast<int>(rope_head_dim),
        static_cast<int>(freq_stride_n),
        input.stride(input.dim() - 2),
        output.stride(output.dim() - 2),
        freqs_cis.stride(0),
        static_cast<float>(eps),
        has_weight);
}

template<typename index_t>
void launchBlock(torch::Tensor input,
                 torch::Tensor weight,
                 torch::Tensor freqs_cis,
                 torch::Tensor position_ids,
                 torch::Tensor output,
                 int64_t       rows,
                 int64_t       d,
                 int64_t       rope_head_dim,
                 int64_t       freq_stride_n,
                 bool          has_weight,
                 double        eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_indexed_rmsnorm_rope_block_kernel<index_t><<<static_cast<uint32_t>(rows), kBlockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        has_weight ? reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()) : nullptr,
        reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
        reinterpret_cast<const index_t*>(position_ids.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows,
        static_cast<int>(d),
        static_cast<int>(rope_head_dim),
        static_cast<int>(freq_stride_n),
        input.stride(input.dim() - 2),
        output.stride(output.dim() - 2),
        freqs_cis.stride(0),
        static_cast<float>(eps),
        has_weight);
}

template<typename index_t, int GROUP_HEADS>
void launchGroupWarp(torch::Tensor input,
                     torch::Tensor freqs_cis,
                     torch::Tensor position_ids,
                     torch::Tensor output,
                     int64_t       token_rows,
                     int64_t       h,
                     int64_t       d,
                     int64_t       rope_head_dim,
                     double        eps) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<uint32_t>(token_rows), static_cast<uint32_t>((h + GROUP_HEADS - 1) / GROUP_HEADS));
    dsv4_indexed_rmsnorm_rope_group_warp_kernel<index_t, GROUP_HEADS>
        <<<grid, GROUP_HEADS * kWarpSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const float2*>(freqs_cis.data_ptr()),
        reinterpret_cast<const index_t*>(position_ids.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        token_rows,
        static_cast<int>(d),
        static_cast<int>(rope_head_dim),
        static_cast<int>(h),
        input.stride(input.dim() - 2),
        output.stride(output.dim() - 2),
        freqs_cis.stride(0),
        static_cast<float>(eps));
}

}  // namespace

void dsv4_indexed_rmsnorm_rope(torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor freqs_cis,
                               torch::Tensor position_ids,
                               torch::Tensor output,
                               int64_t       rope_head_dim,
                               double        eps,
                               bool          has_weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(freqs_cis);
    CHECK_INPUT(position_ids);
    CHECK_INPUT(output);

    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input must be bfloat16");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be bfloat16");
    TORCH_CHECK(freqs_cis.scalar_type() == at::ScalarType::ComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(position_ids.scalar_type() == at::ScalarType::Int || position_ids.scalar_type() == at::ScalarType::Long,
                "position_ids must be int32 or int64");
    TORCH_CHECK(input.device() == output.device(), "input and output must be on the same CUDA device");
    TORCH_CHECK(input.device() == freqs_cis.device(), "freqs_cis must be on the same CUDA device as input");
    TORCH_CHECK(input.device() == position_ids.device(), "position_ids must be on the same CUDA device as input");

    TORCH_CHECK(input.dim() == 2 || input.dim() == 3 || input.dim() == 4,
                "input must be [M,D], [B,S,D], or [B,S,H,D]");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input shape");
    TORCH_CHECK(output.strides() == input.strides(), "output stride must match input stride");
    const int64_t d = input.size(input.dim() - 1);
    TORCH_CHECK(input.stride(input.dim() - 1) == 1, "input last-dim stride must be 1");
    TORCH_CHECK(output.stride(output.dim() - 1) == 1, "output last-dim stride must be 1");
    TORCH_CHECK(d > 0 && d <= 512 && d % 2 == 0, "unsupported head_dim for dsv4 indexed RMSNorm+RoPE: ", d);
    TORCH_CHECK(rope_head_dim > 0 && rope_head_dim <= d && rope_head_dim % 2 == 0,
                "rope_head_dim must be even and within head_dim, got ",
                rope_head_dim,
                " vs head_dim ",
                d);
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must be [max_pos, rope_head_dim/2]");
    TORCH_CHECK(freqs_cis.size(1) == rope_head_dim / 2,
                "freqs_cis dim1 must equal rope_head_dim/2, got ",
                freqs_cis.size(1),
                " vs ",
                rope_head_dim / 2);
    TORCH_CHECK(position_ids.dim() == 1, "position_ids must be 1D [M]");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and > 0, got ", eps);

    int64_t token_rows = 0;
    int64_t rows       = 0;
    int64_t freq_stride_n = 1;
    if (input.dim() == 4) {
        token_rows    = input.size(0) * input.size(1);
        freq_stride_n = input.size(2);
        rows          = token_rows * freq_stride_n;
    } else {
        token_rows = input.numel() / d;
        rows       = token_rows;
    }
    TORCH_CHECK(position_ids.size(0) == token_rows,
                "position_ids length must match dynamic token rows, got ",
                position_ids.size(0),
                " vs ",
                token_rows);
    if (has_weight) {
        CHECK_INPUT(weight);
        TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16, "weight must be bfloat16");
        TORCH_CHECK(weight.device() == input.device(), "weight must be on the same CUDA device as input");
        TORCH_CHECK(weight.dim() == 1 && weight.size(0) == d, "weight must be 1D [head_dim]");
        TORCH_CHECK(weight.stride(0) == 1, "weight stride(0) must be 1");
        checkAligned16(weight, "weight");
    } else {
        TORCH_CHECK(weight.numel() == 0 || weight.is_cuda(), "empty CUDA weight placeholder expected when has_weight=false");
    }
    checkLaunchRange(rows, "rows");
    checkLaunchRange(d, "head_dim");
    checkLaunchRange(freq_stride_n, "freq_stride_n");
    checkLaunchRange(freqs_cis.stride(0), "freqs_cis stride(0)");
    checkAligned16(input, "input");
    checkAligned16(output, "output");

    if (rows == 0) {
        return;
    }
    const bool use_q_d128_cached = input.dim() == 4 && !has_weight && d == 128 && rope_head_dim == 64
                                   && freq_stride_n == 64 && input.stride(input.dim() - 2) == 128
                                   && output.stride(output.dim() - 2) == 128;
    const bool use_q_d128_large_m = use_q_d128_cached && useQD128LargeM(token_rows);
    const bool use_q_d512_cached = input.dim() == 4 && !has_weight && d == 512 && rope_head_dim == 64
                                   && freq_stride_n == 64 && input.stride(input.dim() - 2) == 512
                                   && output.stride(output.dim() - 2) == 512;
    const bool use_q_d512_large_m = use_q_d512_cached && useQD512LargeM(token_rows);
    const int q_d128_rows_per_warp = selectQD128RowsPerWarp(token_rows);
    const bool use_group_warp = !use_q_d128_cached && input.dim() == 4 && !has_weight && freq_stride_n >= 2;
    const bool use_kv_d512_cached = has_weight && input.dim() != 4 && d == 512 && rope_head_dim == 64
                                    && freq_stride_n == 1 && input.stride(input.dim() - 2) == 512
                                    && output.stride(output.dim() - 2) == 512;
    if (position_ids.scalar_type() == at::ScalarType::Int) {
        if (use_q_d128_large_m) {
            launchQD128CachedTokenLarge<int32_t>(
                input, freqs_cis, position_ids, output, token_rows, freq_stride_n, eps);
        } else if (use_q_d128_cached) {
            launchQD128Cached<int32_t>(
                input, freqs_cis, position_ids, output, rows, freq_stride_n, q_d128_rows_per_warp, eps);
        } else if (use_q_d512_large_m) {
            launchQD512CachedTokenLarge<int32_t>(
                input, freqs_cis, position_ids, output, token_rows, freq_stride_n, eps);
        } else if (use_group_warp) {
            launchGroupWarp<int32_t, 8>(
                input, freqs_cis, position_ids, output, token_rows, freq_stride_n, d, rope_head_dim, eps);
        } else if (use_kv_d512_cached) {
            launchKVD512Cached<int32_t>(input, weight, freqs_cis, position_ids, output, rows, eps);
        } else {
            launch<int32_t>(
                input, weight, freqs_cis, position_ids, output, rows, d, rope_head_dim, freq_stride_n, has_weight, eps);
        }
    } else {
        if (use_q_d128_large_m) {
            launchQD128CachedTokenLarge<int64_t>(
                input, freqs_cis, position_ids, output, token_rows, freq_stride_n, eps);
        } else if (use_q_d128_cached) {
            launchQD128Cached<int64_t>(
                input, freqs_cis, position_ids, output, rows, freq_stride_n, q_d128_rows_per_warp, eps);
        } else if (use_q_d512_large_m) {
            launchQD512CachedTokenLarge<int64_t>(
                input, freqs_cis, position_ids, output, token_rows, freq_stride_n, eps);
        } else if (use_group_warp) {
            launchGroupWarp<int64_t, 8>(
                input, freqs_cis, position_ids, output, token_rows, freq_stride_n, d, rope_head_dim, eps);
        } else if (use_kv_d512_cached) {
            launchKVD512Cached<int64_t>(input, weight, freqs_cis, position_ids, output, rows, eps);
        } else {
            launch<int64_t>(
                input, weight, freqs_cis, position_ids, output, rows, d, rope_head_dim, freq_stride_n, has_weight, eps);
        }
    }
}

}  // namespace rtp_llm
