// Adapted from /tmp/sglang_m3_latest/python/sglang/jit_kernel/csrc/minimax/minimax_decode_topk.cuh.
// Licensed under the Apache License, Version 2.0.

#include "rtp_llm/models_py/bindings/cuda/kernels/minimax_decode_topk.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace rtp_llm {

namespace {

static constexpr unsigned kWarpSyncMask = 0xFFFFFFFFu;

struct MinimaxDecodeTopKTrait {
    static constexpr uint32_t kMaxTopK        = 32;
    static constexpr uint32_t kCTASize        = 512;
    static constexpr uint32_t kWarpThreads    = 32;
    static constexpr uint32_t kNumWarps       = kCTASize / kWarpThreads;
    static constexpr uint32_t kMaxNumBlocks   = 4096;
    static constexpr uint32_t kSmallThreshold = 8 * kNumWarps;
    static constexpr uint32_t kRadixBits      = 8;
    static constexpr uint32_t kRadixSize      = 1 << kRadixBits;

    struct Smem {
        uint32_t warp_sum[kNumWarps];
        alignas(128) uint32_t counter;
        alignas(128) uint32_t counter_final;
        alignas(128) uint32_t threshold_bin;
        uint32_t equal_count;
        uint32_t above_count;
        uint32_t histogram[2][kRadixSize];
        float    small_scores[kSmallThreshold];
    };

    __device__ static __forceinline__ bool is_greater(float x, float y, int32_t delta) {
        return (x > y) || ((x == y) && delta < 0);
    }

    __device__ static __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
        for (uint32_t offset = 1; offset < kWarpThreads; offset *= 2) {
            uint32_t n = __shfl_up_sync(kWarpSyncMask, val, offset, kWarpThreads);
            if (lane_id >= offset) {
                val += n;
            }
        }
        return val;
    }

    __device__ static __forceinline__ uint32_t warp_reduce_sum(uint32_t val) {
#pragma unroll
        for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(kWarpSyncMask, val, offset, kWarpThreads);
        }
        return __shfl_sync(kWarpSyncMask, val, 0, kWarpThreads);
    }

    __device__ static __forceinline__ float clip_nan(float x) {
        return x != x ? -CUDART_INF_F : x;
    }

    __device__ static __forceinline__ uint32_t score_to_key(float x) {
        uint32_t b = __float_as_uint(x);
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    }

    __device__ static void
    find_threshold(uint32_t* histogram, uint32_t total_active, uint32_t topk_remain, Smem* smem) {
        const uint32_t tx      = threadIdx.x;
        const uint32_t warp_id = tx / kWarpThreads;
        const uint32_t lane_id = tx % kWarpThreads;

        uint32_t hist_val = 0;
        uint32_t warp_inc = 0;
        if (tx < kRadixSize) {
            hist_val = histogram[tx];
            warp_inc = warp_inclusive_sum(lane_id, hist_val);
            if (lane_id == kWarpThreads - 1) {
                smem->warp_sum[warp_id] = warp_inc;
            }
        }
        __syncthreads();
        if (tx < kRadixSize) {
            uint32_t inter_val = lane_id < warp_id ? smem->warp_sum[lane_id] : 0;
            uint32_t inter     = warp_reduce_sum(inter_val);
            uint32_t prefix    = inter + warp_inc;
            uint32_t above     = total_active - prefix;
            if (above < topk_remain && above + hist_val >= topk_remain) {
                smem->threshold_bin = tx;
                smem->above_count   = above;
                smem->equal_count   = hist_val;
            }
        }
        __syncthreads();
    }

    __device__ static void forward(const float* __restrict__ scores,
                                   uint32_t num_blocks,
                                   int32_t* __restrict__ topk_out,
                                   uint32_t topk,
                                   Smem*    smem) {
        const uint32_t tx      = threadIdx.x;
        const uint32_t warp_id = tx / kWarpThreads;
        const uint32_t lane_id = tx % kWarpThreads;

        if (num_blocks <= kSmallThreshold) {
            if (tx < num_blocks) {
                smem->small_scores[tx] = clip_nan(scores[tx]);
            }
            __syncthreads();
            constexpr uint32_t kNumCandidates = kSmallThreshold / kNumWarps;
            constexpr uint32_t kNumTargets    = kSmallThreshold / kWarpThreads;
            float              candidates[kNumCandidates];
            float              target[kNumTargets];
#pragma unroll
            for (uint32_t i = 0; i < kNumTargets; ++i) {
                uint32_t idx = lane_id + i * kWarpThreads;
                target[i]    = idx < num_blocks ? smem->small_scores[idx] : -CUDART_INF_F;
            }
#pragma unroll
            for (uint32_t i = 0; i < kNumCandidates; ++i) {
                uint32_t idx  = warp_id + i * kNumWarps;
                candidates[i] = idx < num_blocks ? smem->small_scores[idx] : -CUDART_INF_F;
            }

#pragma unroll
            for (uint32_t i = 0; i < kNumCandidates; ++i) {
                int32_t idx = static_cast<int32_t>(warp_id + i * kNumWarps);
                if (idx >= static_cast<int32_t>(num_blocks)) {
                    break;
                }
                uint32_t rank = 0;
#pragma unroll
                for (uint32_t j = 0; j < kNumTargets; ++j) {
                    int32_t delta = static_cast<int32_t>(lane_id + j * kWarpThreads) - idx;
                    rank += is_greater(target[j], candidates[i], delta);
                }
                rank = warp_reduce_sum(rank);
                if (lane_id == 0 && rank < topk) {
                    topk_out[rank] = idx;
                }
            }
        } else if (num_blocks <= kCTASize) {
            bool     active      = tx < num_blocks;
            float    value       = active ? clip_nan(scores[tx]) : -CUDART_INF_F;
            uint32_t key         = score_to_key(value);
            uint32_t topk_remain = topk;
            uint32_t write_pos   = topk;
            if (tx < kRadixSize) {
                smem->histogram[0][tx] = 0;
            }
            if (tx == kRadixSize) {
                smem->counter       = 0;
                smem->counter_final = 0;
            }
            __syncthreads();
            uint32_t total_active = num_blocks;

#pragma unroll
            for (int round = 0; round < 4; ++round) {
                uint32_t  shift     = 24 - round * 8;
                uint32_t  bin       = (key >> shift) & 0xFFu;
                uint32_t  hb        = round & 1;
                uint32_t* histogram = smem->histogram[hb];

                if (active) {
                    atomicAdd(&histogram[bin], 1);
                }
                if (round < 3 && tx < kRadixSize) {
                    smem->histogram[hb ^ 1][tx] = 0;
                }
                __syncthreads();

                find_threshold(histogram, total_active, topk_remain, smem);
                uint32_t threshold_bin = smem->threshold_bin;
                uint32_t above_count   = smem->above_count;
                uint32_t equal_count   = smem->equal_count;

                if (round < 3) {
                    total_active = equal_count;
                }
                topk_remain -= above_count;

                if (active) {
                    if (bin > threshold_bin) {
                        write_pos = atomicAdd(&smem->counter, 1);
                        active    = false;
                    } else if (bin < threshold_bin) {
                        active = false;
                    } else if (round == 3) {
                        write_pos = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
                    }
                }

                if (round == 3 || topk_remain == 0) {
                    break;
                }
            }
            if (write_pos < topk) {
                topk_out[write_pos] = static_cast<int32_t>(tx);
            }
        } else {
            constexpr uint32_t kIters = kMaxNumBlocks / kCTASize;
            uint32_t           key[kIters];
            uint32_t           active = 0;
#pragma unroll
            for (uint32_t i = 0; i < kIters; ++i) {
                uint32_t idx = i * kCTASize + tx;
                if (idx < num_blocks) {
                    key[i] = score_to_key(clip_nan(scores[idx]));
                    active |= 1u << i;
                }
            }
            if (tx < kRadixSize) {
                smem->histogram[0][tx] = 0;
            }
            if (tx == kRadixSize) {
                smem->counter       = 0;
                smem->counter_final = 0;
            }
            __syncthreads();

            uint32_t topk_remain  = topk;
            uint32_t total_active = num_blocks;

#pragma unroll
            for (int round = 0; round < 4; ++round) {
                uint32_t shift = 24 - round * 8;
                uint32_t hb    = round & 1;
#pragma unroll
                for (uint32_t i = 0; i < kIters; ++i) {
                    if (active & (1u << i)) {
                        atomicAdd(&smem->histogram[hb][(key[i] >> shift) & 0xFFu], 1);
                    }
                }
                if (round < 3 && tx < kRadixSize) {
                    smem->histogram[hb ^ 1][tx] = 0;
                }
                __syncthreads();

                find_threshold(smem->histogram[hb], total_active, topk_remain, smem);
                uint32_t threshold_bin = smem->threshold_bin;
                uint32_t above_count   = smem->above_count;
                uint32_t equal_count   = smem->equal_count;

                if (round < 3) {
                    total_active = equal_count;
                }
                topk_remain -= above_count;

#pragma unroll
                for (uint32_t i = 0; i < kIters; ++i) {
                    if (active & (1u << i)) {
                        uint32_t bin = (key[i] >> shift) & 0xFFu;
                        if (bin > threshold_bin) {
                            topk_out[atomicAdd(&smem->counter, 1)] = static_cast<int32_t>(i * kCTASize + tx);
                            active &= ~(1u << i);
                        } else if (bin < threshold_bin) {
                            active &= ~(1u << i);
                        } else if (round == 3) {
                            uint32_t pos = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
                            if (pos < topk) {
                                topk_out[pos] = static_cast<int32_t>(i * kCTASize + tx);
                            }
                        }
                    }
                }

                if (round == 3 || topk_remain == 0) {
                    break;
                }
            }
        }
    }
};

template<typename SeqLenT>
__global__ __launch_bounds__(MinimaxDecodeTopKTrait::kCTASize) void minimax_decode_topk_kernel(
    const float* __restrict__ score,
    const SeqLenT* __restrict__ seq_lens,
    int32_t* __restrict__ topk_idx,
    int batch,
    int num_heads,
    int max_seqblock,
    int block_size,
    int topk) {
    int b  = blockIdx.x;
    int h  = blockIdx.y;
    int tx = threadIdx.x;

    int64_t  seq_len        = static_cast<int64_t>(seq_lens[b]);
    int      num_blocks_raw = static_cast<int>((seq_len + block_size - 1) / block_size);
    int      num_blocks     = num_blocks_raw < max_seqblock ? num_blocks_raw : max_seqblock;
    int32_t* out            = topk_idx + (static_cast<int64_t>(h) * batch + b) * topk;

    if (num_blocks <= topk) {
        for (int i = tx; i < topk; i += MinimaxDecodeTopKTrait::kCTASize) {
            out[i] = i < num_blocks ? i : -1;
        }
        return;
    }

    const float* row = score + (static_cast<int64_t>(h) * batch + b) * max_seqblock;
    __shared__ MinimaxDecodeTopKTrait::Smem smem;
    MinimaxDecodeTopKTrait::forward(row, static_cast<uint32_t>(num_blocks), out, static_cast<uint32_t>(topk), &smem);
}

template<typename SeqLenT>
void launch_minimax_decode_topk(
    const at::Tensor& score, const at::Tensor& seq_lens, at::Tensor& topk_idx, int64_t block_size, int64_t topk) {
    int batch        = static_cast<int>(score.size(1));
    int num_heads    = static_cast<int>(score.size(0));
    int max_seqblock = static_cast<int>(score.size(2));
    if (batch == 0 || num_heads == 0) {
        return;
    }
    dim3         grid(static_cast<unsigned>(batch), static_cast<unsigned>(num_heads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    minimax_decode_topk_kernel<SeqLenT>
        <<<grid, MinimaxDecodeTopKTrait::kCTASize, 0, stream>>>(score.data_ptr<float>(),
                                                                seq_lens.data_ptr<SeqLenT>(),
                                                                topk_idx.data_ptr<int32_t>(),
                                                                batch,
                                                                num_heads,
                                                                max_seqblock,
                                                                static_cast<int>(block_size),
                                                                static_cast<int>(topk));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "minimax_decode_topk failed: ", cudaGetErrorString(err));
}

}  // namespace

void minimax_decode_topk(
    const at::Tensor& score, const at::Tensor& seq_lens, at::Tensor& topk_idx, int64_t block_size, int64_t topk) {
#ifndef USE_ROCM
    TORCH_CHECK(score.is_cuda(), "score must be CUDA tensor");
    TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA tensor");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be CUDA tensor");
    TORCH_CHECK(score.dtype() == torch::kFloat32, "score must be float32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32 || seq_lens.dtype() == torch::kInt64,
                "seq_lens must be int32 or int64");
    TORCH_CHECK(topk_idx.dtype() == torch::kInt32, "topk_idx must be int32");
    TORCH_CHECK(score.dim() == 3, "score must have shape [num_heads, batch, max_seqblock]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
    TORCH_CHECK(topk_idx.dim() == 3, "topk_idx must have shape [num_heads, batch, topk]");
    TORCH_CHECK(score.is_contiguous(), "score must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(seq_lens.size(0) == score.size(1), "seq_lens batch mismatch");
    TORCH_CHECK(topk_idx.size(0) == score.size(0) && topk_idx.size(1) == score.size(1) && topk_idx.size(2) == topk,
                "topk_idx shape mismatch");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(topk > 0 && topk <= static_cast<int64_t>(MinimaxDecodeTopKTrait::kMaxTopK),
                "topk must be in [1, 32], got ",
                topk);
    TORCH_CHECK(score.size(2) <= static_cast<int64_t>(MinimaxDecodeTopKTrait::kMaxNumBlocks),
                "max_seqblock exceeds 4096: ",
                score.size(2));

    if (seq_lens.dtype() == torch::kInt32) {
        launch_minimax_decode_topk<int32_t>(score, seq_lens, topk_idx, block_size, topk);
    } else {
        launch_minimax_decode_topk<int64_t>(score, seq_lens, topk_idx, block_size, topk);
    }
#else
    TORCH_CHECK(false, "minimax_decode_topk is not supported on ROCm");
#endif
}

}  // namespace rtp_llm
