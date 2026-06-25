#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

#include <algorithm>
#include <c10/util/Exception.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace rtp_llm {

namespace {

__global__ void cudaGraphPrepareFillKernel(CudaGraphPrepareFillParams params) {
    const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int32_t region_idx = 0; region_idx < params.region_count; ++region_idx) {
        const auto region = params.regions[region_idx];
        if (region.ptr == nullptr || region.count <= 0) {
            continue;
        }
        for (int64_t i = tid; i < region.count; i += stride) {
            region.ptr[i] = region.value;
        }
    }
}

__global__ void prepareFlashInferDecodeParamsKernel(const int32_t* sequence_lengths_plus_1,
                                                    const int32_t* block_ids,
                                                    int32_t*       batch_indice,
                                                    int32_t*       page_indice,
                                                    int32_t*       decode_page_indptr,
                                                    int32_t*       paged_kv_last_page_len,
                                                    int32_t*       qo_indptr,
                                                    int32_t*       kvlen,
                                                    int32_t*       positions,
                                                    int64_t*       slot_mapping,
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block,
                                                    int32_t        captured_batch_capacity) {
    // Replay path is small-batch metadata; one CUDA block avoids any host prefix-sum.
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int32_t page_offset        = 0;
    decode_page_indptr[0]      = 0;
    qo_indptr[0]               = 0;
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;

    for (int32_t batch = 0; batch < batch_size; ++batch) {
        const int32_t seq_len = sequence_lengths_plus_1[batch] > 1 ? sequence_lengths_plus_1[batch] : 1;
        const int32_t pages   = (seq_len + safe_page_sz - 1) / safe_page_sz;

        batch_indice[batch]           = batch;
        positions[batch]              = seq_len - 1;
        kvlen[batch]                  = seq_len;
        paged_kv_last_page_len[batch] = (seq_len - 1) % safe_page_sz + 1;
        const int32_t block_index     = (seq_len - 1) / safe_page_sz;
        const int32_t block_offset    = (seq_len - 1) % safe_page_sz;
        const int32_t block_number =
            block_index < max_blocks_per_batch ? block_ids[batch * max_blocks_per_batch + block_index] : 0;
        slot_mapping[batch] = static_cast<int64_t>(block_number) * safe_page_sz + static_cast<int64_t>(block_offset);

        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        for (int32_t page = 0; page < pages_to_copy; ++page) {
            page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
        }
        page_offset += pages_to_copy;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch + 1;
    }

    // Decode CUDA graph replay can use a graph captured for a larger batch
    // than the current live batch. Clear stale entries so the captured kernels
    // do not process phantom rows with old kvlen/page metadata and block_id=0.
    for (int32_t batch = batch_size; batch < captured_batch_capacity; ++batch) {
        batch_indice[batch]           = 0;
        positions[batch]              = 0;
        kvlen[batch]                  = 0;
        paged_kv_last_page_len[batch] = 0;
        slot_mapping[batch]           = -1;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch_size;
    }
}

// Generic prefill cuda graph metadata kernel. Used by both:
//   - target verify (SparseMla, with sparse-specific outputs)
//   - draft prefill (FlashInfer, sparse-specific outputs as nullptr)
// Pass nullptr for ks/ke/expanded_seq_lens/topk_indices_offset to skip those.
__global__ void prepareSparseMlaTargetVerifyParamsKernel(const int32_t* input_lengths,
                                                         const int32_t* prefix_lengths,
                                                         const int32_t* block_ids,
                                                         int32_t*       batch_indice,
                                                         int32_t*       page_indice,
                                                         int32_t*       decode_page_indptr,
                                                         int32_t*       paged_kv_last_page_len,
                                                         int32_t*       qo_indptr,
                                                         int32_t*       prefill_ragged_kv_len_indptr,
                                                         int32_t*       kvlen,
                                                         int32_t*       positions,
                                                         int64_t*       slot_mapping,
                                                         int32_t*       expanded_seq_lens,
                                                         int32_t*       topk_indices_offset,
                                                         int32_t*       ks,
                                                         int32_t*       ke,
                                                         int32_t        batch_size,
                                                         int32_t        max_blocks_per_batch,
                                                         int32_t        seq_size_per_block,
                                                         int32_t        captured_batch_capacity,
                                                         int32_t        captured_total_tokens) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;
    int32_t       token_offset = 0;
    int32_t       page_offset  = 0;
    int32_t       accu_kv_len  = 0;
    int32_t       k_offset     = 0;

    decode_page_indptr[0]           = 0;
    qo_indptr[0]                    = 0;
    prefill_ragged_kv_len_indptr[0] = 0;

    for (int32_t i = 0; i < batch_size; ++i) {
        const int32_t input_len  = input_lengths[i];
        const int32_t prefix_len = prefix_lengths[i];
        const int32_t kv_len     = input_len + prefix_len;

        for (int32_t j = 0; j < input_len; ++j) {
            const int32_t position      = j + prefix_len;
            batch_indice[token_offset]  = i;
            positions[token_offset]     = position;
            const int32_t seq_len_value = kv_len - input_len + 1 + j;
            if (expanded_seq_lens != nullptr) {
                expanded_seq_lens[token_offset] = seq_len_value;
            }
            if (topk_indices_offset != nullptr) {
                topk_indices_offset[token_offset] = 0;
            }
            if (ks != nullptr) {
                ks[token_offset] = k_offset;
            }
            if (ke != nullptr) {
                ke[token_offset] = k_offset + seq_len_value;
            }

            // slot_mapping: physical KV cache slot for this token
            const int32_t block_index  = position / safe_page_sz;
            const int32_t block_offset = position % safe_page_sz;
            const int32_t block_number =
                block_index < max_blocks_per_batch ? block_ids[i * max_blocks_per_batch + block_index] : 0;
            slot_mapping[token_offset] =
                static_cast<int64_t>(block_number) * safe_page_sz + static_cast<int64_t>(block_offset);

            token_offset++;
        }
        k_offset += kv_len;
        accu_kv_len += kv_len;

        kvlen[i]                    = kv_len;
        paged_kv_last_page_len[i]   = (kv_len - 1) % safe_page_sz + 1;
        const int32_t pages         = (kv_len + safe_page_sz - 1) / safe_page_sz;
        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        for (int32_t p = 0; p < pages_to_copy; ++p) {
            page_indice[page_offset + p] = block_ids[i * max_blocks_per_batch + p];
        }
        page_offset += pages_to_copy;

        decode_page_indptr[i + 1]           = page_offset;
        qo_indptr[i + 1]                    = token_offset;
        prefill_ragged_kv_len_indptr[i + 1] = accu_kv_len;
    }

    // Zero-fill stale entries beyond the active batch to prevent CUDA graph
    // replay from processing phantom batch elements with stale metadata.
    for (int32_t i = batch_size; i < captured_batch_capacity; ++i) {
        kvlen[i]                            = 0;
        paged_kv_last_page_len[i]           = 0;
        decode_page_indptr[i + 1]           = page_offset;
        qo_indptr[i + 1]                    = token_offset;
        prefill_ragged_kv_len_indptr[i + 1] = accu_kv_len;
    }
    for (int32_t t = token_offset; t < captured_total_tokens; ++t) {
        batch_indice[t] = 0;
        positions[t]    = 0;
        if (slot_mapping != nullptr)
            slot_mapping[t] = -1;
        if (expanded_seq_lens != nullptr)
            expanded_seq_lens[t] = 0;
        if (topk_indices_offset != nullptr)
            topk_indices_offset[t] = 0;
        if (ks != nullptr)
            ks[t] = 0;
        if (ke != nullptr)
            ke[t] = 0;
    }
}

// =====================================================================
// Optimized V1 decode kernel — single block + CUB BlockScan + warp-stride
// 62.7× mean speedup vs baseline (L20D bench). Used when batch_size <= 256.
// =====================================================================

constexpr int kDecodeV1Threads = 256;
constexpr int kWarpSize        = 32;

template<int BLOCK_THREADS>
__global__ void prepareFlashInferDecodeParamsKernelV1(const int32_t* __restrict__ sequence_lengths_plus_1,
                                                      const int32_t* __restrict__ block_ids,
                                                      int32_t* __restrict__ batch_indice,
                                                      int32_t* __restrict__ page_indice,
                                                      int32_t* __restrict__ decode_page_indptr,
                                                      int32_t* __restrict__ paged_kv_last_page_len,
                                                      int32_t* __restrict__ qo_indptr,
                                                      int32_t* __restrict__ kvlen,
                                                      int32_t* __restrict__ positions,
                                                      int64_t* __restrict__ slot_mapping,
                                                      int32_t batch_size,
                                                      int32_t max_blocks_per_batch,
                                                      int32_t seq_size_per_block,
                                                      int32_t captured_batch_capacity) {
    using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int32_t                         smem_seq_len[BLOCK_THREADS];
    __shared__ int32_t                         smem_pages[BLOCK_THREADS];
    __shared__ int32_t                         smem_page_prefix[BLOCK_THREADS];
    __shared__ int32_t                         smem_total_pages;

    const int     tid          = static_cast<int>(threadIdx.x);
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;

    int32_t seq_len = 0;
    int32_t pages   = 0;
    if (tid < batch_size) {
        const int32_t raw_seq   = sequence_lengths_plus_1[tid];
        seq_len                 = raw_seq > 1 ? raw_seq : 1;
        const int32_t raw_pages = (seq_len + safe_page_sz - 1) / safe_page_sz;
        pages                   = raw_pages < max_blocks_per_batch ? raw_pages : max_blocks_per_batch;
        smem_seq_len[tid]       = seq_len;
        smem_pages[tid]         = pages;
    } else {
        smem_seq_len[tid] = 0;
        smem_pages[tid]   = 0;
    }
    __syncthreads();

    int32_t page_prefix = 0;
    int32_t page_total  = 0;
    BlockScan(scan_storage).ExclusiveSum(pages, page_prefix, page_total);
    smem_page_prefix[tid] = page_prefix;
    if (tid == 0) {
        smem_total_pages = page_total;
    }
    __syncthreads();

    if (tid == 0) {
        decode_page_indptr[0] = 0;
        qo_indptr[0]          = 0;
    }
    if (tid < batch_size) {
        const int32_t cur_seq      = smem_seq_len[tid];
        const int32_t cur_pages    = smem_pages[tid];
        const int32_t pos          = cur_seq - 1;
        const int32_t block_index  = pos / safe_page_sz;
        const int32_t block_offset = pos % safe_page_sz;
        const int32_t block_number =
            block_index < max_blocks_per_batch ? block_ids[tid * max_blocks_per_batch + block_index] : 0;

        batch_indice[tid]           = tid;
        positions[tid]              = pos;
        kvlen[tid]                  = cur_seq;
        paged_kv_last_page_len[tid] = (cur_seq - 1) % safe_page_sz + 1;
        slot_mapping[tid] = static_cast<int64_t>(block_number) * safe_page_sz + static_cast<int64_t>(block_offset);
        decode_page_indptr[tid + 1] = page_prefix + cur_pages;
        qo_indptr[tid + 1]          = tid + 1;
    }

    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    const int n_warps = BLOCK_THREADS / kWarpSize;
    for (int b = warp_id; b < batch_size; b += n_warps) {
        const int32_t prefix = smem_page_prefix[b];
        const int32_t cnt    = smem_pages[b];
        for (int32_t p = lane_id; p < cnt; p += kWarpSize) {
            page_indice[prefix + p] = block_ids[b * max_blocks_per_batch + p];
        }
    }

    const int32_t total_pages = smem_total_pages;
    for (int b = batch_size + tid; b < captured_batch_capacity; b += BLOCK_THREADS) {
        batch_indice[b]           = 0;
        positions[b]              = 0;
        kvlen[b]                  = 0;
        paged_kv_last_page_len[b] = 0;
        slot_mapping[b]           = -1;
        decode_page_indptr[b + 1] = total_pages;
        qo_indptr[b + 1]          = batch_size;
    }
}

// =====================================================================
// Optimized V3 target-verify / prefill kernels — warp-shuffle cumsum
// + warp-per-batch writes. 40.9× mean speedup vs baseline (L20D bench).
// No batch_size limitation, no CUB dependency.
// =====================================================================

constexpr int kV3WarpSize       = 32;
constexpr int kV3WarpsPerBlock  = 8;
constexpr int kV3BlockThreads   = kV3WarpSize * kV3WarpsPerBlock;
constexpr int kV3CumsumWarpSize = 32;

__device__ __forceinline__ int32_t warpInclusiveScan(int32_t v) {
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < kV3CumsumWarpSize; offset <<= 1) {
        int32_t n = __shfl_up_sync(mask, v, offset);
        if (static_cast<int>(threadIdx.x) >= offset) {
            v += n;
        }
    }
    return v;
}

__global__ void prepareTargetVerifyCumsumKernel(const int32_t* __restrict__ input_lengths,
                                                const int32_t* __restrict__ prefix_lengths,
                                                int32_t* __restrict__ decode_page_indptr,
                                                int32_t* __restrict__ paged_kv_last_page_len,
                                                int32_t* __restrict__ qo_indptr,
                                                int32_t* __restrict__ prefill_ragged_kv_len_indptr,
                                                int32_t* __restrict__ kvlen,
                                                int32_t batch_size,
                                                int32_t safe_page_sz,
                                                int32_t max_blocks_per_batch,
                                                int32_t captured_batch_capacity) {
    if (blockIdx.x != 0 || threadIdx.x >= kV3CumsumWarpSize) {
        return;
    }

    if (threadIdx.x == 0) {
        decode_page_indptr[0]           = 0;
        qo_indptr[0]                    = 0;
        prefill_ragged_kv_len_indptr[0] = 0;
    }

    int32_t        running_token_offset = 0;
    int32_t        running_page_offset  = 0;
    int32_t        running_kv_offset    = 0;
    const unsigned full_mask            = 0xffffffffu;

    for (int32_t chunk_start = 0; chunk_start < batch_size; chunk_start += kV3CumsumWarpSize) {
        const int32_t lane = static_cast<int32_t>(threadIdx.x);
        const int32_t bid  = chunk_start + lane;
        const bool    live = bid < batch_size;

        int32_t input_len   = 0;
        int32_t kv_len      = 0;
        int32_t pages       = 0;
        int32_t last_pg_len = 0;
        if (live) {
            input_len               = input_lengths[bid];
            const int32_t pfx       = prefix_lengths[bid];
            kv_len                  = input_len + pfx;
            const int32_t pages_raw = (kv_len + safe_page_sz - 1) / safe_page_sz;
            pages                   = pages_raw < max_blocks_per_batch ? pages_raw : max_blocks_per_batch;
            last_pg_len             = (kv_len - 1) % safe_page_sz + 1;
        }

        const int32_t tok_inclusive = warpInclusiveScan(input_len);
        const int32_t pg_inclusive  = warpInclusiveScan(pages);
        const int32_t kv_inclusive  = warpInclusiveScan(kv_len);

        const int32_t tok_chunk_total = __shfl_sync(full_mask, tok_inclusive, kV3CumsumWarpSize - 1);
        const int32_t pg_chunk_total  = __shfl_sync(full_mask, pg_inclusive, kV3CumsumWarpSize - 1);
        const int32_t kv_chunk_total  = __shfl_sync(full_mask, kv_inclusive, kV3CumsumWarpSize - 1);

        if (live) {
            qo_indptr[bid + 1]                    = running_token_offset + tok_inclusive;
            decode_page_indptr[bid + 1]           = running_page_offset + pg_inclusive;
            prefill_ragged_kv_len_indptr[bid + 1] = running_kv_offset + kv_inclusive;
            kvlen[bid]                            = kv_len;
            paged_kv_last_page_len[bid]           = last_pg_len;
        }
        running_token_offset += tok_chunk_total;
        running_page_offset += pg_chunk_total;
        running_kv_offset += kv_chunk_total;
    }

    const int32_t tail_tokens = running_token_offset;
    const int32_t tail_pages  = running_page_offset;
    const int32_t tail_kv     = running_kv_offset;
    for (int32_t b = batch_size + static_cast<int32_t>(threadIdx.x); b < captured_batch_capacity;
         b += kV3CumsumWarpSize) {
        kvlen[b]                            = 0;
        paged_kv_last_page_len[b]           = 0;
        decode_page_indptr[b + 1]           = tail_pages;
        qo_indptr[b + 1]                    = tail_tokens;
        prefill_ragged_kv_len_indptr[b + 1] = tail_kv;
    }
}

__global__ void prepareTargetVerifyWriteKernel(const int32_t* __restrict__ input_lengths,
                                               const int32_t* __restrict__ prefix_lengths,
                                               const int32_t* __restrict__ block_ids,
                                               const int32_t* __restrict__ decode_page_indptr,
                                               const int32_t* __restrict__ qo_indptr,
                                               const int32_t* __restrict__ prefill_ragged_kv_len_indptr,
                                               int32_t* __restrict__ batch_indice,
                                               int32_t* __restrict__ page_indice,
                                               int32_t* __restrict__ positions,
                                               int64_t* __restrict__ slot_mapping,
                                               int32_t* __restrict__ expanded_seq_lens,
                                               int32_t* __restrict__ topk_indices_offset,
                                               int32_t* __restrict__ ks,
                                               int32_t* __restrict__ ke,
                                               int32_t batch_size,
                                               int32_t max_blocks_per_batch,
                                               int32_t safe_page_sz,
                                               int32_t captured_total_tokens) {
    const int32_t warp_id_in_block = static_cast<int32_t>(threadIdx.x) / kV3WarpSize;
    const int32_t lane             = static_cast<int32_t>(threadIdx.x) % kV3WarpSize;
    const int32_t global_warp_id   = static_cast<int32_t>(blockIdx.x) * kV3WarpsPerBlock + warp_id_in_block;
    const int32_t total_warps      = static_cast<int32_t>(gridDim.x) * kV3WarpsPerBlock;

    const bool has_sparse = (expanded_seq_lens != nullptr);

    for (int32_t bid = global_warp_id; bid < batch_size; bid += total_warps) {
        const int32_t input_len   = input_lengths[bid];
        const int32_t prefix_len  = prefix_lengths[bid];
        const int32_t kv_len      = input_len + prefix_len;
        const int32_t token_start = qo_indptr[bid];
        const int32_t page_start  = decode_page_indptr[bid];
        const int32_t k_off_base  = has_sparse ? prefill_ragged_kv_len_indptr[bid] : 0;

        const int32_t pages_raw     = (kv_len + safe_page_sz - 1) / safe_page_sz;
        const int32_t pages_to_copy = pages_raw < max_blocks_per_batch ? pages_raw : max_blocks_per_batch;

        for (int32_t j = lane; j < input_len; j += kV3WarpSize) {
            const int32_t t        = token_start + j;
            const int32_t position = j + prefix_len;
            batch_indice[t]        = bid;
            positions[t]           = position;

            const int32_t block_index  = position / safe_page_sz;
            const int32_t block_offset = position % safe_page_sz;
            const int32_t block_number =
                block_index < max_blocks_per_batch ? block_ids[bid * max_blocks_per_batch + block_index] : 0;
            slot_mapping[t] = static_cast<int64_t>(block_number) * static_cast<int64_t>(safe_page_sz)
                              + static_cast<int64_t>(block_offset);

            if (has_sparse) {
                const int32_t seq_len_value = kv_len - input_len + 1 + j;
                expanded_seq_lens[t]        = seq_len_value;
                topk_indices_offset[t]      = 0;
                ks[t]                       = k_off_base;
                ke[t]                       = k_off_base + seq_len_value;
            }
        }

        for (int32_t p = lane; p < pages_to_copy; p += kV3WarpSize) {
            page_indice[page_start + p] = block_ids[bid * max_blocks_per_batch + p];
        }
    }

    // Tail-fill stale token entries. Read live_total_tokens from qo_indptr[batch_size]
    // (written by cumsum kernel on the same stream — ordering guaranteed).
    const int32_t live_total_tokens = qo_indptr[batch_size];
    const int32_t global_tid = static_cast<int32_t>(blockIdx.x) * kV3BlockThreads + static_cast<int32_t>(threadIdx.x);
    const int32_t total_tids = static_cast<int32_t>(gridDim.x) * kV3BlockThreads;
    for (int32_t t = live_total_tokens + global_tid; t < captured_total_tokens; t += total_tids) {
        batch_indice[t] = 0;
        positions[t]    = 0;
        if (slot_mapping != nullptr) {
            slot_mapping[t] = -1;
        }
        if (has_sparse) {
            expanded_seq_lens[t]   = 0;
            topk_indices_offset[t] = 0;
            ks[t]                  = 0;
            ke[t]                  = 0;
        }
    }
}

}  // namespace

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, cudaStream_t stream) {
    TORCH_CHECK(params.region_count >= 0 && params.region_count <= kMaxCudaGraphPrepareFillRegions,
                "invalid cuda graph prepare fill region count: ",
                params.region_count);

    int64_t total_count = 0;
    for (int32_t i = 0; i < params.region_count; ++i) {
        total_count += params.regions[i].count > 0 ? params.regions[i].count : 0;
    }
    if (total_count <= 0) {
        return;
    }

    constexpr int block_size = 256;
    const int     blocks     = static_cast<int>(std::min<int64_t>((total_count + block_size - 1) / block_size, 1024));
    cudaGraphPrepareFillKernel<<<blocks, block_size, 0, stream>>>(params);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "cuda graph prepare fill kernel failed: ", cudaGetErrorString(result));
}

void invokePrepareFlashInferDecodeParams(const int32_t* sequence_lengths_plus_1,
                                         const int32_t* block_ids,
                                         int32_t*       batch_indice,
                                         int32_t*       page_indice,
                                         int32_t*       decode_page_indptr,
                                         int32_t*       paged_kv_last_page_len,
                                         int32_t*       qo_indptr,
                                         int32_t*       kvlen,
                                         int32_t*       positions,
                                         int64_t*       slot_mapping,
                                         int32_t        batch_size,
                                         int32_t        max_blocks_per_batch,
                                         int32_t        seq_size_per_block,
                                         int32_t        captured_batch_capacity,
                                         cudaStream_t   stream) {
    TORCH_CHECK(sequence_lengths_plus_1 != nullptr, "sequence_lengths_plus_1 is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(batch_indice != nullptr && page_indice != nullptr && decode_page_indptr != nullptr
                    && paged_kv_last_page_len != nullptr && qo_indptr != nullptr && kvlen != nullptr
                    && positions != nullptr && slot_mapping != nullptr,
                "FlashInfer decode metadata output buffer is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    if (batch_size <= kDecodeV1Threads) {
        prepareFlashInferDecodeParamsKernelV1<kDecodeV1Threads>
            <<<1, kDecodeV1Threads, 0, stream>>>(sequence_lengths_plus_1,
                                                 block_ids,
                                                 batch_indice,
                                                 page_indice,
                                                 decode_page_indptr,
                                                 paged_kv_last_page_len,
                                                 qo_indptr,
                                                 kvlen,
                                                 positions,
                                                 slot_mapping,
                                                 batch_size,
                                                 max_blocks_per_batch,
                                                 seq_size_per_block,
                                                 captured_batch_capacity);
    } else {
        prepareFlashInferDecodeParamsKernel<<<1, 1, 0, stream>>>(sequence_lengths_plus_1,
                                                                 block_ids,
                                                                 batch_indice,
                                                                 page_indice,
                                                                 decode_page_indptr,
                                                                 paged_kv_last_page_len,
                                                                 qo_indptr,
                                                                 kvlen,
                                                                 positions,
                                                                 slot_mapping,
                                                                 batch_size,
                                                                 max_blocks_per_batch,
                                                                 seq_size_per_block,
                                                                 captured_batch_capacity);
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
}

// Non-sparse prefill cuda graph kernel — sparse-specific outputs nullptr.
void invokePrepareFlashInferPrefillParams(const int32_t* input_lengths,
                                          const int32_t* prefix_lengths,
                                          const int32_t* block_ids,
                                          int32_t*       batch_indice,
                                          int32_t*       page_indice,
                                          int32_t*       decode_page_indptr,
                                          int32_t*       paged_kv_last_page_len,
                                          int32_t*       qo_indptr,
                                          int32_t*       prefill_ragged_kv_len_indptr,
                                          int32_t*       kvlen,
                                          int32_t*       positions,
                                          int64_t*       slot_mapping,
                                          int32_t        batch_size,
                                          int32_t        max_blocks_per_batch,
                                          int32_t        seq_size_per_block,
                                          int32_t        captured_total_tokens,
                                          cudaStream_t   stream) {
    TORCH_CHECK(input_lengths != nullptr, "input_lengths is null");
    TORCH_CHECK(prefix_lengths != nullptr, "prefix_lengths is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(slot_mapping != nullptr, "slot_mapping is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;
    prepareTargetVerifyCumsumKernel<<<1, kV3CumsumWarpSize, 0, stream>>>(input_lengths,
                                                                         prefix_lengths,
                                                                         decode_page_indptr,
                                                                         paged_kv_last_page_len,
                                                                         qo_indptr,
                                                                         prefill_ragged_kv_len_indptr,
                                                                         kvlen,
                                                                         batch_size,
                                                                         safe_page_sz,
                                                                         max_blocks_per_batch,
                                                                         /*captured_batch_capacity=*/batch_size);
    const int32_t write_blocks = std::max((batch_size + kV3WarpsPerBlock - 1) / kV3WarpsPerBlock, 1);
    prepareTargetVerifyWriteKernel<<<write_blocks, kV3BlockThreads, 0, stream>>>(input_lengths,
                                                                                 prefix_lengths,
                                                                                 block_ids,
                                                                                 decode_page_indptr,
                                                                                 qo_indptr,
                                                                                 prefill_ragged_kv_len_indptr,
                                                                                 batch_indice,
                                                                                 page_indice,
                                                                                 positions,
                                                                                 slot_mapping,
                                                                                 /*expanded_seq_lens=*/nullptr,
                                                                                 /*topk_indices_offset=*/nullptr,
                                                                                 /*ks=*/nullptr,
                                                                                 /*ke=*/nullptr,
                                                                                 batch_size,
                                                                                 max_blocks_per_batch,
                                                                                 safe_page_sz,
                                                                                 captured_total_tokens);
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer prefill CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
}

void invokePrepareSparseMlaTargetVerifyParams(const int32_t* input_lengths,
                                              const int32_t* prefix_lengths,
                                              const int32_t* block_ids,
                                              int32_t*       batch_indice,
                                              int32_t*       page_indice,
                                              int32_t*       decode_page_indptr,
                                              int32_t*       paged_kv_last_page_len,
                                              int32_t*       qo_indptr,
                                              int32_t*       prefill_ragged_kv_len_indptr,
                                              int32_t*       kvlen,
                                              int32_t*       positions,
                                              int64_t*       slot_mapping,
                                              int32_t*       expanded_seq_lens,
                                              int32_t*       topk_indices_offset,
                                              int32_t*       ks,
                                              int32_t*       ke,
                                              int32_t        batch_size,
                                              int32_t        max_blocks_per_batch,
                                              int32_t        seq_size_per_block,
                                              int32_t        captured_batch_capacity,
                                              int32_t        captured_total_tokens,
                                              cudaStream_t   stream) {
    TORCH_CHECK(input_lengths != nullptr, "input_lengths is null");
    TORCH_CHECK(prefix_lengths != nullptr, "prefix_lengths is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(slot_mapping != nullptr, "slot_mapping is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;
    prepareTargetVerifyCumsumKernel<<<1, kV3CumsumWarpSize, 0, stream>>>(input_lengths,
                                                                         prefix_lengths,
                                                                         decode_page_indptr,
                                                                         paged_kv_last_page_len,
                                                                         qo_indptr,
                                                                         prefill_ragged_kv_len_indptr,
                                                                         kvlen,
                                                                         batch_size,
                                                                         safe_page_sz,
                                                                         max_blocks_per_batch,
                                                                         captured_batch_capacity);
    const int32_t write_blocks = std::max((batch_size + kV3WarpsPerBlock - 1) / kV3WarpsPerBlock, 1);
    prepareTargetVerifyWriteKernel<<<write_blocks, kV3BlockThreads, 0, stream>>>(input_lengths,
                                                                                 prefix_lengths,
                                                                                 block_ids,
                                                                                 decode_page_indptr,
                                                                                 qo_indptr,
                                                                                 prefill_ragged_kv_len_indptr,
                                                                                 batch_indice,
                                                                                 page_indice,
                                                                                 positions,
                                                                                 slot_mapping,
                                                                                 expanded_seq_lens,
                                                                                 topk_indices_offset,
                                                                                 ks,
                                                                                 ke,
                                                                                 batch_size,
                                                                                 max_blocks_per_batch,
                                                                                 safe_page_sz,
                                                                                 captured_total_tokens);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "SparseMLA target verify CUDA graph prepare kernel failed: ",
                cudaGetErrorString(result));
}

}  // namespace rtp_llm
