#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

#include <algorithm>
#include <c10/util/Exception.h>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

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
        const int32_t fill_value =
            (region.value_ptr != nullptr ? *region.value_ptr : region.value) + region.value_offset;
        for (int64_t i = tid; i < region.count; i += stride) {
            region.ptr[i] = fill_value + region.step * i;
        }
    }
}

__global__ void prepareFlashInferDecodeParamsKernel(const int32_t* sequence_lengths_plus_1,
                                                    const int32_t* block_ids,
                                                    int32_t*       batch_indice,
                                                    int32_t*       decode_page_indptr,
                                                    int32_t*       paged_kv_last_page_len,
                                                    int32_t*       qo_indptr,
                                                    int32_t*       kvlen,
                                                    int32_t*       positions,
                                                    int64_t*       slot_mapping,
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block) {
    // Only the small per-request prefix sum is serial; page copies run separately.
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

        batch_indice[batch] = batch;
        positions[batch]    = seq_len - 1;
        if (slot_mapping != nullptr) {
            const int32_t position = seq_len - 1;
            const int32_t page     = position / safe_page_sz;
            const int32_t block    = page < max_blocks_per_batch ? block_ids[batch * max_blocks_per_batch + page] : 0;
            slot_mapping[batch]    = static_cast<int64_t>(block) * safe_page_sz + position % safe_page_sz;
        }
        kvlen[batch]                  = seq_len;
        paged_kv_last_page_len[batch] = (seq_len - 1) % safe_page_sz + 1;

        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        page_offset += pages_to_copy;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch + 1;
    }
}

__global__ void copyFlashInferDecodePagesKernel(const int32_t* block_ids,
                                                const int32_t* decode_page_indptr,
                                                int32_t*       page_indice,
                                                int32_t        max_blocks_per_batch) {
    const int32_t batch       = blockIdx.x;
    const int32_t page_offset = decode_page_indptr[batch];
    const int32_t page_count  = decode_page_indptr[batch + 1] - page_offset;
    for (int32_t page = threadIdx.x; page < page_count; page += blockDim.x) {
        page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
    }
}

#if !USING_ROCM
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

#endif

}  // namespace

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, CudaGraphPrepareStream stream) {
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
#if USING_ROCM
    const auto result = hipGetLastError();
    TORCH_CHECK(result == hipSuccess, "cuda graph prepare fill kernel failed: ", hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "cuda graph prepare fill kernel failed: ", cudaGetErrorString(result));
#endif
}

void invokePrepareFlashInferDecodeParams(const int32_t*         sequence_lengths_plus_1,
                                         const int32_t*         block_ids,
                                         int32_t*               batch_indice,
                                         int32_t*               page_indice,
                                         int32_t*               decode_page_indptr,
                                         int32_t*               paged_kv_last_page_len,
                                         int32_t*               qo_indptr,
                                         int32_t*               kvlen,
                                         int32_t*               positions,
                                         int32_t                batch_size,
                                         int32_t                max_blocks_per_batch,
                                         int32_t                seq_size_per_block,
                                         CudaGraphPrepareStream stream,
                                         int64_t*               slot_mapping) {
    TORCH_CHECK(sequence_lengths_plus_1 != nullptr, "sequence_lengths_plus_1 is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(batch_indice != nullptr && page_indice != nullptr && decode_page_indptr != nullptr
                    && paged_kv_last_page_len != nullptr && qo_indptr != nullptr && kvlen != nullptr
                    && positions != nullptr,
                "FlashInfer decode metadata output buffer is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    prepareFlashInferDecodeParamsKernel<<<1, 1, 0, stream>>>(sequence_lengths_plus_1,
                                                             block_ids,
                                                             batch_indice,
                                                             decode_page_indptr,
                                                             paged_kv_last_page_len,
                                                             qo_indptr,
                                                             kvlen,
                                                             positions,
                                                             slot_mapping,
                                                             batch_size,
                                                             max_blocks_per_batch,
                                                             seq_size_per_block);
    // The stream orders prefix generation before the coalesced per-request copies.
    copyFlashInferDecodePagesKernel<<<batch_size, 256, 0, stream>>>(
        block_ids, decode_page_indptr, page_indice, max_blocks_per_batch);
#if USING_ROCM
    const auto result = hipGetLastError();
    TORCH_CHECK(
        result == hipSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", hipGetErrorString(result));
#else
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
#endif
}

#if !USING_ROCM
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

#endif

}  // namespace rtp_llm
