#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

#include <algorithm>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

namespace rtp_llm {

namespace {

constexpr int32_t kMaxParallelDecodeBatch = 1024;

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
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block) {
    // Keep the original serial behavior as a correctness fallback for an
    // unusually large batch. Normal decode batches use one thread per request
    // for metadata and one warp per request for coalesced page-table copies.
    if (batch_size > kMaxParallelDecodeBatch) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        int32_t       page_offset = 0;
        const int32_t page_size   = seq_size_per_block > 0 ? seq_size_per_block : 1;
        decode_page_indptr[0]     = 0;
        qo_indptr[0]              = 0;
        for (int32_t batch = 0; batch < batch_size; ++batch) {
            const int32_t seq_len       = sequence_lengths_plus_1[batch] > 1 ? sequence_lengths_plus_1[batch] : 1;
            const int32_t pages         = (seq_len + page_size - 1) / page_size;
            const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;

            batch_indice[batch]           = batch;
            positions[batch]              = seq_len - 1;
            kvlen[batch]                  = seq_len;
            paged_kv_last_page_len[batch] = (seq_len - 1) % page_size + 1;
            for (int32_t page = 0; page < pages_to_copy; ++page) {
                page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
            }
            page_offset += pages_to_copy;
            decode_page_indptr[batch + 1] = page_offset;
            qo_indptr[batch + 1]          = batch + 1;
        }
        return;
    }

    __shared__ int32_t pages_per_batch[kMaxParallelDecodeBatch];
    __shared__ int32_t page_offsets[kMaxParallelDecodeBatch];

    const int32_t batch     = threadIdx.x;
    const int32_t page_size = seq_size_per_block > 0 ? seq_size_per_block : 1;
    int32_t       seq_len   = 1;
    if (batch < batch_size) {
        seq_len                = sequence_lengths_plus_1[batch] > 1 ? sequence_lengths_plus_1[batch] : 1;
        const int32_t pages    = (seq_len + page_size - 1) / page_size;
        pages_per_batch[batch] = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
    }
    __syncthreads();

    if (batch == 0) {
        int32_t page_offset   = 0;
        decode_page_indptr[0] = 0;
        qo_indptr[0]          = 0;
        for (int32_t i = 0; i < batch_size; ++i) {
            page_offsets[i] = page_offset;
            page_offset += pages_per_batch[i];
            decode_page_indptr[i + 1] = page_offset;
        }
    }
    __syncthreads();

    if (batch < batch_size) {
        batch_indice[batch]           = batch;
        positions[batch]              = seq_len - 1;
        kvlen[batch]                  = seq_len;
        paged_kv_last_page_len[batch] = (seq_len - 1) % page_size + 1;
        qo_indptr[batch + 1]          = batch + 1;
    }

    const int32_t warp_id    = threadIdx.x / 32;
    const int32_t lane_id    = threadIdx.x % 32;
    const int32_t warp_count = blockDim.x / 32;
    for (int32_t request = warp_id; request < batch_size; request += warp_count) {
        const int32_t src_offset = request * max_blocks_per_batch;
        const int32_t dst_offset = page_offsets[request];
        for (int32_t page = lane_id; page < pages_per_batch[request]; page += 32) {
            page_indice[dst_offset + page] = block_ids[src_offset + page];
        }
    }
}

__global__ void prepareTokenSpeedMlaFromCompactKernel(const int32_t* page_indices,
                                                      const int32_t* page_indptr,
                                                      const int32_t* sequence_lengths,
                                                      int32_t*       block_tables,
                                                      int32_t*       output_sequence_lengths,
                                                      int32_t        batch_size,
                                                      int32_t        padded_blocks) {
    const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t total = batch_size * padded_blocks;
    if (index >= total) {
        return;
    }
    const int32_t batch      = index / padded_blocks;
    const int32_t column     = index - batch * padded_blocks;
    const int32_t page_begin = page_indptr[batch];
    const int32_t live_pages = page_indptr[batch + 1] - page_begin;
    // Threads write independent table cells. Column zero also owns the only
    // sequence-length write for its batch, so no block-wide synchronization is needed.
    block_tables[index] = column < live_pages ? page_indices[page_begin + column] : 0;
    if (column == 0) {
        output_sequence_lengths[batch] = sequence_lengths[batch];
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
                                         int32_t        batch_size,
                                         int32_t        max_blocks_per_batch,
                                         int32_t        seq_size_per_block,
                                         cudaStream_t   stream) {
    TORCH_CHECK(sequence_lengths_plus_1 != nullptr, "sequence_lengths_plus_1 is null");
    TORCH_CHECK(block_ids != nullptr, "block_ids is null");
    TORCH_CHECK(batch_indice != nullptr && page_indice != nullptr && decode_page_indptr != nullptr
                    && paged_kv_last_page_len != nullptr && qo_indptr != nullptr && kvlen != nullptr
                    && positions != nullptr,
                "FlashInfer decode metadata output buffer is null");
    if (batch_size <= 0 || max_blocks_per_batch <= 0) {
        return;
    }
    const int32_t threads =
        batch_size <= kMaxParallelDecodeBatch ? std::min(kMaxParallelDecodeBatch, std::max(32, batch_size * 32)) : 1;
    prepareFlashInferDecodeParamsKernel<<<1, threads, 0, stream>>>(sequence_lengths_plus_1,
                                                                   block_ids,
                                                                   batch_indice,
                                                                   page_indice,
                                                                   decode_page_indptr,
                                                                   paged_kv_last_page_len,
                                                                   qo_indptr,
                                                                   kvlen,
                                                                   positions,
                                                                   batch_size,
                                                                   max_blocks_per_batch,
                                                                   seq_size_per_block);
    const auto result = cudaGetLastError();
    TORCH_CHECK(
        result == cudaSuccess, "FlashInfer decode CUDA graph prepare kernel failed: ", cudaGetErrorString(result));
}

void invokePrepareTokenSpeedMlaFromCompact(const int32_t* page_indices,
                                           const int32_t* page_indptr,
                                           const int32_t* sequence_lengths,
                                           int32_t*       block_tables,
                                           int32_t*       output_sequence_lengths,
                                           int32_t        batch_size,
                                           int32_t        padded_blocks,
                                           cudaStream_t   stream) {
    TORCH_CHECK(page_indices != nullptr && page_indptr != nullptr && sequence_lengths != nullptr,
                "TokenSpeed compact metadata input is null");
    TORCH_CHECK(block_tables != nullptr && output_sequence_lengths != nullptr,
                "TokenSpeed compact metadata output is null");
    constexpr int threads = 256;
    const int     blocks  = (batch_size * padded_blocks + threads - 1) / threads;
    prepareTokenSpeedMlaFromCompactKernel<<<blocks, threads, 0, stream>>>(
        page_indices, page_indptr, sequence_lengths, block_tables, output_sequence_lengths, batch_size, padded_blocks);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "TokenSpeed compact metadata kernel failed: ", cudaGetErrorString(result));
}

}  // namespace rtp_llm
