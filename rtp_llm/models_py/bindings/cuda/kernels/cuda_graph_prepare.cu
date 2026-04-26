#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

#include <algorithm>
#include <c10/util/Exception.h>
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
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block) {
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

        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        for (int32_t page = 0; page < pages_to_copy; ++page) {
            page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
        }
        page_offset += pages_to_copy;
        decode_page_indptr[batch + 1] = page_offset;
        qo_indptr[batch + 1]          = batch + 1;
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
    prepareFlashInferDecodeParamsKernel<<<1, 1, 0, stream>>>(sequence_lengths_plus_1,
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

}  // namespace rtp_llm
