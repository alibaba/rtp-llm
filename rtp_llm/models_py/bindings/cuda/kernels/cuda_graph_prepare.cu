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
                                                    int64_t*       slot_mapping,
                                                    int32_t        batch_size,
                                                    int32_t        max_blocks_per_batch,
                                                    int32_t        seq_size_per_block,
                                                    int32_t        captured_batch_capacity) {
    const int32_t batch = static_cast<int32_t>(blockIdx.x);
    if (batch >= captured_batch_capacity) {
        return;
    }

    const int32_t safe_page_sz = seq_size_per_block > 0 ? seq_size_per_block : 1;
    if (batch == 0 && threadIdx.x == 0) {
        decode_page_indptr[0] = 0;
        qo_indptr[0]          = 0;
    }

    if (batch < batch_size) {
        const int32_t seq_len = sequence_lengths_plus_1[batch] > 1 ? sequence_lengths_plus_1[batch] : 1;
        const int32_t pages   = (seq_len + safe_page_sz - 1) / safe_page_sz;

        int32_t page_offset = 0;
        for (int32_t prev_batch = 0; prev_batch < batch; ++prev_batch) {
            const int32_t prev_seq_len =
                sequence_lengths_plus_1[prev_batch] > 1 ? sequence_lengths_plus_1[prev_batch] : 1;
            const int32_t prev_pages = (prev_seq_len + safe_page_sz - 1) / safe_page_sz;
            page_offset += prev_pages < max_blocks_per_batch ? prev_pages : max_blocks_per_batch;
        }

        const int32_t pages_to_copy = pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        for (int32_t page = threadIdx.x; page < pages_to_copy; page += blockDim.x) {
            page_indice[page_offset + page] = block_ids[batch * max_blocks_per_batch + page];
        }

        if (threadIdx.x == 0) {
            batch_indice[batch]           = batch;
            positions[batch]              = seq_len - 1;
            kvlen[batch]                  = seq_len;
            paged_kv_last_page_len[batch] = (seq_len - 1) % safe_page_sz + 1;
            const int32_t block_index     = (seq_len - 1) / safe_page_sz;
            const int32_t block_offset    = (seq_len - 1) % safe_page_sz;
            const int32_t block_number =
                block_index < max_blocks_per_batch ? block_ids[batch * max_blocks_per_batch + block_index] : 0;
            slot_mapping[batch] =
                static_cast<int64_t>(block_number) * safe_page_sz + static_cast<int64_t>(block_offset);
            decode_page_indptr[batch + 1] = page_offset + pages_to_copy;
            qo_indptr[batch + 1]          = batch + 1;
        }
        return;
    }

    // Decode CUDA graph replay can use a graph captured for a larger batch
    // than the current live batch. Clear stale entries so the captured kernels
    // do not process phantom rows with old kvlen/page metadata and block_id=0.
    if (threadIdx.x == 0) {
        int32_t page_offset = 0;
        for (int32_t active_batch = 0; active_batch < batch_size; ++active_batch) {
            const int32_t seq_len =
                sequence_lengths_plus_1[active_batch] > 1 ? sequence_lengths_plus_1[active_batch] : 1;
            const int32_t pages = (seq_len + safe_page_sz - 1) / safe_page_sz;
            page_offset += pages < max_blocks_per_batch ? pages : max_blocks_per_batch;
        }
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
    constexpr int block_size = 128;
    prepareFlashInferDecodeParamsKernel<<<captured_batch_capacity, block_size, 0, stream>>>(sequence_lengths_plus_1,
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
    prepareSparseMlaTargetVerifyParamsKernel<<<1, 1, 0, stream>>>(input_lengths,
                                                                  prefix_lengths,
                                                                  block_ids,
                                                                  batch_indice,
                                                                  page_indice,
                                                                  decode_page_indptr,
                                                                  paged_kv_last_page_len,
                                                                  qo_indptr,
                                                                  prefill_ragged_kv_len_indptr,
                                                                  kvlen,
                                                                  positions,
                                                                  slot_mapping,
                                                                  /*expanded_seq_lens=*/nullptr,
                                                                  /*topk_indices_offset=*/nullptr,
                                                                  /*ks=*/nullptr,
                                                                  /*ke=*/nullptr,
                                                                  batch_size,
                                                                  max_blocks_per_batch,
                                                                  seq_size_per_block,
                                                                  batch_size,
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
    prepareSparseMlaTargetVerifyParamsKernel<<<1, 1, 0, stream>>>(input_lengths,
                                                                  prefix_lengths,
                                                                  block_ids,
                                                                  batch_indice,
                                                                  page_indice,
                                                                  decode_page_indptr,
                                                                  paged_kv_last_page_len,
                                                                  qo_indptr,
                                                                  prefill_ragged_kv_len_indptr,
                                                                  kvlen,
                                                                  positions,
                                                                  slot_mapping,
                                                                  expanded_seq_lens,
                                                                  topk_indices_offset,
                                                                  ks,
                                                                  ke,
                                                                  batch_size,
                                                                  max_blocks_per_batch,
                                                                  seq_size_per_block,
                                                                  captured_batch_capacity,
                                                                  captured_total_tokens);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "SparseMLA target verify CUDA graph prepare kernel failed: ",
                cudaGetErrorString(result));
}

}  // namespace rtp_llm
