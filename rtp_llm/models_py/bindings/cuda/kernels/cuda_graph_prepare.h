#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace rtp_llm {

constexpr int kMaxCudaGraphPrepareFillRegions = 32;

struct CudaGraphPrepareFillRegion {
    int32_t*       ptr       = nullptr;
    int64_t        count     = 0;
    int32_t        value     = 0;
    const int32_t* value_ptr = nullptr;
};

struct CudaGraphPrepareFillParams {
    int32_t                    region_count = 0;
    CudaGraphPrepareFillRegion regions[kMaxCudaGraphPrepareFillRegions];
};

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, cudaStream_t stream);

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
                                         cudaStream_t   stream);

}  // namespace rtp_llm
