#pragma once

#include <cstdint>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

namespace rtp_llm {

#if USING_ROCM
using CudaGraphPrepareStream = hipStream_t;
#else
using CudaGraphPrepareStream = cudaStream_t;
#endif

constexpr int kMaxCudaGraphPrepareFillRegions = 32;

struct CudaGraphPrepareFillRegion {
    int32_t*       ptr       = nullptr;
    const int32_t* value_ptr = nullptr;
    int64_t        count     = 0;
    int32_t        value     = 0;
    int32_t        value_offset = 0;
    int32_t        step         = 0;
};

struct CudaGraphPrepareFillParams {
    int32_t                    region_count = 0;
    CudaGraphPrepareFillRegion regions[kMaxCudaGraphPrepareFillRegions];
    uint8_t*                   zero_bytes                     = nullptr;
    int64_t                    zero_byte_count                = 0;
    bool*                      valid_token_mask               = nullptr;
    const bool*                source_valid_token_mask        = nullptr;
    int64_t                    mask_count                     = 0;
    int64_t                    source_mask_count              = 0;
};

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, CudaGraphPrepareStream stream);

struct SequenceParallelPadRegion {
    const uint8_t* source                  = nullptr;
    uint8_t*       destination             = nullptr;
    int64_t        source_group_bytes      = 0;
    int64_t        destination_group_bytes = 0;
    int64_t        groups                  = 1;
    uint64_t       fill_bits               = 0;
    int32_t        element_bytes           = 0;
};

struct SequenceParallelPadParams {
    int32_t                   region_count = 0;
    SequenceParallelPadRegion regions[kMaxCudaGraphPrepareFillRegions];
};

void invokeSequenceParallelPad(SequenceParallelPadParams params, CudaGraphPrepareStream stream);

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
                                         CudaGraphPrepareStream stream);

}  // namespace rtp_llm
