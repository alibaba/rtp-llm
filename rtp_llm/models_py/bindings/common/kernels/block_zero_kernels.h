#pragma once

#include <cstddef>
#include <cstdint>

#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

enum class BlockZeroVariant : int {
    kLayerFused      = 0,   // grid=(bs),            each TB loops all layers
    kLayerFusedCG    = 1,   // grid=(bs),            + cache-bypass stores
    kLayerParallel   = 2,   // grid=(bs,chunks),     layers split into chunks
    kLayerParallelCG = 3,   // grid=(bs,chunks),     + cache-bypass stores
    kLayerPerBlock   = 4,   // grid=(bs,layer_num),  one TB per (batch,layer)
};

/// Zero the latest incomplete KV cache block for each (batch, layer) pair.
///
/// A block is considered "incomplete" when `(total_tokens - 1) % seq_size_per_block == 0`,
/// i.e. the current token is the first to land in a fresh block.  Mid-block positions are
/// skipped with a single integer modulo — no warp divergence since all threads in a block
/// share the same batch_idx.
void invokeZeroIncompleteKvCacheBlocks(const void* const* layer_base_addrs,
                                       const int32_t*     kv_cache_block_id,
                                       const int32_t*     token_counts,
                                       const int32_t*     layer_to_group,
                                       size_t             batch_size,
                                       size_t             layer_num,
                                       size_t             batch_dim,
                                       size_t             max_blocks_per_batch,
                                       size_t             block_stride_bytes,
                                       size_t             seq_size_per_block,
                                       cudaStream_t       stream);

/// Variant-selectable version for benchmarking.
void invokeZeroIncompleteKvCacheBlocksVariant(
    const void* const* layer_base_addrs,
    const int32_t*     kv_cache_block_id,
    const int32_t*     token_counts,
    const int32_t*     layer_to_group,
    size_t             batch_size,
    size_t             layer_num,
    size_t             batch_dim,
    size_t             max_blocks_per_batch,
    size_t             block_stride_bytes,
    size_t             seq_size_per_block,
    cudaStream_t       stream,
    BlockZeroVariant   variant);

}  // namespace rtp_llm
