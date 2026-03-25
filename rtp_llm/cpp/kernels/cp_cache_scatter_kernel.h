#pragma once

#include <cstdint>

#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

/// Scatter interleaved CP-sharded KV cache blocks into contiguous decode-layout blocks.
///
/// Prefill with CP writes tokens in round-robin interleaved order within each
/// virtual block.  For virtual block v, peer p's physical block contains tokens
/// at positions: v * block_size * cp_size + slot * cp_size + p
/// (slot = 0..block_size-1).
///
/// Decode expects contiguous token order within standard blocks.  This kernel
/// merges cp_size interleaved physical blocks (one per peer) into cp_size
/// contiguous standard blocks.
///
/// The kernel processes one virtual block at a time using a temporary buffer
/// of size (block_size * cp_size * elem_stride_bytes).  For each virtual block:
///   1. Copy all cp_size physical blocks into the temp buffer in interleaved order
///   2. Scatter from temp buffer back to the cp_size decode blocks in contiguous order
///
/// @param block_addrs       Array of block base addresses, indexed by block_id.
///                          block_addrs[block_id] points to the start of that block's data.
/// @param block_ids         Decode-side block IDs for this request, length = total_decode_blocks.
///                          For virtual block v, the cp_size decode blocks are at
///                          block_ids[v * cp_size + 0..cp_size-1].
/// @param temp_buffer       Temporary GPU buffer of size >= (virtual_block_count * block_size * cp_size * elem_stride_bytes).
/// @param virtual_block_count Number of virtual blocks.
/// @param cp_size           Number of CP peers (= number of physical blocks per virtual block).
/// @param block_size        Tokens per physical block.
/// @param elem_stride_bytes Bytes per token in the KV cache (e.g., compressed_kv_dim * sizeof(dtype)).
/// @param stream            CUDA stream.
void invokeCPCacheScatter(void**       block_addrs,
                          const int*   block_ids,
                          void*        temp_buffer,
                          int          virtual_block_count,
                          int          cp_size,
                          int          block_size,
                          int          elem_stride_bytes,
                          cudaStream_t stream);

}  // namespace rtp_llm
