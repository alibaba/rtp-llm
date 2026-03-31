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

/// Scatter round-robin interleaved tokens from a contiguous temp buffer into
/// paged decode KV cache blocks.
///
/// During PD-separated CP prefill, each prefill rank writes its shard of a
/// virtual block into a contiguous temp buffer via RDMA.  The temp buffer
/// layout per virtual block is cp_size consecutive physical-block-sized
/// regions (one per peer), each containing block_size tokens in the peer's
/// interleaved order.
///
/// This kernel reassembles the tokens into contiguous decode-layout blocks:
///   For peer p, physical slot s (0..block_size-1) in the temp buffer:
///     global_token_offset = s * cp_size + p
///   This token goes to:
///     decode_block_idx = global_token_offset / block_size
///     decode_slot      = global_token_offset % block_size
///
/// @param dst_block_addrs   Array of decode block base addresses indexed by block_id.
/// @param dst_block_ids     Decode-side block IDs, length = ceil(total_tokens / block_size).
///                          dst_block_ids[i] is the physical block id for the i-th
///                          contiguous decode block of this request.
/// @param src_temp_buffer   Contiguous GPU buffer holding RDMA-received data.
///                          Layout: [virtual_block_count, cp_size, block_size, elem_stride_bytes].
///                          For virtual block v, peer p's data starts at offset
///                          (v * cp_size + p) * block_size * elem_stride_bytes.
/// @param virtual_block_count Number of virtual blocks.
/// @param cp_size           Number of CP peers.
/// @param block_size        Tokens per physical block.
/// @param total_tokens      Actual total token count (may be < virtual_block_count * cp_size * block_size).
/// @param elem_stride_bytes Bytes per token in the KV cache.
/// @param stream            CUDA stream.
void invokeCPCacheScatter(void**       dst_block_addrs,
                          const int*   dst_block_ids,
                          const void*  src_temp_buffer,
                          int          virtual_block_count,
                          int          cp_size,
                          int          block_size,
                          int          total_tokens,
                          int          elem_stride_bytes,
                          cudaStream_t stream);

/// Paged variant: both source (staging) and destination (decode) are paged blocks.
///
/// Used when staging blocks are borrowed from BlockPool instead of a contiguous
/// temp buffer.  Source data layout within each staging block is identical to
/// the contiguous version: peer p's slot s holds the token at
/// global offset (s * cp_size + p) within the virtual block.
///
/// @param dst_block_addrs  Array of decode block base addresses indexed by block_id.
/// @param dst_block_ids    Decode-side block IDs, length = ceil(total_tokens / block_size).
/// @param src_block_addrs  Array of staging block base addresses indexed by block_id.
/// @param src_block_ids    Staging block IDs, length = virtual_block_count * cp_size.
///                         Layout: [vblock_0_peer_0, vblock_0_peer_1, ..., vblock_1_peer_0, ...].
/// @param virtual_block_count Number of virtual blocks.
/// @param cp_size           Number of CP peers.
/// @param block_size        Tokens per physical block.
/// @param total_tokens      Actual total token count.
/// @param elem_stride_bytes Bytes per token in the KV cache.
/// @param stream            CUDA stream.
void invokeCPCacheScatterPaged(void**       dst_block_addrs,
                               const int*   dst_block_ids,
                               void**       src_block_addrs,
                               const int*   src_block_ids,
                               int          virtual_block_count,
                               int          cp_size,
                               int          block_size,
                               int          total_tokens,
                               int          elem_stride_bytes,
                               int          addr_table_size,
                               cudaStream_t stream);

/// Paged variant for indexer K cache stored in kv_scale with packed layout:
/// [all token fp8 K][all token fp32 scales], i.e. scales live at block tail
/// instead of being interleaved per token.
void invokeCPCacheScatterPagedPackedScale(void**       dst_block_addrs,
                                          const int*   dst_block_ids,
                                          void**       src_block_addrs,
                                          const int*   src_block_ids,
                                          int          virtual_block_count,
                                          int          cp_size,
                                          int          block_size,
                                          int          total_tokens,
                                          int          quant_bytes_per_token,
                                          int          scale_bytes_per_token,
                                          int          addr_table_size,
                                          cudaStream_t stream);

}  // namespace rtp_llm
