#include "rtp_llm/cpp/kernels/cp_cache_scatter_kernel.h"
#include <cassert>

namespace rtp_llm {

/// Each thread block handles one virtual block.
///
/// Source (temp buffer) layout per virtual block:
///   cp_size consecutive regions of block_size tokens each.
///   Region p (peer p) slot s holds the token at global offset (s * cp_size + p)
///   within this virtual block's token range.
///
/// Destination (decode blocks) layout:
///   Contiguous blocks of block_size tokens each.
///   Decode block d (relative to virtual block start) holds tokens
///   [d * block_size .. (d+1) * block_size - 1].
///
/// The kernel reads from the temp buffer in interleaved order and writes to
/// decode blocks in contiguous order, using int4 (16-byte) granularity.
__global__ void cpCacheScatterKernel(void**      dst_block_addrs,
                                     const int*  dst_block_ids,
                                     const void* src_temp_buffer,
                                     int         cp_size,
                                     int         block_size,
                                     int         total_tokens,
                                     int         elem_stride_bytes) {
    const int vb = blockIdx.x;

    const int tokens_per_vb    = block_size * cp_size;
    const int elem_stride_int4 = elem_stride_bytes / 16;
    const int vb_token_start   = vb * tokens_per_vb;

    // Source: contiguous temp buffer slice for this virtual block
    const int4* src = reinterpret_cast<const int4*>(static_cast<const char*>(src_temp_buffer)
                                                    + (size_t)vb * tokens_per_vb * elem_stride_bytes);

    // For each token in this virtual block, compute source and destination positions.
    for (int t = 0; t < tokens_per_vb; t++) {
        int global_token = vb_token_start + t;
        if (global_token >= total_tokens) {
            break;
        }

        // Source position in temp buffer: peer p's slot s
        int peer         = t % cp_size;
        int slot_in_peer = t / cp_size;
        int src_offset   = (peer * block_size + slot_in_peer) * elem_stride_int4;

        // Destination: contiguous decode block
        int   dst_block_idx = global_token / block_size;
        int   dst_slot      = global_token % block_size;
        int   dst_block_id  = dst_block_ids[dst_block_idx];
        int4* dst           = reinterpret_cast<int4*>(dst_block_addrs[dst_block_id]);
        int   dst_offset    = dst_slot * elem_stride_int4;

        for (int e = threadIdx.x; e < elem_stride_int4; e += blockDim.x) {
            dst[dst_offset + e] = src[src_offset + e];
        }
    }
}

void invokeCPCacheScatter(void**       dst_block_addrs,
                          const int*   dst_block_ids,
                          const void*  src_temp_buffer,
                          int          virtual_block_count,
                          int          cp_size,
                          int          block_size,
                          int          total_tokens,
                          int          elem_stride_bytes,
                          cudaStream_t stream) {
    if (virtual_block_count <= 0 || cp_size <= 1 || total_tokens <= 0) {
        return;
    }
    assert(elem_stride_bytes % 16 == 0);

    const int threads_per_block = 256;
    cpCacheScatterKernel<<<virtual_block_count, threads_per_block, 0, stream>>>(
        dst_block_addrs, dst_block_ids, src_temp_buffer, cp_size, block_size, total_tokens, elem_stride_bytes);
}

}  // namespace rtp_llm
