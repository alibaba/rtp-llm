#include "rtp_llm/cpp/kernels/cp_cache_scatter_kernel.h"
#include <cassert>

namespace rtp_llm {

/// Each thread block handles one virtual block.
/// Step 1: Gather cp_size interleaved physical blocks into temp buffer in token order.
/// Step 2: Scatter from temp buffer into cp_size contiguous decode blocks.
///
/// Token mapping within a virtual block:
///   For peer p, physical slot s (0..block_size-1):
///     global_token_offset = s * cp_size + p   (within the virtual block's token range)
///   This token should go to:
///     decode_block_idx = global_token_offset / block_size
///     decode_slot      = global_token_offset % block_size
///
/// We process at int4 (16-byte) granularity for coalesced memory access.
///
/// temp_buffer layout: [virtual_block_count, block_size * cp_size, elem_stride_bytes]
/// Each thread block uses its own slice: temp_buffer + blockIdx.x * tokens_per_vb * elem_stride_bytes
__global__ void cpCacheScatterKernel(void**     block_addrs,
                                     const int* block_ids,
                                     void*      temp_buffer,
                                     int        cp_size,
                                     int        block_size,
                                     int        elem_stride_bytes) {
    const int vb = blockIdx.x;  // virtual block index

    const int tokens_per_vb    = block_size * cp_size;
    const int elem_stride_int4 = elem_stride_bytes / 16;

    // Each VB gets its own temp slice
    int4* temp = reinterpret_cast<int4*>(
        static_cast<char*>(temp_buffer) + (size_t)vb * tokens_per_vb * elem_stride_bytes);

    // Step 1: Gather from interleaved physical blocks into temp buffer.
    // Physical block for peer p of virtual block vb is at block_ids[vb * cp_size + p].
    // Physical slot s in peer p maps to global token offset (s * cp_size + p).
    for (int p = 0; p < cp_size; p++) {
        int   phys_block_id   = block_ids[vb * cp_size + p];
        int4* phys_block_data = reinterpret_cast<int4*>(block_addrs[phys_block_id]);

        for (int s = 0; s < block_size; s++) {
            int global_token_offset = s * cp_size + p;
            for (int e = threadIdx.x; e < elem_stride_int4; e += blockDim.x) {
                temp[global_token_offset * elem_stride_int4 + e] = phys_block_data[s * elem_stride_int4 + e];
            }
        }
    }

    __syncthreads();

    // Step 2: Scatter from temp buffer to decode blocks in contiguous order.
    // Decode block d (relative to this VB) holds tokens [d*block_size .. (d+1)*block_size-1].
    for (int d = 0; d < cp_size; d++) {
        int   decode_block_id   = block_ids[vb * cp_size + d];
        int4* decode_block_data = reinterpret_cast<int4*>(block_addrs[decode_block_id]);

        for (int slot = 0; slot < block_size; slot++) {
            int src_token = d * block_size + slot;
            for (int e = threadIdx.x; e < elem_stride_int4; e += blockDim.x) {
                decode_block_data[slot * elem_stride_int4 + e] = temp[src_token * elem_stride_int4 + e];
            }
        }
    }
}

void invokeCPCacheScatter(void**       block_addrs,
                          const int*   block_ids,
                          void*        temp_buffer,
                          int          virtual_block_count,
                          int          cp_size,
                          int          block_size,
                          int          elem_stride_bytes,
                          cudaStream_t stream) {
    if (virtual_block_count <= 0 || cp_size <= 1) {
        return;
    }
    assert(elem_stride_bytes % 16 == 0);

    const int threads_per_block = 256;
    // temp_buffer must be at least virtual_block_count * block_size * cp_size * elem_stride_bytes
    cpCacheScatterKernel<<<virtual_block_count, threads_per_block, 0, stream>>>(
        block_addrs, block_ids, temp_buffer, cp_size, block_size, elem_stride_bytes);
}

}  // namespace rtp_llm
