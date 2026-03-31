#include "rtp_llm/cpp/kernels/cp_cache_scatter_kernel.h"
#include <cassert>

namespace rtp_llm {

// Alignment-safe per-token copy: uses the widest vector type the pointer
// alignment actually supports, falling back to byte copies if needed.
__device__ __forceinline__ void copyTokenData(void* __restrict__ dst, const void* __restrict__ src, int bytes) {
    auto d   = reinterpret_cast<char*>(dst);
    auto s   = reinterpret_cast<const char*>(src);
    int  off = 0;

    uintptr_t align = reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(s);

    if ((align & 15) == 0) {
        int  n   = bytes >> 4;
        auto d16 = reinterpret_cast<int4*>(d);
        auto s16 = reinterpret_cast<const int4*>(s);
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            d16[i] = s16[i];
        off = n << 4;
    } else if ((align & 7) == 0) {
        int  n  = bytes >> 3;
        auto d8 = reinterpret_cast<uint2*>(d);
        auto s8 = reinterpret_cast<const uint2*>(s);
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            d8[i] = s8[i];
        off = n << 3;
    } else if ((align & 3) == 0) {
        int  n  = bytes >> 2;
        auto d4 = reinterpret_cast<uint32_t*>(d);
        auto s4 = reinterpret_cast<const uint32_t*>(s);
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            d4[i] = s4[i];
        off = n << 2;
    } else {
        for (int i = threadIdx.x; i < bytes; i += blockDim.x)
            d[i] = s[i];
        return;
    }

    // Copy any remaining tail bytes (only thread 0).
    if (threadIdx.x == 0) {
        for (int i = off; i < bytes; ++i)
            d[i] = s[i];
    }
}

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
__global__ void cpCacheScatterKernel(void**      dst_block_addrs,
                                     const int*  dst_block_ids,
                                     const void* src_temp_buffer,
                                     int         cp_size,
                                     int         block_size,
                                     int         total_tokens,
                                     int         elem_stride_bytes) {
    const int vb = blockIdx.x;

    const int tokens_per_vb  = block_size * cp_size;
    const int vb_token_start = vb * tokens_per_vb;

    const char* src_base = static_cast<const char*>(src_temp_buffer) + (size_t)vb * tokens_per_vb * elem_stride_bytes;

    for (int t = 0; t < tokens_per_vb; t++) {
        int global_token = vb_token_start + t;
        if (global_token >= total_tokens)
            break;

        int peer         = t % cp_size;
        int slot_in_peer = t / cp_size;
        int src_byte_off = (peer * block_size + slot_in_peer) * elem_stride_bytes;

        int   dst_block_idx = global_token / block_size;
        int   dst_slot      = global_token % block_size;
        int   dst_block_id  = dst_block_ids[dst_block_idx];
        char* dst           = static_cast<char*>(dst_block_addrs[dst_block_id]);
        int   dst_byte_off  = dst_slot * elem_stride_bytes;

        copyTokenData(dst + dst_byte_off, src_base + src_byte_off, elem_stride_bytes);
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
    assert(elem_stride_bytes > 0);

    const int threads_per_block = 256;
    cpCacheScatterKernel<<<virtual_block_count, threads_per_block, 0, stream>>>(
        dst_block_addrs, dst_block_ids, src_temp_buffer, cp_size, block_size, total_tokens, elem_stride_bytes);
}

__global__ void cpCacheScatterPagedKernel(void**     dst_block_addrs,
                                          const int* dst_block_ids,
                                          void**     src_block_addrs,
                                          const int* src_block_ids,
                                          int        cp_size,
                                          int        block_size,
                                          int        total_tokens,
                                          int        elem_stride_bytes,
                                          int        addr_table_size) {
    const int vb = blockIdx.x;
    (void)addr_table_size;

    const int tokens_per_vb  = block_size * cp_size;
    const int vb_token_start = vb * tokens_per_vb;

    for (int t = 0; t < tokens_per_vb; t++) {
        int global_token = vb_token_start + t;
        if (global_token >= total_tokens)
            break;

        int peer         = t % cp_size;
        int slot_in_peer = t / cp_size;

        int         src_idx      = vb * cp_size + peer;
        int         src_bid      = src_block_ids[src_idx];
        const char* src          = static_cast<const char*>(src_block_addrs[src_bid]);
        int         src_byte_off = slot_in_peer * elem_stride_bytes;

        int   dst_block_idx = global_token / block_size;
        int   dst_slot      = global_token % block_size;
        int   dst_bid       = dst_block_ids[dst_block_idx];
        char* dst           = static_cast<char*>(dst_block_addrs[dst_bid]);
        int   dst_byte_off  = dst_slot * elem_stride_bytes;

        copyTokenData(dst + dst_byte_off, src + src_byte_off, elem_stride_bytes);
    }
}

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
                               cudaStream_t stream) {
    if (virtual_block_count <= 0 || cp_size <= 1 || total_tokens <= 0) {
        return;
    }
    assert(elem_stride_bytes > 0);

    const int threads_per_block = 256;
    cpCacheScatterPagedKernel<<<virtual_block_count, threads_per_block, 0, stream>>>(dst_block_addrs,
                                                                                     dst_block_ids,
                                                                                     src_block_addrs,
                                                                                     src_block_ids,
                                                                                     cp_size,
                                                                                     block_size,
                                                                                     total_tokens,
                                                                                     elem_stride_bytes,
                                                                                     addr_table_size);
}

}  // namespace rtp_llm
