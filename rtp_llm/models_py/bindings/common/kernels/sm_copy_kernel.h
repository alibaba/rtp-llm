#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

// 支持N卡 & PPU
// COPY FROM tair mempool (pace)，就不改成RTP的namespace了，保留原namespace
// RTP只需要gather和scatter操作
// Split KV: warmup_sm_copy_split_kernels on first use; launch_*_copy_split from SplitKvCacheCopy.
// Variable-length staged and CRC copies use CopyTileKernel.h.

namespace sDevMPS {
/**
 * @brief Scatter from contiguous src to num_dsts pairs (dst_kv_cache[i], dst_kv_scale[i]).
 * Src layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...]; stride = kv_cache_size + kv_scale_size per dst.
 */
void launch_scatter_copy_split(const void*  src,
                               void**       dst_kv_cache_ptrs,
                               void**       dst_kv_scale_ptrs,
                               size_t       kv_cache_size,
                               size_t       kv_scale_size,
                               int          num_dsts,
                               int          block_num,
                               cudaStream_t stream);

/**
 * @brief Gather from num_srcs pairs (src_kv_cache[i], src_kv_scale[i]) to contiguous dst.
 * Dst layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...].
 */
void launch_gather_copy_split(const void** src_kv_cache_ptrs,
                              const void** src_kv_scale_ptrs,
                              size_t       kv_cache_size,
                              size_t       kv_scale_size,
                              void*        dst,
                              int          num_srcs,
                              int          block_num,
                              cudaStream_t stream);

/**
 * @brief JIT-load split KV/scale gather+scatter kernels on \p stream (e.g. before NCCL init).
 * @return true on success; frees temp allocations before return.
 */
bool warmup_sm_copy_split_kernels(cudaStream_t stream);

}  // namespace sDevMPS
