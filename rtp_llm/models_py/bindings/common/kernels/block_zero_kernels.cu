#include "rtp_llm/models_py/bindings/common/kernels/block_zero_kernels.h"

#include <algorithm>

namespace rtp_llm {

// ─── Shared helpers ─────────────────────────────────────────────────────────

template <bool CacheBypass>
__device__ __forceinline__ void zero_block_region(char* dst, size_t block_stride_bytes) {
    const size_t n_uint4   = block_stride_bytes / sizeof(uint4);
    const size_t remainder = block_stride_bytes % sizeof(uint4);
    const uint4  zero4     = make_uint4(0u, 0u, 0u, 0u);

    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (size_t i = threadIdx.x; i < n_uint4; i += blockDim.x) {
        if constexpr (CacheBypass) {
#if USING_CUDA
            asm volatile(
                "st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(dst4 + i), "r"(zero4.x), "r"(zero4.y), "r"(zero4.z), "r"(zero4.w)
                : "memory");
#else
            dst4[i] = zero4;
#endif
        } else {
            dst4[i] = zero4;
        }
    }

    if (remainder > 0) {
        const size_t tail_start = n_uint4 * sizeof(uint4);
        for (size_t i = threadIdx.x; i < remainder; i += blockDim.x) {
            dst[tail_start + i] = 0;
        }
    }
}

__device__ __forceinline__ bool batch_needs_zero(
    const int32_t* __restrict__ token_counts,
    size_t batch_idx, size_t batch_size, size_t seq_size_per_block,
    size_t max_blocks_per_batch, size_t& last_block_index_out)
{
    if (batch_idx >= batch_size)
        return false;
    const int32_t tokens = token_counts[batch_idx];
    if (tokens <= 0)
        return false;
    if ((tokens - 1) % static_cast<int32_t>(seq_size_per_block) != 0)
        return false;
    last_block_index_out = static_cast<size_t>(tokens - 1) / seq_size_per_block;
    return last_block_index_out < max_blocks_per_batch;
}

__device__ __forceinline__ char* get_block_dst(
    const void* const* __restrict__ layer_base_addrs,
    const int32_t* __restrict__ kv_cache_block_id,
    const int32_t* __restrict__ layer_to_group,
    size_t layer_idx, size_t batch_idx, size_t batch_dim,
    size_t max_blocks_per_batch, size_t last_block_index,
    size_t block_stride_bytes)
{
    const void* base = layer_base_addrs[layer_idx];
    if (!base)
        return nullptr;

    const size_t group_idx = layer_to_group
        ? static_cast<size_t>(layer_to_group[layer_idx]) : 0;

    const int32_t block_id =
        kv_cache_block_id[group_idx * batch_dim * max_blocks_per_batch
                          + batch_idx * max_blocks_per_batch
                          + last_block_index];
    if (block_id <= 0)  // block 0 is reserved (never allocated); -1 is NULL_BLOCK_IDX
        return nullptr;

    return static_cast<char*>(const_cast<void*>(base))
           + static_cast<size_t>(block_id) * block_stride_bytes;
}

// ─── V0/V1: Layer-fused kernel ─────────────────────────────────────────────
// grid = (batch_size).  Each TB loops over all layers serially.

template <bool CacheBypass>
__global__ void __launch_bounds__(1024)
block_zero_layer_fused_kernel(
    const void* const* __restrict__ layer_base_addrs,
    const int32_t* __restrict__ kv_cache_block_id,
    const int32_t* __restrict__ token_counts,
    const int32_t* __restrict__ layer_to_group,
    size_t batch_size, size_t layer_num, size_t batch_dim,
    size_t max_blocks_per_batch, size_t block_stride_bytes,
    size_t seq_size_per_block)
{
    size_t last_block_index;
    if (!batch_needs_zero(token_counts, blockIdx.x, batch_size,
                          seq_size_per_block, max_blocks_per_batch, last_block_index))
        return;

    for (size_t l = 0; l < layer_num; ++l) {
        char* dst = get_block_dst(layer_base_addrs, kv_cache_block_id, layer_to_group,
                                  l, blockIdx.x, batch_dim, max_blocks_per_batch,
                                  last_block_index, block_stride_bytes);
        if (dst)
            zero_block_region<CacheBypass>(dst, block_stride_bytes);
    }
}

// ─── V2/V3: Layer-parallel kernel ──────────────────────────────────────────
// grid = (batch_size, layer_chunks).  Each TB handles a slice of layers.

template <bool CacheBypass>
__global__ void __launch_bounds__(1024)
block_zero_layer_parallel_kernel(
    const void* const* __restrict__ layer_base_addrs,
    const int32_t* __restrict__ kv_cache_block_id,
    const int32_t* __restrict__ token_counts,
    const int32_t* __restrict__ layer_to_group,
    size_t batch_size, size_t layer_num, size_t batch_dim,
    size_t max_blocks_per_batch, size_t block_stride_bytes,
    size_t seq_size_per_block, size_t layers_per_chunk)
{
    size_t last_block_index;
    if (!batch_needs_zero(token_counts, blockIdx.x, batch_size,
                          seq_size_per_block, max_blocks_per_batch, last_block_index))
        return;

    const size_t layer_start = blockIdx.y * layers_per_chunk;
    const size_t layer_end   = min(layer_start + layers_per_chunk, layer_num);

    for (size_t l = layer_start; l < layer_end; ++l) {
        char* dst = get_block_dst(layer_base_addrs, kv_cache_block_id, layer_to_group,
                                  l, blockIdx.x, batch_dim, max_blocks_per_batch,
                                  last_block_index, block_stride_bytes);
        if (dst)
            zero_block_region<CacheBypass>(dst, block_stride_bytes);
    }
}

// ─── V4: One TB per (batch, layer) ──────────────────────────────────────────
// grid = (batch_size, layer_num).  Maximum SM utilization for small batches.

__global__ void __launch_bounds__(256)
block_zero_per_layer_kernel(
    const void* const* __restrict__ layer_base_addrs,
    const int32_t* __restrict__ kv_cache_block_id,
    const int32_t* __restrict__ token_counts,
    const int32_t* __restrict__ layer_to_group,
    size_t batch_size, size_t layer_num, size_t batch_dim,
    size_t max_blocks_per_batch, size_t block_stride_bytes,
    size_t seq_size_per_block)
{
    size_t last_block_index;
    if (!batch_needs_zero(token_counts, blockIdx.x, batch_size,
                          seq_size_per_block, max_blocks_per_batch, last_block_index))
        return;

    const size_t l = blockIdx.y;
    if (l >= layer_num)
        return;

    char* dst = get_block_dst(layer_base_addrs, kv_cache_block_id, layer_to_group,
                              l, blockIdx.x, batch_dim, max_blocks_per_batch,
                              last_block_index, block_stride_bytes);
    if (dst)
        zero_block_region<false>(dst, block_stride_bytes);
}

// ─── Dispatch ───────────────────────────────────────────────────────────────

static int pickThreads(size_t block_stride_bytes) {
    const size_t n_uint4 = block_stride_bytes / sizeof(uint4);
    if (n_uint4 > 256)  return 1024;
    if (n_uint4 <= 128)  return 128;
    return 256;
}

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
    BlockZeroVariant   variant)
{
    if (batch_size == 0 || layer_num == 0 || block_stride_bytes == 0)
        return;

    const int threads = pickThreads(block_stride_bytes);

#define LAUNCH_FUSED(CB)                                                       \
    block_zero_layer_fused_kernel<CB>                                           \
        <<<dim3((unsigned)batch_size), dim3(threads), 0, stream>>>(            \
            layer_base_addrs, kv_cache_block_id, token_counts, layer_to_group, \
            batch_size, layer_num, batch_dim, max_blocks_per_batch,            \
            block_stride_bytes, seq_size_per_block)

#define LAUNCH_PARALLEL(CB)                                                    \
    do {                                                                        \
        constexpr size_t kTargetTBs = 512;                                     \
        size_t layer_chunks = std::max(size_t(1),                              \
            std::min(layer_num, (kTargetTBs + batch_size - 1) / batch_size));  \
        size_t layers_per_chunk = (layer_num + layer_chunks - 1) / layer_chunks;\
        layer_chunks = (layer_num + layers_per_chunk - 1) / layers_per_chunk;  \
        block_zero_layer_parallel_kernel<CB>                                    \
            <<<dim3((unsigned)batch_size, (unsigned)layer_chunks),              \
               dim3(threads), 0, stream>>>(                                    \
                layer_base_addrs, kv_cache_block_id, token_counts,             \
                layer_to_group, batch_size, layer_num, batch_dim,              \
                max_blocks_per_batch, block_stride_bytes, seq_size_per_block,  \
                layers_per_chunk);                                             \
    } while (0)

    switch (variant) {
        case BlockZeroVariant::kLayerFused:       LAUNCH_FUSED(false);    break;
        case BlockZeroVariant::kLayerFusedCG:     LAUNCH_FUSED(true);     break;
        case BlockZeroVariant::kLayerParallel:    LAUNCH_PARALLEL(false); break;
        case BlockZeroVariant::kLayerParallelCG:  LAUNCH_PARALLEL(true);  break;
        case BlockZeroVariant::kLayerPerBlock:
            block_zero_per_layer_kernel
                <<<dim3((unsigned)batch_size, (unsigned)layer_num),
                   dim3(std::min(threads, 256)), 0, stream>>>(
                    layer_base_addrs, kv_cache_block_id, token_counts,
                    layer_to_group, batch_size, layer_num, batch_dim,
                    max_blocks_per_batch, block_stride_bytes, seq_size_per_block);
            break;
    }

#undef LAUNCH_FUSED
#undef LAUNCH_PARALLEL
}

// Production entry point — kLayerParallel: best expected per-step latency
// (low no-op overhead ~2us dominates since most steps skip zeroing).
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
                                       cudaStream_t       stream) {
    invokeZeroIncompleteKvCacheBlocksVariant(
        layer_base_addrs, kv_cache_block_id, token_counts, layer_to_group,
        batch_size, layer_num, batch_dim, max_blocks_per_batch,
        block_stride_bytes, seq_size_per_block, stream,
        BlockZeroVariant::kLayerParallel);
}

}  // namespace rtp_llm
