#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <unordered_map>
#include <mutex>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

class CacheStore;

class BlockPool {
public:
    BlockPool(const BlockPoolConfig& config,
              AllocationType         allocation_type         = AllocationType::DEVICE,
              bool                   use_pinned_cpu_backing  = false,
              bool                   use_cuda_malloc_backing = false);
    ~BlockPool();

    bool init();

    BlockCachePtr blockCache();

    MemoryType                 where() const;
    std::vector<torch::Tensor> allLayerCacheBase() const;
    std::vector<torch::Tensor> allLayerScaleCacheBase() const;

    // these interfaces are all thread-safe
    std::vector<BlockIdxType> malloc(int num_blocks);
    size_t                    totalBlocksNum() const;
    size_t                    freeBlocksNum() const;
    size_t                    availableBlocksNum() const;
    size_t                    requestRefBlocksNum() const;
    size_t                    connectorRefBlocksNum() const;
    size_t                    blockCacheRefBlocksNum() const;
    // Blocks not held by request or block cache (i.e. free + connector-in-flight).
    // Used by tiered memory eviction to avoid over-eviction.
    size_t notInUseBlocksNum() const;
    void   requestFree(BlockIdxType block_idx);
    void   requestFree(const BlockIndicesType& block_indices);
    void   blockCacheFree(BlockIdxType block_idx);
    void   blockCacheFree(const BlockIndicesType& block_indices);
    void   connectorFree(BlockIdxType block_idx);
    void   connectorFree(const BlockIndicesType& block_indices);
    void   requestReference(BlockIdxType block_idx);
    void   requestReference(const BlockIndicesType& block_indices);
    void   blockCacheReference(BlockIdxType block_idx);
    void   blockCacheReference(const BlockIndicesType& block_indices);
    void   connectorReference(BlockIdxType block_idx);
    void   connectorReference(const BlockIndicesType& block_indices);

    // Sleep/wake_up: reset all block metadata to the fresh-pool state after the physical
    // KV memory has been resumed (content discarded). Rebuilds free_block_ids_ to the full set
    // and re-inits every BlockRefCounter, exactly like initFreeBlocks() on a new pool.
    // Does NOT touch block_cache_ (callers clear it separately via BlockCache::clear()) and
    // does NOT recreate the underlying buffer (VA must stay stable).
    // Caller must guarantee no in-flight users of the pool (engine drained).
    void resetMetadata();

    // Sleep/wake_up: host memory-cache tier discard / reallocate.
    // Only valid for AllocationType::HOST pools (the pinned host KV offload tier).
    // releaseHostBuffer() drops the pinned host buffer (torch::empty(...).pin_memory())
    // and all tensors that view into it, returning the ~memory_cache_size_mb of pinned
    // RAM to the OS on sleep; it also empties free_block_ids_ so malloc() cannot hand
    // out blocks while released. reallocateHostBuffer() re-allocates the buffer and
    // resets all block metadata to a fresh pool on wake.
    // Caller must guarantee the pool is drained/quiesced (no in-flight copies) and must
    // clear any external cache-key->block LRU that indexes into the freed buffer.
    void releaseHostBuffer();
    void reallocateHostBuffer();

    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    void    deregUserMr();
    int64_t getMrCostTimeMs() const {
        return mr_cost_time_ms_;
    }
    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    void* getBaseAddress() const {
        return cache_base_ptr_;
    }
    size_t getTotalSizeBytes() const {
        return config_.total_size_bytes;
    }
    const std::string& poolName() const {
        return config_.pool_name;
    }

private:
    void initFreeBlocks();
    void tryFreeBlocks(const BlockIndicesType& block_indices);
    // global_layer_id -> {layout_index, local_layer_id}
    std::pair<int, int> mapGlobalLayerIdToLocal(int global_layer_id) const;
    void                checkLayoutValidity(int layout_id) const;

    // Helper functions for init()
    void validateConfig() const;
    void initializeCacheBuffer();
    void initializePinnedCpuBuffer(const char* log_context);
    void initializeCudaMallocBuffer();
    void initializeLayerMappings();
    void initializeLayoutStrategies();

    // Helper functions for initializeLayoutStrategies()
    void          processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor, size_t& global_layer_begin);
    torch::Tensor createTensor(const torch::Tensor& full_tensor,
                               int64_t              offset,
                               int64_t              size,
                               size_t               layout_idx,
                               const std::string&   tensor_type);
    void          initializeLayoutStrategy(size_t                    layout_idx,
                                           const MemoryLayoutConfig& layout_cfg,
                                           torch::Tensor&            kv_cache_tensor,
                                           torch::Tensor&            kv_scale_tensor);
    void processLayerTensors(size_t layout_idx, const MemoryLayoutConfig& layout_cfg, size_t& global_layer_begin);

    // Helper functions for regUserMr/deregUserMr
    void registerUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                 size_t                               layout_idx,
                                 size_t                               offset_bytes,
                                 size_t                               bytes,
                                 size_t                               stride_bytes,
                                 bool                                 gpu,
                                 const std::string&                   buffer_type);
    void deregisterUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                   size_t                               layout_idx,
                                   size_t                               offset_bytes,
                                   bool                                 gpu,
                                   const std::string&                   buffer_type);

private:
    BlockPoolConfig config_;

    mutable std::mutex     free_mu_;
    mutable std::mutex     ref_mu_;
    std::set<BlockIdxType> free_block_ids_;
    BlockRefCounter        request_ref_counter_;
    BlockRefCounter        connector_ref_counter_;
    BlockRefCounter        req_con_ref_counter_;
    BlockRefCounter        block_cache_ref_counter_;
    BlockRefCounter        req_cache_ref_counter_;

    AllocationType allocation_type_;
    bool           use_pinned_cpu_backing_;
    bool           use_cuda_malloc_backing_;

    BlockCachePtr block_cache_;

    torch::Tensor               cache_aligned_buffer_;
    void*                       cache_base_ptr_  = nullptr;
    bool                        host_released_   = false;  // HOST pool: buffer freed for sleep
    bool                        kvcache_reg_mr_  = false;
    int64_t                     mr_cost_time_ms_ = 0;
    std::shared_ptr<CacheStore> cache_store_;

    std::vector<std::unique_ptr<MemoryLayoutStrategy>> layout_strategies_;
    std::vector<std::pair<int, int>>                   global_layer_to_local_;
    std::vector<torch::Tensor>                         global_layer_kv_tensors_;
    std::vector<torch::Tensor>                         global_layer_kv_scale_tensors_;

    mutable std::recursive_mutex mutex_;
};

using BlockPoolPtr = std::shared_ptr<BlockPool>;

}  // namespace rtp_llm
