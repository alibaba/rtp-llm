#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

class BlockPool {
public:
    BlockPool(const BlockPoolConfig& config,
              rtp_llm::DeviceBase*   device,
              AllocationType         allocation_type = AllocationType::DEVICE);
    ~BlockPool();

    bool init();

    BlockCachePtr blockCache();

    size_t totalBlocksNum() const;
    size_t freeBlocksNum() const;
    size_t availableBlocksNum() const;

    MemoryType                 where() const;
    std::vector<torch::Tensor> allLayerCacheBase() const;
    std::vector<torch::Tensor> allLayerScaleCacheBase() const;

    std::vector<BlockIdxType> malloc(int num_blocks);
    void                      requestFree(BlockIdxType block_idx);
    void                      requestFree(const BlockIndicesType& block_indices);
    void                      blockCacheFree(BlockIdxType block_idx);
    void                      blockCacheFree(const BlockIndicesType& block_indices);
    void                      requestReference(BlockIdxType block_idx);
    void                      requestReference(const BlockIndicesType& block_indices);
    void                      blockCacheReference(BlockIdxType block_idx);
    void                      blockCacheReference(const BlockIndicesType& block_indices);

    void    regUserMr(size_t model_id);
    void    deregUserMr();
    int64_t getMrCostTimeMs() const {
        return mr_cost_time_ms_;
    }
    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

private:
    void initFreeBlocks();
    void freeImpl(const BlockIndicesType& block_indices);
    // global_layer_id -> {layout_index, local_layer_id}
    std::pair<int, int> mapGlobalLayerIdToLocal(int global_layer_id) const;
    void                checkLayoutValidity(int layout_id) const;

    // Helper functions for init()
    void validateConfig() const;
    void initializeCacheBuffer();
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
                                 const std::string&                   buffer_type);
    void deregisterUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                   size_t                               layout_idx,
                                   size_t                               offset_bytes,
                                   const std::string&                   buffer_type);

private:
    BlockPoolConfig        config_;
    mutable std::mutex     free_mu_;
    mutable std::mutex     ref_mu_;
    std::set<BlockIdxType> free_block_ids_;
    BlockRefCounter        all_ref_counter_;
    BlockRefCounter        request_ref_counter_;
    rtp_llm::DeviceBase*   device_;
    AllocationType         allocation_type_;

    BlockCachePtr block_cache_;

    rtp_llm::BufferPtr cache_aligned_buffer_;
    void*              cache_base_ptr_  = nullptr;
    bool               kvcache_reg_mr_  = false;
    int64_t            mr_cost_time_ms_ = 0;

    std::vector<std::unique_ptr<MemoryLayoutStrategy>> layout_strategies_;

    std::vector<std::pair<int, int>> global_layer_to_local_;

    std::vector<torch::Tensor> global_layer_kv_tensors_;
    std::vector<torch::Tensor> global_layer_kv_scale_tensors_;
};

using BlockPoolPtr = std::shared_ptr<BlockPool>;

}  // namespace rtp_llm
