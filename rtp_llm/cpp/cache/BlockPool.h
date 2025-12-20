#pragma once

#include <memory>
#include <set>
#include <vector>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache/types.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"

namespace rtp_llm {

class BlockPool {
public:
    BlockPool(const BlockPoolConfig& config,
              rtp_llm::DeviceBase*   device,
              AllocationType         allocation_type = AllocationType::DEVICE);
    ~BlockPool();

    bool init();

    BlockCacheV1Ptr blockCache();

    size_t totalBlocksNum() const;
    size_t freeBlocksNum() const;
    size_t availableBlocksNum() const;

    MemoryType                 where() const;
    std::vector<torch::Tensor> layerCacheBase() const;

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
    BlockAddrInfo      convertIndexToAddr(int layer_id, int block_id) const;
    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    void* getKCacheAddr(int layer_id, int block_id) const;
    void* getVCacheAddr(int layer_id, int block_id) const;

    // For backward compatibility with old cache system, TODO, 这里可以删除吗？
    KVCacheBuffer kvCacheBuffer() const;

    void incrBlockRefCounter(const BlockIndicesType& blocks) {}
    void decrBlockRefCounter(const BlockIndicesType& blocks) {}

private:
    void initFreeBlocks();
    void freeImpl(const BlockIndicesType& block_indices);

private:
    BlockPoolConfig                        config_;
    std::set<BlockIdxType>                 free_block_ids_;
    std::unordered_map<int, torch::Tensor> layer_kv_tensors_;  // global_layer_id -> kv cache addresses
    BlockRefCounter                        all_ref_counter_;
    BlockRefCounter                        request_ref_counter_;
    rtp_llm::DeviceBase*                   device_;
    AllocationType                         allocation_type_;

    BlockCacheV1Ptr block_cache_;

    rtp_llm::BufferPtr cache_aligned_buffer_;
    void*              cache_base_ptr_  = nullptr;
    bool               kvcache_reg_mr_  = false;
    int64_t            mr_cost_time_ms_ = 0;

    std::unique_ptr<MemoryLayoutStrategy> layout_strategy_;
};

using BlockPoolPtr = std::shared_ptr<BlockPool>;

}  // namespace rtp_llm
