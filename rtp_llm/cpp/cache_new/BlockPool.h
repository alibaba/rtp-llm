#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <set>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache_new/BlockRefCounter.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"

namespace rtp_llm {

// TODO, 0 号block不能算作free
class BlockPool {
public:
    BlockPool(const BlockPoolConfig& config,
              rtp_llm::DeviceBase*   device,
              AllocationType         atype = AllocationType::DEVICE);
    ~BlockPool();

    bool init();

    BlockCacheV1Ptr blockCache();

    // size_t totalBlocks() const;
    size_t totalBlockNums() const;
    size_t freeBlockNums() const;

    MemoryType                 where() const;
    std::vector<torch::Tensor> layerCacheBase() const;

    std::vector<BlockIdxType> malloc(int num_blocks);
    void                      free(BlockIdxType block_idx);
    void                      free(const BlockIndicesType& block_indices);
    void                      reference(const BlockIndicesType& block_indices);

    void            regUserMr(size_t model_id);
    BlockAddrInfo   convertIndexToAddr(int layer_id, int block_id) const;
    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const;

    void* getKCacheAddr(int layer_id, int block_id) const;
    void* getVCacheAddr(int layer_id, int block_id) const;

    // For backward compatibility with old cache system
    KVCacheBuffer kvCacheBuffer() const;

    void incrBlockRefCounter(const BlockIndicesType& blocks) {}
    void decrBlockRefCounter(const BlockIndicesType& blocks) {}

private:
    void initFreeBlocks();
    void deregUserMr();
    // void initKvCacheNormal();
    // void initKvCacheMla();
    // void initKvCacheScale();
    // void initLinearCache();
    // void initFreeBlock();

    // void incrBlockRefCounter(const std::vector<int>& blocks);
    // void decrBlockRefCounter(const std::vector<int>& blocks);

private:
    BlockPoolConfig                        config_;
    std::set<BlockIdxType>                 free_block_ids_;
    std::unordered_map<int, torch::Tensor> layer_kv_tensors_;  // global_layer_id -> kv cache addresses
    BlockRefCounter                        block_ref_counter_;
    rtp_llm::DeviceBase*                   device_;
    AllocationType                         atype_;

    BlockCacheV1Ptr block_cache_;

    rtp_llm::BufferPtr cache_aligned_buffer_;
    void*              cache_base_ptr_  = nullptr;
    bool               kvcache_reg_mr_  = false;
    int64_t            mr_cost_time_ms_ = 0;

    std::unique_ptr<MemoryLayoutStrategy> layout_strategy_;
};

using BlockPoolPtr = std::shared_ptr<BlockPool>;

}  // namespace rtp_llm
