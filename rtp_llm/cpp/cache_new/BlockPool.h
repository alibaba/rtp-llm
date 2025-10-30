#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"


namespace rtp_llm {

class BlockPool {
public:
    struct KVCacheBuffer {
        torch::Tensor      kv_blocks;
    };

    BlockPool(const BlockPoolConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE);

    bool init();

    // size_t totalBlocks() const;
    size_t freeBlockNums() const;

    std::vector<torch::Tensor> layerCacheBase() const;

    std::vector<int> alloc(int num_blocks);
    void free(const std::vector<int>& block_ids);
    void reference(const std::vector<int>& block_ids);

    void regUserMr(size_t model_id);
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const;
    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const;
    

    void* getKCacheAddr(int layer_id, int block_id) const;
    void* getVCacheAddr(int layer_id, int block_id) const;

private:
    // void initKvCacheNormal();
    // void initKvCacheMla();
    // void initKvCacheScale();
    // void initLinearCache();
    // void initFreeBlock();

    // void incrBlockRefCounter(const std::vector<int>& blocks);
    // void decrBlockRefCounter(const std::vector<int>& blocks);

private:
    BlockPoolConfig config_;
    std::set<int> free_block_ids;
    std::unordered_map<int, torch::Tensor> layer_kv_tensors_;        // global_layer_id -> kv cache addresses
    KVCacheBuffer kv_cache_;
    BlockRefCounter block_ref_counter_;
    rtp_llm::DeviceBase* device_;
    AllocationType atype_;
    
    rtp_llm::BufferPtr cache_aligned_buffer_;
    void*              cache_base_ptr_ = nullptr;
    bool               kvcache_reg_mr_ = false;
    int64_t            mr_cost_time_ms_ = 0;
    
    // 新增：布局策略
    std::unique_ptr<MemoryLayoutStrategy> layout_strategy_;
};

using BlockPoolPtr = std::shared_ptr<BlockPool>;

}  // namespace rtp_llm
