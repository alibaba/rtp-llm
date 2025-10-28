#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache_new/KVCacheSpec.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCache.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheGroup {
public:
    KVCacheGroup(const std::vector<int>& layer_ids, 
                 const KVCacheSpec& group_spec,
                 BlockCachePtr block_cache,
                 BlockPoolPtr block_pool)
        : layer_ids_(layer_ids), group_spec_(group_spec), 
          block_cache_(block_cache), block_pool_(block_pool) {}

    virtual ~KVCacheGroup() = default;

    // 纯虚函数，由派生类实现
    virtual bool init() = 0;
    virtual std::vector<int> alloc(int needed_blocks) = 0;
    virtual MatchResult match(std::vector<int64_t> cache_keys) const = 0;
    virtual void free(std::vector<int> block_indices) = 0;
    virtual void insertIntoCache(std::vector<int64_t> cache_keys, std::vector<int> block_indices) = 0;
    virtual std::unordered_map<int, torch::Tensor> layerCacheBase() const = 0;
    virtual BufferPtr convertIndexToAddr(int layer_id, int block_id) const = 0;
    virtual KVCacheGroupType type() const = 0;

    // evict first if block_pool's blocks are not enough when alloc 
    virtual bool evict(int need_evict_len) = 0;

protected:
    std::vector<int> layer_ids_;
    KVCacheSpec group_spec_;
    // BlockCachePtr block_cache_;
    BlockPoolPtr block_pool_;

    std::unordered_map<int, torch::Tensor> gloabl_layer_to_kv_tensors;
    std::unordered_map<int, torch::Tensor> gloabl_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm

