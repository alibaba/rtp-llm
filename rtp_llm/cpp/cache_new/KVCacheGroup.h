#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheGroup {
public:
    KVCacheGroup(const std::vector<int>& layer_ids, 
                 std::shared_ptr<KVCacheSpec> group_spec,
                 BlockPoolPtr block_pool)
        : layer_ids_(layer_ids), group_spec_(std::move(group_spec)), 
          block_pool_(block_pool) {}

    virtual ~KVCacheGroup() = default;

    virtual bool init() = 0;
    virtual std::vector<BlockIdxType> alloc(int needed_blocks) = 0;
    virtual MatchResult match(std::vector<CacheKeyType> cache_keys) const = 0;
    virtual void free(std::vector<BlockIdxType> block_indices) = 0;
    virtual void insertIntoCache(std::vector<CacheKeyType> cache_keys,
                                 std::vector<BlockIdxType> block_indices,
                                 std::vector<std::vector<float>> loss) = 0;
    virtual std::unordered_map<int, torch::Tensor> layerCacheBase() const = 0;
    virtual BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const = 0;
    virtual BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const = 0;

    virtual size_t freeBlockNums() const = 0;
    virtual bool evict(int need_evict_len) = 0;
    virtual int seqSizePerBlock() const = 0;

protected:
    std::vector<int> layer_ids_;
    std::shared_ptr<KVCacheSpec> group_spec_;
    // BlockCachePtr block_cache_;
    BlockPoolPtr block_pool_;

    int seq_size_per_block_;
    std::unordered_map<int, torch::Tensor> gloabl_layer_to_kv_tensors;
    std::unordered_map<int, int> gloabl_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm

