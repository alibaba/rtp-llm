#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class FullKVCacheGroup: public KVCacheGroup {
public:
    FullKVCacheGroup(const std::vector<int>& layer_ids, 
                     std::shared_ptr<KVCacheSpec> group_spec,
                     std::shared_ptr<BlockCacheV1> block_cache,
                     BlockPoolPtr block_pool)
        : KVCacheGroup(layer_ids, group_spec, block_pool), block_cache_(block_cache) {}
    
    bool init() override;
    std::vector<BlockIdxType> alloc(int needed_blocks) override;
    MatchResult match(std::vector<CacheKeyType> cache_keys) const override;
    void free(std::vector<BlockIdxType> block_indices) override;
    void insertIntoCache(std::vector<CacheKeyType> cache_keys,
                         std::vector<BlockIdxType> block_indices,
                         std::vector<std::vector<float>> loss) override;
    std::unordered_map<int, torch::Tensor> layerCacheBase() const override;
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const override;
    
    size_t freeBlockNums() const override;
    bool evict(int need_evict_len) override;
    int seqSizePerBlock() const override;

private:
    std::shared_ptr<BlockCacheV1> block_cache_;
};

}  // namespace rtp_llm
