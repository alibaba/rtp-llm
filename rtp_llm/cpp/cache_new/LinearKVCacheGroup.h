// #pragma once

// #include "rtp_llm/cpp/cache_new/KVCacheGroup.h"

// namespace rtp_llm {

// class LinearKVCacheGroup: public KVCacheGroup {
// public:
//     LinearKVCacheGroup(const std::vector<int>& layer_ids, 
//                        const KVCacheSpec& group_spec,
//                        BlockCachePtr block_cache,
//                        BlockPoolPtr block_pool)
//         : KVCacheGroup(layer_ids, group_spec, block_cache, block_pool) {}
    
//     // 实现基类的纯虚函数
//     bool init() override;
//     std::vector<int> alloc(int needed_blocks) override;
//     MatchResult match(std::vector<int64_t> cache_keys) const override;
//     void free(std::vector<int> block_indices) override;
//     void insertIntoCache(std::vector<int64_t> cache_keys, std::vector<int> block_indices) override;
//     std::unordered_map<int, torch::Tensor> layerCacheBase() const override;
//     BufferPtr convertIndexToAddr(int layer_id, int block_id) const override;
//     KVCacheGroupType type() const override;
    
//     // evict first if block_pool's blocks are not enough when alloc 
//     bool evict(int need_evict_len) override;
// };

// using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

// }  // namespace rtp_llm

