#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

namespace rtp_llm {

class SWAKVCacheGroup: public KVCacheGroup {
public:
    SWAKVCacheGroup(const LayerIdsType&          layer_ids,
                    std::shared_ptr<KVCacheSpec> kvcache_spec,
	                    BlockPoolPtr                 block_pool,
	                    int                          group_id,
	                    int                          linear_step  = 0,
	                    SharedBlockCache*            shared_cache = nullptr,
	                    const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
	        KVCacheGroup(layer_ids, kvcache_spec, block_pool, group_id, shared_cache, metrics_reporter),
	        linear_step_(linear_step) {}

    MatchResult match(const CacheKeysType& cache_keys) override;
    MatchResult matchSingleKey(CacheKeyType cache_key) const;
    bool malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) override;
    void free(const BlockIndicesType& block_indices) override;
    void reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) override;
    int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const override;
    NeedBlocksInfo getNeedBlocks(int  common_seq_len,
                                 int  seq_len,
                                 int  reserve_step,
                                 int  reuse_blocks_len,
                                 bool reuse_enabled = false) const override;

private:
    void filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const;
    bool shouldCheckSWATailBlockIds() const;
    void checkSWATailBlockIds(const BlockIds& block_ids, const char* caller) const;

    int linear_step_ = 0;
};

using SWAKVCacheGroupPtr = std::shared_ptr<SWAKVCacheGroup>;

}  // namespace rtp_llm
