#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>

#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class FullKVCacheGroup: public KVCacheGroup {
public:
    FullKVCacheGroup(const LayerIdsType& layer_ids, std::shared_ptr<KVCacheSpec> group_spec, BlockPoolPtr block_pool):
        KVCacheGroup(layer_ids, group_spec, block_pool) {}

    bool        init() override;
    bool        malloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) override;
    MatchResult match(CacheKeysType& cache_keys) override;
    void        free(const BlockIndicesType& block_indices) override;
    void        insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices, bool is_resident) override;
    void        removeSkippedBlocks(BlockIndicesType& block_indices) override;
    int         needBlocksNum(int seq_len, int current_blocks) const override;
    size_t      freeBlockNums() const override;

    std::unordered_map<int, torch::Tensor> layerCacheBase() const override;
    BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const override;
    BlockBufferInfo                        convertIndexToBuffer(int layer_id, int block_id) const override;

private:
};

}  // namespace rtp_llm
