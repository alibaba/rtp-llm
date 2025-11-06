#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <set>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

// first call:
// match, alloc, insert to cache

// other call:
// insert to cache, removeSkippedBlocks, alloc

// query finish:
// free

class KVCacheGroup {
public:
    // TODO, KVCacheSpec 也包含了layer ids.
    KVCacheGroup(const LayerIdsType& layer_ids, std::shared_ptr<KVCacheSpec> group_spec, BlockPoolPtr block_pool):
        layer_ids_(layer_ids),
        group_spec_(std::move(group_spec)),
        block_pool_(block_pool),
        block_cache_(block_pool_->blockCache()),
        seq_size_per_block_(group_spec_->seq_size_per_block) {}

    virtual ~KVCacheGroup() = default;

    virtual bool init()                                                                          = 0;
    virtual bool malloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) = 0;
    // TODO, match 替换为try match，和touch
    virtual MatchResult match(CacheKeysType& cache_keys)                                                       = 0;
    virtual void        free(const BlockIndicesType& block_indices)                                            = 0;
    virtual void insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices, bool is_resident) = 0;
    virtual void removeSkippedBlocks(BlockIndicesType& block_indices)                                          = 0;
    virtual int  needBlocksNum(int seq_len, int current_blocks) const                                          = 0;

    virtual std::unordered_map<int, torch::Tensor> layerCacheBase() const                                 = 0;
    virtual BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const   = 0;
    virtual BlockBufferInfo                        convertIndexToBuffer(int layer_id, int block_id) const = 0;

    virtual size_t freeBlockNums() const = 0;
    bool           ensureFreeBlocks(int need_blocks);

    int seqSizePerBlock() const {
        return seq_size_per_block_;
    }

protected:
    LayerIdsType                 layer_ids_;
    std::shared_ptr<KVCacheSpec> group_spec_;
    BlockPoolPtr                 block_pool_;
    BlockCacheV1Ptr              block_cache_;

    int                                    seq_size_per_block_;
    std::unordered_map<int, torch::Tensor> gloabl_layer_to_kv_tensors;
    std::unordered_map<int, int>           gloabl_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
