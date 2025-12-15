#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockCache.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class KVCacheGroup {
public:
    KVCacheGroup(const LayerIdsType& layer_ids, KVCacheSpecPtr kvcache_spec, BlockPoolPtr block_pool, int group_id):
        layer_ids_(layer_ids),
        kvcache_spec_(std::move(kvcache_spec)),
        block_pool_(block_pool),
        block_cache_(block_pool_->blockCache()),
        group_id_(group_id),
        seq_size_per_block_(kvcache_spec_->seq_size_per_block) {}

    virtual ~KVCacheGroup() = default;

    bool         init();
    virtual bool malloc(const CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) = 0;
    // TODO, match的时候热度不增加，最终匹配成功的时候再去增加热度。
    virtual MatchResult match(const CacheKeysType& cache_keys) = 0;
    MatchResult         matchSingleKey(CacheKeyType cache_key);
    virtual void        free(const BlockIndicesType& block_indices) = 0;
    virtual void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) = 0;
    virtual void removeSkippedBlocks(BlockIndicesType& block_indices)                                         = 0;
    virtual int  needBlocksNum(int seq_len, int current_blocks) const                                         = 0;
    virtual void reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices)        = 0;

    void                                   reference(const BlockIndicesType& new_block_indices);
    std::unordered_map<int, torch::Tensor> layerCacheBase() const;
    BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const;
    BlockBufferPtrInfo                     convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    size_t freeBlocksNum() const;
    bool   ensureFreeBlocks(int need_blocks);
    int    seqSizePerBlock() const;

protected:
    LayerIdsType    layer_ids_;
    KVCacheSpecPtr  kvcache_spec_;
    BlockPoolPtr    block_pool_;
    BlockCacheV1Ptr block_cache_;
    int             group_id_ = 0;

    int                                    seq_size_per_block_;
    std::unordered_map<int, torch::Tensor> gloabl_layer_to_kv_tensors;
    std::unordered_map<int, int>           gloabl_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
