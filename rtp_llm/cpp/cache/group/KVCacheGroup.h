#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include <torch/torch.h>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"

namespace rtp_llm {

struct NeedBlocksInfo {
    int common_blocks = 0;  // shared blocks across batches
    int extra_blocks  = 0;  // extra blocks per batch
};

class KVCacheGroup {
public:
    KVCacheGroup(const LayerIdsType&                 layer_ids,
                 KVCacheSpecPtr                      kvcache_spec,
                 BlockPoolPtr                        block_pool,
                 int                                 group_id,
                 CacheGroupPolicy                    policy           = CacheGroupPolicy{},
                 SharedBlockCache*                    shared_cache     = nullptr,
                 const kmonitor::MetricsReporterPtr&  metrics_reporter = nullptr):
        layer_ids_(layer_ids),
        kvcache_spec_(std::move(kvcache_spec)),
        block_pool_(block_pool),
        policy_(policy),
        shared_cache_(shared_cache),
        metrics_reporter_(metrics_reporter),
        group_id_(group_id),
        seq_size_per_block_(kvcache_spec_->seq_size_per_block) {}

    virtual ~KVCacheGroup() = default;

    bool init();
    virtual bool malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    virtual MatchResult match(const CacheKeysType& cache_keys);
    virtual MatchResult matchPrefix(const CacheKeysType& cache_keys);
    virtual MatchResult matchSingleKey(CacheKeyType cache_key) const;
    virtual void        free(const BlockIndicesType& block_indices) = 0;
    virtual void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    virtual int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const                      = 0;
    virtual NeedBlocksInfo getNeedBlocks(
        int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled = false) const = 0;
    virtual void reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices)                         = 0;

    void                                   reference(const BlockIndicesType& new_block_indices);
    std::unordered_map<int, torch::Tensor> allLayerCacheBase() const;
    std::unordered_map<int, torch::Tensor> allLayerScaleCacheBase() const;
    BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const;
    std::vector<BlockInfo>                 convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    size_t freeBlocksNum() const;
    bool   ensureFreeBlocks(int need_blocks);
    int    seqSizePerBlock() const;
    int    group_id() const;
    const CacheGroupPolicy& policy() const;

    CacheReusePolicy reusePolicy() const;
    CacheEvictPolicy evictPolicy() const;
    int              explicitBlockNum() const;
    int              activeTailBlocks() const;
    bool             isCpShardable() const;
    bool             prefixReusable() const;
    bool             hasSparseSlots() const;
    bool             hasKernelBlockSubdiv() const;
    bool             transferTailBlocks() const;
    bool             cpCompactTailBlocks() const;
    bool             isReservable() const;
    bool             usesPinnedCpuBacking() const;

protected:
    LayerIdsType      layer_ids_;
    KVCacheSpecPtr    kvcache_spec_;
    BlockPoolPtr      block_pool_;
    CacheGroupPolicy  policy_;
    SharedBlockCache* shared_cache_ = nullptr;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
    int               group_id_     = 0;

    int                                    seq_size_per_block_;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_tensors;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_scale_tensors;
    std::unordered_map<int, int>           global_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
