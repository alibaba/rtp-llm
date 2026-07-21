#pragma once

#include <functional>
#include <mutex>
#include <memory>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include <torch/torch.h>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

namespace rtp_llm {

struct NeedBlocksInfo {
    int common_blocks = 0;  // shared blocks across batches
    int extra_blocks  = 0;  // extra blocks per batch
};

class KVCacheGroup {
public:
    KVCacheGroup(GroupBase                           cache_group,
                 DeviceBlockPoolPtr                  block_pool,
                 int                                 group_id,
                 const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        cache_group_(std::move(cache_group)),
        block_pool_(std::move(block_pool)),
        metrics_reporter_(metrics_reporter),
        group_id_(group_id) {}

    // Transition-only constructor for HybridPool and existing focused tests.
    KVCacheGroup(const LayerIdsType&                 layer_ids,
                 KVCacheSpecPtr                      kvcache_spec,
                 DeviceBlockPoolPtr                  block_pool,
                 int                                 group_id,
                 CacheGroupPolicy                    policy           = CacheGroupPolicy{},
                 const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        KVCacheGroup(makeLegacyCacheGroup(layer_ids, std::move(kvcache_spec), policy),
                     std::move(block_pool),
                     group_id,
                     metrics_reporter) {}

    virtual ~KVCacheGroup() = default;

    bool                init();
    virtual bool        malloc(BlockIds&            block_ids,
                               int                  seq_len,
                               bool                 enable_reuse_cache   = false,
                               int                  reserve_step         = 0,
                               std::vector<size_t>* backfilled_positions = nullptr) = 0;
    virtual MatchResult match(const CacheKeysType& cache_keys);
    virtual MatchResult matchPrefix(const CacheKeysType& cache_keys) const;
    virtual MatchResult matchSingleKey(CacheKeyType cache_key) const;
    virtual void
         insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident);
    bool materializePositions(BlockIds& block_ids, const std::vector<size_t>& positions);
    virtual void free(const BlockIndicesType& block_indices)                                                     = 0;
    virtual void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    virtual int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const                      = 0;
    // Estimate peak additional blocks needed when generating remaining_tokens more tokens.
    virtual int estimatePeakNeedBlocks(int                     seq_len,
                                       const BlockIndicesType& current_block_indices,
                                       int                     remaining_tokens,
                                       int                     reserve_step,
                                       bool                    enable_reuse_cache) const = 0;
    // Estimate the physical-block peak of a fresh batch by following initMalloc's real order:
    // allocate the common prefix once, reference it from every sequence, then allocate each private suffix.
    virtual int            estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                              int  common_seq_len,
                                                              int  remaining_tokens,
                                                              int  reserve_step,
                                                              bool enable_reuse_cache,
                                                              int  target_batch_size) const = 0;
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
    using EvictCallback = std::function<size_t(size_t)>;
    void                    setEvictCallback(EvictCallback callback);
    int                     seqSizePerBlock() const;
    const std::string&      tag() const;
    const GroupBase&        config() const;
    int                     group_id() const;
    const CacheGroupPolicy& policy() const;
    bool                    prefixReuseEnabled() const;
    CacheEvictPolicy        evictPolicy() const;
    DeviceBlockPoolPtr      blockPool() const {
        return block_pool_;
    }
    uint32_t explicitBlockNum() const;
    size_t   activeTailBlocks() const;

    virtual bool                 prefixReusable() const;
    virtual bool                 hasSparseSlots() const;
    virtual bool                 hasKernelBlockSubdiv() const;
    virtual bool                 transferTailBlocks() const;
    virtual bool                 isReservable() const;
    virtual CacheMemoryPlacement memoryPlacement() const;

protected:
    static GroupBase
    makeLegacyCacheGroup(const LayerIdsType& layer_ids, KVCacheSpecPtr spec, const CacheGroupPolicy& policy) {
        GroupBase group;
        group.tag                       = spec == nullptr ? std::string{} : spec->tag;
        group.spec                      = std::move(spec);
        group.policy                    = policy;
        group.layer_ids                 = layer_ids;
        group.seq_size_per_block        = group.spec == nullptr ? 1 : group.spec->seq_size_per_block;
        group.kernel_seq_size_per_block = group.seq_size_per_block;
        group.kv_block_stride_bytes     = group.spec == nullptr ? 0 : group.spec->block_size_bytes();
        group.kv_scale_stride_bytes     = group.spec == nullptr ? 0 : group.spec->scale_block_size_bytes();
        return group;
    }

    GroupBase                    cache_group_;
    DeviceBlockPoolPtr           block_pool_;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
    int                          group_id_         = -1;
    EvictCallback                evict_callback_;
    mutable std::mutex           evict_callback_mutex_;

    std::unordered_map<int, torch::Tensor> global_layer_to_kv_tensors;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_scale_tensors;
    std::unordered_map<int, int>           global_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
