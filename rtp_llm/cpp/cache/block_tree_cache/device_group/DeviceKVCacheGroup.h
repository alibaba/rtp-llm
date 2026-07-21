#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>

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

// DeviceKVCacheGroup: single-DeviceBlockPool sequence-allocation entity, owned by
// ComponentGroup (one per device pool). Provides the KVCacheGroup-equivalent interface
// consumed by KVCacheAllocator. Whole-sequence prefix matching is superseded by
// BlockTreeCache::match(), so no match/matchPrefix/matchSingleKey here.
class DeviceKVCacheGroup {
public:
    DeviceKVCacheGroup(const LayerIdsType&                 layer_ids,
                       KVCacheSpecPtr                      kvcache_spec,
                       DeviceBlockPoolPtr                  block_pool,
                       int                                 group_id,
                       CacheGroupPolicy                    policy           = CacheGroupPolicy{},
                       const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        layer_ids_(layer_ids),
        kvcache_spec_(std::move(kvcache_spec)),
        block_pool_(block_pool),
        policy_(policy),
        metrics_reporter_(metrics_reporter),
        group_id_(group_id),
        seq_size_per_block_(kvcache_spec_->seq_size_per_block) {}

    virtual ~DeviceKVCacheGroup() = default;

    // Injected by the owner; evicts up to `num_blocks` device blocks of the given
    // `group_id` and returns the number actually freed. Used only by ensureFreeBlocks().
    using EvictionFn = std::function<int(int /*group_id*/, size_t /*num_blocks*/)>;

    void setEvictionCallback(EvictionFn fn) {
        eviction_fn_ = std::move(fn);
    }

    bool init();
    // Allocate blocks for `seq_len` tokens; appends new IDs to `block_ids` via BlockIds::add().
    virtual bool malloc(BlockIds& block_ids, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    virtual void free(const BlockIndicesType& block_indices)                                                     = 0;
    virtual void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    virtual int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const                      = 0;
    virtual NeedBlocksInfo getNeedBlocks(
        int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled = false) const = 0;
    virtual void reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices)                         = 0;

    // Materialize exact request slots selected by a load-back ticket. Sparse groups
    // intentionally leave other slots NULL, so malloc() must not infer this from the
    // block value alone.
    bool materializePositions(BlockIds& block_ids, const std::vector<size_t>& positions);

    void                                   reference(const BlockIndicesType& new_block_indices);
    std::unordered_map<int, torch::Tensor> allLayerCacheBase() const;
    std::unordered_map<int, torch::Tensor> allLayerScaleCacheBase() const;
    BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const;
    std::vector<BlockInfo>                 convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    size_t                  freeBlocksNum() const;
    bool                    ensureFreeBlocks(int need_blocks);
    int                     seqSizePerBlock() const;
    int                     group_id() const;
    const CacheGroupPolicy& policy() const;
    CacheReusePolicy        reusePolicy() const;
    CacheEvictPolicy        evictPolicy() const;
    uint32_t                explicitBlockNum() const;
    size_t                  activeTailBlocks() const;

    virtual bool isCpShardable() const;
    virtual bool prefixReusable() const;
    virtual bool hasSparseSlots() const;
    virtual bool hasKernelBlockSubdiv() const;
    virtual bool transferTailBlocks() const;
    virtual bool cpCompactTailBlocks() const;
    virtual bool isReservable() const;
    virtual bool usesPinnedCpuBacking() const;

protected:
    LayerIdsType                 layer_ids_;
    KVCacheSpecPtr               kvcache_spec_;
    DeviceBlockPoolPtr           block_pool_;
    CacheGroupPolicy             policy_;
    EvictionFn                   eviction_fn_;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
    int                          group_id_         = 0;

    int                                    seq_size_per_block_;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_tensors;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_scale_tensors;
    std::unordered_map<int, int>           global_layer_to_local_layer;
};

using DeviceKVCacheGroupPtr = std::shared_ptr<DeviceKVCacheGroup>;

}  // namespace rtp_llm
