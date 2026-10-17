#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

namespace rtp_llm {

class CoordinatorCacheManager;

using RequiredPositions = std::unordered_set<size_t>;

struct NeedBlocksInfo {
    int common_blocks = 0;  // shared blocks across batches
    int extra_blocks  = 0;  // extra blocks per batch
};

class SingleTypeCacheManager {
public:
    SingleTypeCacheManager(GroupBase cache_group, DeviceBlockPoolPtr block_pool, int group_id):
        cache_group_(std::move(cache_group)), block_pool_(std::move(block_pool)), group_id_(group_id) {}

    // Transition-only constructor for HybridPool and existing focused tests.
    SingleTypeCacheManager(const LayerIdsType& layer_ids,
                           KVCacheSpecPtr      kvcache_spec,
                           DeviceBlockPoolPtr  block_pool,
                           int                 group_id,
                           CacheGroupPolicy    policy = CacheGroupPolicy{}):
        SingleTypeCacheManager(makeLegacyCacheGroup(std::move(kvcache_spec), policy), std::move(block_pool), group_id) {
        initializeLayerMapping(layer_ids);
    }

    virtual ~SingleTypeCacheManager() = default;

    bool init();
    bool init(const LayerIdsType& layer_ids) {
        initializeLayerMapping(layer_ids);
        return init();
    }
    virtual bool malloc(BlockIds&                block_ids,
                        int                      seq_len,
                        bool                     enable_reuse_cache   = false,
                        int                      reserve_step         = 0,
                        std::vector<size_t>*     backfilled_positions = nullptr,
                        const RequiredPositions& required_positions   = {})                                        = 0;
    virtual void removeSkippedBlocks(BlockIds& block_ids, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    // FULL/LINEAR resources retain their admission-time allocation. Sparse tail
    // groups override these to backfill a chunk tail without changing table width.
    virtual bool preparePrefillChunk(BlockIds&, int, std::vector<size_t>*) { return true; }
    virtual void releaseBeforePrefillChunk(BlockIds&, int, bool) {}
    virtual int  needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const                      = 0;
    // Estimate peak additional blocks needed when generating remaining_tokens more tokens.
    virtual int estimatePeakNeedBlocks(int                     seq_len,
                                       const BlockIndicesType& current_block_indices,
                                       int                     remaining_tokens,
                                       int                     reserve_step,
                                       bool                    enable_reuse_cache) const = 0;
    // Estimate the physical-block peak of a fresh batch by following initMalloc's real order:
    // allocate the common prefix once, reference it from every sequence, then allocate each private suffix.
    virtual int                            estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                                              int  common_seq_len,
                                                                              int  remaining_tokens,
                                                                              int  reserve_step,
                                                                              bool enable_reuse_cache,
                                                                              int  target_batch_size) const       = 0;
    virtual NeedBlocksInfo                 getNeedBlocks(int                      common_seq_len,
                                                         int                      seq_len,
                                                         int                      reserve_step,
                                                         int                      reuse_blocks_len,
                                                         bool                     reuse_enabled      = false,
                                                         const RequiredPositions& required_positions = {}) const = 0;
    void                                   reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices);
    void                                   reference(const BlockIndicesType& block_indices);
    void                                   unreference(const BlockIndicesType& block_indices);
    std::unordered_map<int, torch::Tensor> allLayerCacheBase() const;
    std::unordered_map<int, torch::Tensor> allLayerScaleCacheBase() const;
    BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const;
    std::vector<BlockInfo>                 convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    size_t freeBlocksNum() const;
    bool   ensureFreeBlocks(int need_blocks);
    using EvictCallback = std::function<size_t(size_t)>;
    int                     seqSizePerBlock() const;
    const std::string&      tag() const;
    const GroupBase&        config() const;
    int                     group_id() const;
    const CacheGroupPolicy& policy() const;
    bool                    prefixReuseEnabled() const;
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
    static GroupBase makeLegacyCacheGroup(KVCacheSpecPtr spec, const CacheGroupPolicy& policy) {
        GroupBase group;
        group.tag    = spec == nullptr ? std::string{} : spec->tag;
        group.spec   = std::move(spec);
        group.policy = policy;
        return group;
    }

    void initializeLayerMapping(const LayerIdsType& layer_ids) {
        global_layer_to_local_layer.clear();
        for (size_t i = 0; i < layer_ids.size(); ++i) {
            global_layer_to_local_layer.emplace(layer_ids[i], static_cast<int>(i));
        }
    }

    GroupBase          cache_group_;
    DeviceBlockPoolPtr block_pool_;
    int                group_id_ = -1;
    EvictCallback      evict_callback_;

    std::unordered_map<int, torch::Tensor> global_layer_to_kv_tensors;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_scale_tensors;
    std::unordered_map<int, int>           global_layer_to_local_layer;

private:
    friend class CoordinatorCacheManager;
    void setEvictCallback(EvictCallback callback);
};

using SingleTypeCacheManagerPtr = std::shared_ptr<SingleTypeCacheManager>;

}  // namespace rtp_llm
