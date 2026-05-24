#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"  // UnifiedRefCounter (M03-PR3)
#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

// M01-PR2: super-block-scope free list. Owned by HybridPoolKVCacheAllocator
// when ``CacheConfig::super_block_layout.enabled == true``. ID 0 is reserved
// (M01 §1.1 invariant 1 — kept disjoint from NULL_BLOCK_IDX=-1 and BlockPool's
// skipped slot 0). The free list holds IDs ``[1, num_super_blocks)``.
class SuperBlockFreeList {
public:
    explicit SuperBlockFreeList(uint32_t num_super_blocks);

    // Returns next free super_block_id (>0), or -1 if exhausted.
    int  allocSuperBlock();
    void freeSuperBlock(int S);

    // Total budget (== num_super_blocks at construction).
    size_t totalCount() const;
    // Current free count.
    size_t freeCount() const;

private:
    mutable std::mutex mu_;
    std::deque<int>    free_list_;  // values are valid IDs in [1, num_super_blocks_)
    uint32_t           num_super_blocks_{0};
};

class HybridPoolKVCacheAllocator: public HybridKVCacheAllocator {
public:
    HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                               AllocationType                     allocation_type     = AllocationType::DEVICE,
                               const kmonitor::MetricsReporterPtr metrics_reporter    = nullptr,
                               int64_t                            reserve_block_ratio = 0,
                               RoleType                           role_type           = RoleType::PDFUSION);

    // M03-PR3: explicit dtor — null out the SharedBlockCache's raw pointers
    // into our owned counter / super-block free list BEFORE they are
    // destroyed, so any late call into the cache (e.g. a deferred callback)
    // sees a clean wiring rather than a use-after-free.
    ~HybridPoolKVCacheAllocator() override;

    BlockAddrInfo          convertIndexToAddr(int layer_id, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override;
    BlockAddrInfo convertIndexToAddr(int layer_id, KVCacheRegionName region_name, int block_id) const override;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id) const override;
    std::vector<BlockInfo> convertIndexToBuffer(int               layer_id,
                                                KVCacheRegionName region_name,
                                                int               block_id,
                                                int               partition_count,
                                                int               partition_id) const override;
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end) override;

    CacheLayerLayout allLayerCacheBase() const override;

    size_t                  freeBlocksNum() const override;
    size_t                  availableBlocksNum() const override;
    BatchKVCacheResourcePtr popBlocksFromCache(size_t min_blocks_to_free) override;
    void                    blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) override;
    size_t                  requestRefBlocksNum() const override;
    size_t                  connectorRefBlocksNum() const override;
    size_t                  blockCacheRefBlocksNum() const override;
    size_t                  notInUseBlocksNum() const override;
    size_t                  availableTokensNum() const override;
    size_t                  totalBlocksNum() const override;
    size_t                  maxAvailableTokensNum() const override;
    void                    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr) override;
    int64_t                 getMrCostTimeMs() const override;

    // Per-pool access for diagnostics / per-pool metrics reporting.
    const std::vector<BlockPoolPtr>& groupBlockPools() const {
        return group_block_pools_;
    }

    // M01-PR2: unified-path public primitives. Only initialised in doInit()
    // when ``config_.super_block_layout.enabled == true``. When disabled the
    // accessors below RTP_LLM_FAIL — callers MUST check the config flag.
    int    allocSuperBlock();
    void   freeSuperBlock(int S);
    size_t freeSuperBlocksNum() const;

    // M01-PR2: unified malloc/free overrides. Default impls on KVCacheAllocator
    // FAIL; HybridPoolKVCacheAllocator provides the only real implementation.
    MallocResult unifiedMalloc(const MallocInfo& malloc_info) override;
    void         unifiedFree(const FreeInfo& free_info) override;

private:
    bool doInit() override;

    void referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const override;
    void freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) override;
    bool hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const override;

    int groupIdForLayerRegion(int layer_id, KVCacheRegionName region_name) const;
    int defaultGroupIdForLayer(int layer_id) const;

    std::vector<BlockPoolPtr>           group_block_pools_;
    RoleType                            role_type_{RoleType::PDFUSION};
    std::unique_ptr<SuperBlockFreeList> super_block_allocator_;  // M01-PR2: nullptr unless enabled
    // M03-PR3: super-block-scope unified ref counter (5-counter family).
    // nullptr unless ``config_.super_block_layout.enabled``. Lifetime matches
    // the allocator; passed to ``SharedBlockCache`` as a raw pointer so the
    // cache can issue UnifiedRefCounter::bump/dec/incUseRef under the
    // dual-write contract (BlockRefCounter.h).
    std::unique_ptr<UnifiedRefCounter> unified_ref_counter_;
};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
