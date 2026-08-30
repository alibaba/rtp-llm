#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheBlockMapper.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct CacheConfig;

struct CpGroupLayout {
    CpBlockMappingMode mapping            = CpBlockMappingMode::NONE;
    CpBlockSliceMode   slice              = CpBlockSliceMode::NONE;
    size_t             active_tail_blocks = 0;

    bool usesCpCanonicalKeys() const {
        return mapping != CpBlockMappingMode::NONE;
    }
};

/// CP cache planner for Context Parallelism.
///
/// The class keeps the historical CPSlotMapper name, but now owns all CP cache
/// projection decisions: local allocation length, canonical key namespace,
/// cache-store key/offset plans, connector resource projection, and optional
/// intra-block slicing.
class CPSlotMapper {
public:
    CPSlotMapper();
    CPSlotMapper(int cp_rank, int cp_size, int block_size);

    bool isSharded() const {
        return cp_size_ > 1;
    }

    int cpRank() const {
        return cp_rank_;
    }
    int cpSize() const {
        return cp_size_;
    }
    int blockSize() const {
        return block_size_;
    }
    int virtualBlockSize() const {
        return virtual_block_size_;
    }

    CpGroupLayout layoutForGroup(const CacheConfig& config, std::string_view tag) const;
    bool          usesCpCanonicalKeys(const CacheConfig& config, std::string_view tag) const;
    bool          blockRoundRobinGroup(const CacheConfig& config, std::string_view tag) const;
    bool          compactLastRankGroup(const CacheConfig& config, std::string_view tag) const;

    int localBlockCount(int seq_len) const;

    // Legacy FULL-page-RR helper. Prefer the group-aware overload for new code.
    int effectiveSeqLenForAlloc(int actual_seq_len) const;
    int effectiveSeqLenForAlloc(const CacheConfig& config, std::string_view tag, int seq_len) const;

    CacheKeysType         canonicalCacheKeys(const CacheKeysType& full_keys) const;
    BlockDependenciesType canonicalBlockDependencies(const CacheKeysType& canonical_keys) const;
    size_t                logicalSeqSizePerBlock(const CacheConfig& config, std::string_view tag) const;
    CacheKeysType localCacheKeys(const CacheConfig& config, std::string_view tag, const CacheKeysType& full_keys) const;

    size_t cacheKeysPerPhysicalBlock(const CacheConfig& config, std::string_view tag) const;
    // Number of global cache-key blocks per safe reuse scan step. This is only
    // an alignment span; reuse counters never use this as their unit.
    size_t reuseScanAlignmentKeyBlocks(const CacheConfig& config) const;
    // Connector resources retain global cache-key-block counters even when
    // their key/block arrays are projected to the CP-canonical timeline.
    size_t canonicalEntryCountFromGlobalKeyBlocks(size_t global_key_blocks) const;
    size_t globalKeyBlockCountFromCanonicalEntries(size_t canonical_entries) const;
    size_t
    physicalBlocksForCacheKeyPrefix(const CacheConfig& config, std::string_view tag, size_t cache_key_blocks) const;
    // Translate tag-local physical slots to stable ordinals in the complete
    // logical cache-key timeline. CP-canonical groups use each virtual block's
    // last ordinal; logical groups use each physical block's last ordinal.
    std::vector<CacheStoreBlockPair> buildCacheKeyBlockPlan(const CacheConfig& config,
                                                            std::string_view   tag,
                                                            size_t             total_cache_key_blocks,
                                                            size_t             physical_block_count) const;

    std::vector<CacheStoreBlockPair> buildStorePlan(const CacheConfig& config,
                                                    std::string_view   tag,
                                                    size_t             total_logical_blocks,
                                                    size_t             reuse_block_size,
                                                    bool               use_hybrid) const;
    std::vector<CacheStoreBlockPair> buildStorePlan(CacheGroupType group_type,
                                                    size_t         total_logical_blocks,
                                                    size_t         reuse_block_size,
                                                    bool           use_hybrid) const;
    std::vector<CacheStoreBlockPair> buildStorePlan(const CacheGroupPolicy& policy,
                                                    size_t                  total_logical_blocks,
                                                    size_t                  reuse_block_size,
                                                    bool                    use_hybrid) const;

    std::vector<BlockInfo> sliceBlockForPeer(const CacheConfig&     config,
                                             std::string_view       tag,
                                             std::vector<BlockInfo> parts,
                                             size_t                 peer_idx) const;

    KVCacheResource projectConnectorResource(const KVCacheResource& source,
                                             const CacheConfig&     config,
                                             const CacheKeysType&   selected_keys) const;

private:
    int cp_rank_            = 0;
    int cp_size_            = 1;
    int block_size_         = 1;
    int virtual_block_size_ = 1;
};

}  // namespace rtp_llm
