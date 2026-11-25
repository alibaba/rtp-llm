#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

struct BlockTreeGroupSetSizing {
    std::string first_tag;
    size_t      pool_stride = 0;
};

struct BlockTreeCacheSizing {
    std::vector<BlockTreeGroupSetSizing> group_sets;
    size_t                               host_budget_bytes = 0;
    size_t                               disk_budget_bytes = 0;
    // 1.0 retains device-proportional HOST sizing; non-default values split usable FULL/SWA slots.
    double host_swa_block_ratio = 1.0;
};

using TierBlockNumOverrides = std::unordered_map<CacheGroupType, size_t>;

struct BlockTreeCacheBlockNums {
    TierBlockNumOverrides host;
    TierBlockNumOverrides disk;
};

struct GroupSetPoolSizing {
    CacheGroupType type;
    size_t         device_blocks;
    size_t         pool_stride;
};

std::vector<size_t> computeTierBlockCounts(size_t                     capacity_bytes,
                                           const std::vector<size_t>& device_block_counts,
                                           const std::vector<size_t>& pool_strides);

// Ratios apply to usable blocks, while the byte budget also includes each
// pool's reserved block zero. Without both FULL and SWA, preserve equal counts.
std::vector<size_t> computeHostUsableBlockCounts(size_t                             capacity_bytes,
                                                 const std::vector<size_t>&         strides,
                                                 const std::vector<CacheGroupType>& types,
                                                 double                             swa_full_ratio);

std::optional<BlockTreeCacheBlockNums> computeHostDiskBlockCounts(const std::vector<GroupSetPoolSizing>& group_sets,
                                                                  const BlockTreeCacheSizing&            sizing);

/**
 * Final physical block counts: DEVICE per tag, HOST/DISK per GroupSet type.
 */
struct NegotiatedCapacity {
    uint32_t                paged_block_num = 0;
    PPBlockNumOverrides     block_num_overrides;
    BlockTreeCacheBlockNums block_tree;
    /** HOST-to-DISK eviction charge per block, averaged over stages that write this GroupSet. */
    std::unordered_map<CacheGroupType, size_t> disk_eviction_block_bytes;
};

/**
 * Cross-stage capacity negotiation hook, invoked by engine initialization
 * after intra-stage alignment and before KVCacheManager construction.
 * Implementations abort startup on failure, so a returned value is usable.
 */
class CacheCapacityNegotiator {
public:
    virtual ~CacheCapacityNegotiator() = default;

    /**
     * topology is this stage's unfinalized local topology; local_block_num is the
     * stage-aligned device capacity, and block_tree_sizing comes
     * from the already-planned local BlockTree layout.
     */
    virtual NegotiatedCapacity negotiate(const CacheConfig&          topology,
                                         uint32_t                    local_block_num,
                                         const BlockTreeCacheSizing& block_tree_sizing,
                                         const RuntimeConfig&        runtime_config) = 0;
};

}  // namespace rtp_llm
