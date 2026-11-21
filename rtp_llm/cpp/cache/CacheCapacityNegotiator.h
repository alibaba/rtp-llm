#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

struct BlockTreeGroupSetSizing {
    std::string first_tag;
    size_t      pool_stride = 0;
    std::vector<std::string> member_tags;
};

inline std::string blockTreeGroupIdentity(const std::vector<std::string>& members) {
    RTP_LLM_CHECK_WITH_INFO(!members.empty(), "BlockTree group identity needs members");
    return *std::min_element(members.begin(), members.end());
}

struct BlockTreeCacheSizing {
    std::vector<BlockTreeGroupSetSizing> group_sets;
    size_t                               host_budget_bytes = 0;
    size_t                               disk_budget_bytes = 0;
};

using TierBlockNumOverrides = std::unordered_map<std::string, size_t>;

struct BlockTreeCacheBlockNums {
    TierBlockNumOverrides host;
    TierBlockNumOverrides disk;
};

std::vector<size_t> computeTierBlockCounts(size_t                     capacity_bytes,
                                           const std::vector<size_t>& device_block_counts,
                                           const std::vector<size_t>& pool_strides);

/**
 * Final physical block counts: DEVICE per tag, HOST/DISK per canonical first member tag.
 */
struct NegotiatedCapacity {
    uint32_t                paged_block_num = 0;
    PPBlockNumOverrides     block_num_overrides;
    BlockTreeCacheBlockNums block_tree;
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
