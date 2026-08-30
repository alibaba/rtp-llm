#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct CacheConfig;

class CacheBlockMapper {
public:
    static size_t cacheKeysPerPhysicalBlock(const CacheConfig& config, std::string_view tag);
    static size_t reuseScanAlignmentKeyBlocks(const CacheConfig& config);

    static size_t
    physicalBlocksForCacheKeyPrefix(const CacheConfig& config, std::string_view tag, size_t cache_key_blocks);
    static size_t
    physicalBlocksForCacheKeyPrefix(size_t cache_key_blocks, size_t keys_per_physical_block, std::string_view tag);

    static size_t physicalBlockCapacityForCacheKeys(size_t cache_key_blocks, size_t keys_per_physical_block) {
        RTP_LLM_CHECK_WITH_INFO(keys_per_physical_block > 0,
                                "cache-key capacity projection requires positive physical span");
        return cache_key_blocks / keys_per_physical_block + (cache_key_blocks % keys_per_physical_block != 0);
    }

    static size_t cacheKeyCapacityForPhysicalBlocks(size_t physical_block_count, size_t keys_per_physical_block) {
        RTP_LLM_CHECK_WITH_INFO(keys_per_physical_block > 0,
                                "physical block capacity projection requires positive physical span");
        RTP_LLM_CHECK_WITH_INFO(physical_block_count <= std::numeric_limits<size_t>::max() / keys_per_physical_block,
                                "physical block capacity overflow: blocks=%zu span=%zu",
                                physical_block_count,
                                keys_per_physical_block);
        return physical_block_count * keys_per_physical_block;
    }

    static size_t physicalBlockPositionForCacheKeyPosition(size_t cache_key_position, size_t keys_per_physical_block) {
        RTP_LLM_CHECK_WITH_INFO(keys_per_physical_block > 0,
                                "cache-key position projection requires positive physical span");
        return cache_key_position / keys_per_physical_block;
    }

    static size_t representativeCacheKeyPosition(size_t physical_block_position,
                                                 size_t total_cache_key_blocks,
                                                 size_t keys_per_physical_block) {
        RTP_LLM_CHECK_WITH_INFO(total_cache_key_blocks > 0,
                                "representative cache-key position requires a non-empty cache-key timeline");
        const size_t physical_capacity =
            physicalBlockCapacityForCacheKeys(total_cache_key_blocks, keys_per_physical_block);
        RTP_LLM_CHECK_WITH_INFO(physical_block_position < physical_capacity,
                                "physical block position=%zu exceeds cache-key capacity=%zu",
                                physical_block_position,
                                physical_capacity);
        RTP_LLM_CHECK_WITH_INFO(physical_block_position < std::numeric_limits<size_t>::max(),
                                "physical block position cannot be advanced: %zu",
                                physical_block_position);
        const size_t covered_key_count =
            cacheKeyCapacityForPhysicalBlocks(physical_block_position + 1, keys_per_physical_block);
        return std::min(covered_key_count, total_cache_key_blocks) - 1;
    }
    static size_t checkedLeastCommonMultiple(size_t lhs, size_t rhs, std::string_view tag);

    static std::vector<CacheStoreBlockPair> buildCacheKeyBlockPlan(const CacheConfig& config,
                                                                   std::string_view   tag,
                                                                   size_t             total_cache_key_blocks,
                                                                   size_t             physical_block_count);
    static std::vector<CacheStoreBlockPair> buildCacheKeyBlockPlan(size_t           total_cache_key_blocks,
                                                                   size_t           physical_block_count,
                                                                   size_t           keys_per_physical_block,
                                                                   std::string_view tag);
};

}  // namespace rtp_llm
