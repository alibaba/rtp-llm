#include "rtp_llm/cpp/cache/CacheBlockMapper.h"

#include <limits>
#include <numeric>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

size_t CacheBlockMapper::cacheKeysPerPhysicalBlock(const CacheConfig& config, std::string_view tag) {
    const size_t cache_key_span = config.seq_size_per_block;
    const size_t physical_span  = config.group(tag).seqSizePerBlock();
    RTP_LLM_CHECK_WITH_INFO(cache_key_span > 0,
                            "cache-key projection requires positive logical block size for tag=%.*s",
                            static_cast<int>(tag.size()),
                            tag.data());
    RTP_LLM_CHECK_WITH_INFO(physical_span >= cache_key_span && physical_span % cache_key_span == 0,
                            "cache-key projection requires tag=%.*s physical block size=%zu to be a positive "
                            "multiple of logical block size=%zu",
                            static_cast<int>(tag.size()),
                            tag.data(),
                            physical_span,
                            cache_key_span);
    return physical_span / cache_key_span;
}

size_t CacheBlockMapper::reuseScanAlignmentKeyBlocks(const CacheConfig& config) {
    size_t alignment = 1;
    for (const auto& group : config.groups()) {
        const bool participates = group.policy.group_type != CacheGroupType::SWA || group.policy.enable_prefix_reuse;
        if (participates) {
            alignment = checkedLeastCommonMultiple(alignment, cacheKeysPerPhysicalBlock(config, group.tag), group.tag);
        }
    }
    return alignment;
}

size_t CacheBlockMapper::physicalBlocksForCacheKeyPrefix(const CacheConfig& config,
                                                         std::string_view   tag,
                                                         size_t             cache_key_blocks) {
    if (cache_key_blocks == 0) {
        return 0;
    }
    return physicalBlocksForCacheKeyPrefix(cache_key_blocks, cacheKeysPerPhysicalBlock(config, tag), tag);
}

size_t CacheBlockMapper::physicalBlocksForCacheKeyPrefix(size_t           cache_key_blocks,
                                                         size_t           keys_per_physical_block,
                                                         std::string_view tag) {
    RTP_LLM_CHECK_WITH_INFO(keys_per_physical_block > 0,
                            "cache-key prefix projection requires positive physical span for tag=%.*s",
                            static_cast<int>(tag.size()),
                            tag.data());
    if (cache_key_blocks == 0) {
        return 0;
    }
    RTP_LLM_CHECK_WITH_INFO(cache_key_blocks % keys_per_physical_block == 0,
                            "cache-key prefix=%zu is not aligned to tag=%.*s physical span=%zu",
                            cache_key_blocks,
                            static_cast<int>(tag.size()),
                            tag.data(),
                            keys_per_physical_block);
    return cache_key_blocks / keys_per_physical_block;
}

size_t CacheBlockMapper::checkedLeastCommonMultiple(size_t lhs, size_t rhs, std::string_view tag) {
    RTP_LLM_CHECK_WITH_INFO(lhs > 0 && rhs > 0,
                            "reuse cache-key unit LCM requires positive operands: current=%zu group=%zu tag=%.*s",
                            lhs,
                            rhs,
                            static_cast<int>(tag.size()),
                            tag.data());
    const size_t common_divisor = std::gcd(lhs, rhs);
    RTP_LLM_CHECK_WITH_INFO(lhs / common_divisor <= std::numeric_limits<size_t>::max() / rhs,
                            "reuse cache-key unit LCM overflow: current=%zu group=%zu tag=%.*s",
                            lhs,
                            rhs,
                            static_cast<int>(tag.size()),
                            tag.data());
    return lhs / common_divisor * rhs;
}

std::vector<CacheStoreBlockPair> CacheBlockMapper::buildCacheKeyBlockPlan(const CacheConfig& config,
                                                                          std::string_view   tag,
                                                                          size_t             total_cache_key_blocks,
                                                                          size_t             physical_block_count) {
    if (total_cache_key_blocks == 0 || physical_block_count == 0) {
        return {};
    }
    return buildCacheKeyBlockPlan(
        total_cache_key_blocks, physical_block_count, cacheKeysPerPhysicalBlock(config, tag), tag);
}

std::vector<CacheStoreBlockPair> CacheBlockMapper::buildCacheKeyBlockPlan(size_t           total_cache_key_blocks,
                                                                          size_t           physical_block_count,
                                                                          size_t           keys_per_physical_block,
                                                                          std::string_view tag) {
    const size_t max_physical_blocks =
        physicalBlockCapacityForCacheKeys(total_cache_key_blocks, keys_per_physical_block);
    std::vector<CacheStoreBlockPair> plan;
    if (total_cache_key_blocks == 0 || physical_block_count == 0) {
        return plan;
    }
    RTP_LLM_CHECK_WITH_INFO(physical_block_count <= max_physical_blocks,
                            "cache-key projection tag=%.*s physical blocks=%zu exceed key capacity=%zu",
                            static_cast<int>(tag.size()),
                            tag.data(),
                            physical_block_count,
                            max_physical_blocks);
    RTP_LLM_CHECK_WITH_INFO(total_cache_key_blocks - 1 <= static_cast<size_t>(std::numeric_limits<int>::max()),
                            "cache-key projection ordinal exceeds int range: %zu",
                            total_cache_key_blocks - 1);
    RTP_LLM_CHECK_WITH_INFO(physical_block_count - 1 <= static_cast<size_t>(std::numeric_limits<int>::max()),
                            "cache-key projection physical slot exceeds int range: %zu",
                            physical_block_count - 1);

    plan.reserve(physical_block_count);
    for (size_t offset = 0; offset < physical_block_count; ++offset) {
        const size_t key_index =
            representativeCacheKeyPosition(offset, total_cache_key_blocks, keys_per_physical_block);
        plan.push_back({static_cast<int>(key_index), static_cast<int>(offset)});
    }
    return plan;
}

}  // namespace rtp_llm
