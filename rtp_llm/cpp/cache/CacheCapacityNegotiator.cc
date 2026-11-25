#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

std::vector<size_t> computeTierBlockCounts(size_t                     capacity_bytes,
                                           const std::vector<size_t>& device_block_counts,
                                           const std::vector<size_t>& pool_strides) {
    RTP_LLM_CHECK(device_block_counts.size() == pool_strides.size());
    size_t weighted_bytes = 0;
    for (size_t i = 0; i < device_block_counts.size(); ++i) {
        RTP_LLM_CHECK(device_block_counts[i] > 0 && pool_strides[i] > 0);
        RTP_LLM_CHECK_WITH_INFO(device_block_counts[i]
                                    <= (std::numeric_limits<size_t>::max() - weighted_bytes) / pool_strides[i],
                                "BlockTreeCache weighted lower-tier budget overflow at group_set=%zu",
                                i);
        weighted_bytes += device_block_counts[i] * pool_strides[i];
    }
    std::vector<size_t> block_counts;
    block_counts.reserve(device_block_counts.size());
    for (size_t i = 0; i < device_block_counts.size(); ++i) {
        const size_t physical_blocks =
            static_cast<unsigned __int128>(capacity_bytes) * device_block_counts[i] / weighted_bytes;
        RTP_LLM_CHECK_WITH_INFO(
            physical_blocks >= 2 && physical_blocks <= static_cast<size_t>(std::numeric_limits<BlockIdxType>::max()),
            "BlockTreeCache tier capacity out of range: group_set=%zu budget_bytes=%zu device_blocks=%zu "
            "stride_bytes=%zu physical_blocks=%zu",
            i,
            capacity_bytes,
            device_block_counts[i],
            pool_strides[i],
            physical_blocks);
        block_counts.push_back(physical_blocks);
    }
    return block_counts;
}

std::vector<size_t> computeHostUsableBlockCounts(size_t                             capacity_bytes,
                                                 const std::vector<size_t>&         strides,
                                                 const std::vector<CacheGroupType>& types,
                                                 double                             swa_full_ratio) {
    RTP_LLM_CHECK_WITH_INFO(std::isfinite(swa_full_ratio) && swa_full_ratio > 0,
                            "MEMORY_CACHE_SWA_BLOCK_RATIO must be finite and positive, got %f",
                            swa_full_ratio);
    RTP_LLM_CHECK_WITH_INFO(strides.size() == types.size(), "HOST stride/type count mismatch");
    if (strides.empty()) {
        return {};
    }
    const bool split = std::find(types.begin(), types.end(), CacheGroupType::FULL) != types.end()
                       && std::find(types.begin(), types.end(), CacheGroupType::SWA) != types.end();
    size_t min_stride = std::numeric_limits<size_t>::max();
    for (const auto stride : strides) {
        RTP_LLM_CHECK_WITH_INFO(stride > 0, "HOST pool stride must be positive");
        min_stride = std::min(min_stride, stride);
    }
    const auto max_physical_blocks = static_cast<size_t>(std::numeric_limits<BlockIdxType>::max());
    auto       fit                 = [&](size_t baseline, std::vector<size_t>* counts) {
        size_t remaining = capacity_bytes;
        for (size_t i = 0; i < strides.size(); ++i) {
            const long double scaled = split && types[i] == CacheGroupType::SWA ?
                                                                 std::ceil(static_cast<long double>(baseline) * swa_full_ratio) :
                                                                 static_cast<long double>(baseline);
            // Leave room for block zero before converting or multiplying.
            if (scaled >= static_cast<long double>(max_physical_blocks)) {
                return false;
            }
            const auto count = static_cast<size_t>(scaled);
            if (count + 1 > remaining / strides[i]) {
                return false;
            }
            remaining -= (count + 1) * strides[i];
            if (counts) {
                (*counts)[i] = count;
            }
        }
        return true;
    };
    size_t low  = 0;
    size_t high = std::min(capacity_bytes / min_stride, max_physical_blocks - 1);
    while (low < high) {
        const size_t distance = high - low;
        const size_t mid      = low + distance / 2 + distance % 2;
        if (fit(mid, nullptr)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    std::vector<size_t> result(strides.size(), 0);
    if (low > 0) {
        RTP_LLM_CHECK(fit(low, &result));
    }
    return result;
}

std::optional<BlockTreeCacheBlockNums> computeHostDiskBlockCounts(const std::vector<GroupSetPoolSizing>& group_sets,
                                                                  const BlockTreeCacheSizing&            sizing) {
    std::vector<CacheGroupType> types;
    std::vector<size_t>         device_blocks;
    std::vector<size_t>         strides;
    for (const auto& group_set : group_sets) {
        types.push_back(group_set.type);
        device_blocks.push_back(group_set.device_blocks);
        strides.push_back(group_set.pool_stride);
    }

    std::vector<size_t> host;
    if (sizing.host_budget_bytes > 0) {
        const bool split = sizing.host_swa_block_ratio != 1.0
                           && std::find(types.begin(), types.end(), CacheGroupType::FULL) != types.end()
                           && std::find(types.begin(), types.end(), CacheGroupType::SWA) != types.end();
        if (split) {
            host = computeHostUsableBlockCounts(sizing.host_budget_bytes, strides, types, sizing.host_swa_block_ratio);
            for (auto& count : host) {
                if (count == 0) {
                    return std::nullopt;
                }
                ++count;  /** Include the reserved block zero in the physical count. */
            }
        } else {
            host = computeTierBlockCounts(sizing.host_budget_bytes, device_blocks, strides);
        }
    }
    const auto disk = sizing.disk_budget_bytes > 0 ?
                          computeTierBlockCounts(sizing.disk_budget_bytes, device_blocks, strides) :
                          std::vector<size_t>{};
    BlockTreeCacheBlockNums result;
    for (size_t i = 0; i < group_sets.size(); ++i) {
        if (!host.empty()) {
            result.host.emplace(group_sets[i].type, host[i]);
        }
        if (!disk.empty()) {
            result.disk.emplace(group_sets[i].type, disk[i]);
        }
    }
    return result;
}

}  // namespace rtp_llm
