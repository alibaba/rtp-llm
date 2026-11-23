#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

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

namespace {

template<typename BlockCount>
std::vector<size_t>
maxTierUsableBlockCounts(size_t capacity_bytes, const std::vector<size_t>& strides, BlockCount block_count) {
    if (strides.empty()) {
        return {};
    }
    size_t min_stride = std::numeric_limits<size_t>::max();
    for (const auto stride : strides) {
        RTP_LLM_CHECK_WITH_INFO(stride > 0, "local tier pool stride must be positive");
        min_stride = std::min(min_stride, stride);
    }
    const auto max_physical_blocks = static_cast<size_t>(std::numeric_limits<BlockIdxType>::max());
    auto       fit                 = [&](size_t baseline, std::vector<size_t>* counts) {
        size_t remaining = capacity_bytes;
        for (size_t i = 0; i < strides.size(); ++i) {
            const long double scaled = block_count(baseline, i);
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

}  // namespace

std::vector<size_t> computeHostUsableBlockCounts(size_t                             capacity_bytes,
                                                 const std::vector<size_t>&         strides,
                                                 const std::vector<CacheGroupType>& types,
                                                 double                             swa_full_ratio) {
    RTP_LLM_CHECK_WITH_INFO(std::isfinite(swa_full_ratio) && swa_full_ratio > 0,
                            "MEMORY_CACHE_SWA_BLOCK_RATIO must be finite and positive, got %f",
                            swa_full_ratio);
    RTP_LLM_CHECK_WITH_INFO(strides.size() == types.size(), "HOST stride/type count mismatch");
    const bool split = std::find(types.begin(), types.end(), CacheGroupType::FULL) != types.end()
                       && std::find(types.begin(), types.end(), CacheGroupType::SWA) != types.end();
    return maxTierUsableBlockCounts(capacity_bytes, strides, [&](size_t baseline, size_t i) {
        return split && types[i] == CacheGroupType::SWA ?
                   std::ceil(static_cast<long double>(baseline) * swa_full_ratio) :
                   static_cast<long double>(baseline);
    });
}

std::vector<size_t> computeLocalTierBlockCounts(size_t                             capacity_bytes,
                                               const std::vector<size_t>&         device_block_counts,
                                               const std::vector<size_t>&         pool_strides,
                                               const std::vector<CacheGroupType>& types,
                                               LowerTierLinearPoolRatio           linear_ratio,
                                               double                             swa_full_ratio) {
    RTP_LLM_CHECK(pool_strides.size() == types.size() && types.size() == device_block_counts.size());
    const auto full = std::find(types.begin(), types.end(), CacheGroupType::FULL);
    const bool custom_swa = swa_full_ratio != 1.0 && full != types.end()
                            && std::find(types.begin(), types.end(), CacheGroupType::SWA) != types.end();
    if (linear_ratio.full_pages == 0 && !custom_swa) {
        return computeTierBlockCounts(capacity_bytes, device_block_counts, pool_strides);
    }

    std::vector<size_t> usable;
    if (linear_ratio.full_pages == 0) {
        usable = computeHostUsableBlockCounts(capacity_bytes, pool_strides, types, swa_full_ratio);
    } else {
        RTP_LLM_CHECK(full != types.end() && linear_ratio.linear_states > 0
                      && std::find(types.begin(), types.end(), CacheGroupType::LINEAR) != types.end());
        RTP_LLM_CHECK_WITH_INFO(!custom_swa || (std::isfinite(swa_full_ratio) && swa_full_ratio > 0),
                                "MEMORY_CACHE_SWA_BLOCK_RATIO must be finite and positive, got %f",
                                swa_full_ratio);
        const auto full_blocks = device_block_counts[std::distance(types.begin(), full)];
        RTP_LLM_CHECK(full_blocks > 0);
        const auto roundUp = [](uint64_t numerator, uint64_t denominator) {
            return numerator / denominator + (numerator % denominator != 0);
        };
        usable = maxTierUsableBlockCounts(capacity_bytes, pool_strides, [&](size_t baseline, size_t i) -> long double {
            switch (types[i]) {
                case CacheGroupType::FULL:
                    return baseline;
                case CacheGroupType::LINEAR:
                    return roundUp(static_cast<uint64_t>(baseline) * linear_ratio.linear_states, linear_ratio.full_pages);
                case CacheGroupType::SWA:
                    return custom_swa ? std::ceil(static_cast<long double>(baseline) * swa_full_ratio) :
                                        roundUp(static_cast<uint64_t>(baseline) * device_block_counts[i], full_blocks);
            }
            throw std::invalid_argument("unknown local cache group type");
        });
    }
    for (auto& count : usable) {
        RTP_LLM_CHECK_WITH_INFO(count > 0, "local tier budget cannot hold a usable block per group set");
        ++count;
    }
    return usable;
}

}  // namespace rtp_llm
