#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

LowerTierLinearPoolRatio resolveLowerTierLinearPoolRatio(const LowerTierLinearPoolRequest& request) {
    constexpr const char* name = "LOWER_TIER_LINEAR_POOL_AVG_PROMPT_TOKENS";
    if (request.prompt_tokens == -1) {
        return {};
    }
    RTP_LLM_CHECK_WITH_INFO(request.prompt_tokens > 0, "%s must be -1 or positive", name);
    RTP_LLM_CHECK_WITH_INFO(request.reuse_tail_blocks > 0
                                && request.reuse_tail_blocks < std::numeric_limits<int>::max(),
                            "%s requires REUSE_CACHE and a finite REUSE_CACHE_TAIL_BLOCKS",
                            name);
    RTP_LLM_CHECK_WITH_INFO(request.full_page_tokens > 0,
                            "%s requires reusable FULL and LINEAR pools with a common FULL page span "
                            "and one LINEAR checkpoint per base page",
                            name);
    return {static_cast<uint32_t>(1 + (request.prompt_tokens - 1) / request.full_page_tokens),
            static_cast<uint32_t>(request.reuse_tail_blocks) + 1};
}

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
    for (const auto stride : strides) {
        RTP_LLM_CHECK_WITH_INFO(stride > 0, "local tier pool stride must be positive");
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
    // A PP stage may have no FULL pool and can fit more reference pages
    // than its own physical block count when its pool ratios are below one.
    size_t high = max_physical_blocks - 1;
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
                                               double                             swa_full_ratio,
                                               size_t                             reference_full_device_blocks) {
    RTP_LLM_CHECK(pool_strides.size() == types.size() && types.size() == device_block_counts.size());
    if (types.empty()) {
        return {};
    }
    const auto full = std::find(types.begin(), types.end(), CacheGroupType::FULL);
    const bool custom_swa = swa_full_ratio != 1.0 && (full != types.end() || reference_full_device_blocks > 0)
                            && std::find(types.begin(), types.end(), CacheGroupType::SWA) != types.end();
    if (linear_ratio.full_pages == 0 && !custom_swa) {
        return computeTierBlockCounts(capacity_bytes, device_block_counts, pool_strides);
    }

    std::vector<size_t> usable;
    if (linear_ratio.full_pages == 0) {
        usable = computeHostUsableBlockCounts(capacity_bytes, pool_strides, types, swa_full_ratio);
    } else {
        RTP_LLM_CHECK(linear_ratio.linear_states > 0
                      && (reference_full_device_blocks > 0
                          || (full != types.end()
                              && std::find(types.begin(), types.end(), CacheGroupType::LINEAR) != types.end())));
        RTP_LLM_CHECK_WITH_INFO(!custom_swa || (std::isfinite(swa_full_ratio) && swa_full_ratio > 0),
                                "MEMORY_CACHE_SWA_BLOCK_RATIO must be finite and positive, got %f",
                                swa_full_ratio);
        const auto full_blocks = reference_full_device_blocks > 0 ?
                                     reference_full_device_blocks :
                                     device_block_counts[std::distance(types.begin(), full)];
        RTP_LLM_CHECK(full_blocks > 0);
        const auto roundUp = [](uint64_t numerator, uint64_t denominator) {
            return numerator / denominator + (numerator % denominator != 0);
        };
        usable = maxTierUsableBlockCounts(capacity_bytes, pool_strides, [&](size_t baseline, size_t i) -> long double {
            switch (types[i]) {
                case CacheGroupType::FULL:
                    return baseline;
                case CacheGroupType::LINEAR:
                    return linear_ratio.linearUsableBlocks(baseline);
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
