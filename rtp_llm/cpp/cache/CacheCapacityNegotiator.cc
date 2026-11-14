#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"

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

}  // namespace rtp_llm
