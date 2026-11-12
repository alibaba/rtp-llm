#include "rtp_llm/cpp/cache/block_tree_cache/metrics/EvictionMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

EvictionTimingMetrics::EvictionTimingMetrics(const CandidateMeta& candidate_meta):
    tier_enter_time_us(candidate_meta.tier_enter_time_us),
    last_access_time_us(candidate_meta.last_access_time_us),
    selected_time_us(currentTimeUs()) {}

}  // namespace rtp_llm
