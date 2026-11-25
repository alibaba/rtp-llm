#pragma once

#include <optional>
#include <sstream>

#include "rtp_llm/cpp/cache/LinearPrefillTrace.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

// Call only after a successful transfer and tree settlement, while saved
// endpoints are still held. This observes host metadata; it submits no copy,
// acquires no reference, and adds no synchronization to the transfer workflow.
inline void traceCompletedLinearTransfer(const std::vector<GroupSetPtr>& groups,
                                         const TransferDescriptor& desc,
                                         CacheKeyType key,
                                         std::optional<int64_t> request_id = {},
                                         std::optional<size_t> checkpoint_index = {}) noexcept {
    if (!linear_prefill_trace::enabled()) {
        return;
    }
    try {
        if (desc.group_set_id >= groups.size() || desc.target_tier == Tier::NONE) {
            return;
        }
        const auto& group = groups[desc.group_set_id];
        if (group->groupType() != CacheGroupType::LINEAR) {
            return;
        }
        for (size_t member = 0; member < group->groupTags().size(); ++member) {
            const auto source_index = desc.source_tier == Tier::DEVICE ? member : 0;
            const auto target_index = desc.target_tier == Tier::DEVICE ? member : 0;
            if (source_index >= desc.source_blocks.size() || target_index >= desc.target_blocks.size()) {
                continue;
            }
            std::ostringstream fields;
            fields << "\"request_id\":" << request_id.value_or(-1)
                   << ",\"group_tag\":" << linear_prefill_trace::quote(group->groupTags()[member])
                   << ",\"cache_key\":" << key
                   << ",\"source_tier\":" << linear_prefill_trace::quote(tierName(desc.source_tier))
                   << ",\"target_tier\":" << linear_prefill_trace::quote(tierName(desc.target_tier))
                   << ",\"source_block\":" << desc.source_blocks[source_index]
                   << ",\"target_block\":" << desc.target_blocks[target_index]
                   << ",\"payload_bytes\":" << group->payloadBytes()
                   << ",\"crc_enabled\":" << (group->crcEnabled() ? "true" : "false");
            if (checkpoint_index) {
                fields << ",\"checkpoint_index\":" << *checkpoint_index;
            }
            linear_prefill_trace::emit("linear_cache_transfer_completed", fields.str());
        }
    } catch (...) {
        // Missing evidence fails the smoke auditor; diagnostics must never
        // unwind an asynchronous callback and change its reference cleanup.
        RTP_LLM_LOG_ERROR("failed to record completed LINEAR cache transfer");
    }
}

}  // namespace rtp_llm
