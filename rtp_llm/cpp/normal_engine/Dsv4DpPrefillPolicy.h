#pragma once

#include "rtp_llm/cpp/config/RoleTypes.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <string_view>

namespace rtp_llm {

inline bool dsv4DpPrefillPhaseSyncEnabled(std::string_view model_type,
                                          const char*      moe_strategy,
                                          RoleType         role,
                                          int64_t          pp_size,
                                          int64_t          tp_size,
                                          int64_t          dp_size,
                                          int64_t          ep_size,
                                          int64_t          world_size,
                                          bool             speculative,
                                          bool             local_cp,
                                          bool             ffn_disaggregate) {
    return model_type == "deepseek_v4" && moe_strategy && std::string_view(moe_strategy) == "sm120_decode"
           && role == RoleType::PDFUSION && pp_size == 1 && tp_size == 1 && dp_size > 1 && ep_size == dp_size
           && world_size == dp_size && !speculative && !local_cp && !ffn_disaggregate;
}

// All EP peers must choose the same collective protocol: eager prefill exchanges
// row counts, whereas captured decode starts with a packed-payload all-gather.
// Deferred decode streams remain owned by the scheduler, unchanged, for its next
// round. Do not turn a real decode stream into a prefill or advance its tokens.
template<typename StreamPtr, typename ReduceAnyPrefill, typename MakeFakePrefill>
bool alignDsv4DpPrefillPhase(std::list<StreamPtr>& streams,
                             ReduceAnyPrefill      reduce_any_prefill,
                             MakeFakePrefill       make_fake_prefill) {
    const bool local_prefill =
        std::any_of(streams.begin(), streams.end(), [](const StreamPtr& stream) { return stream->isContextStream(); });
    // Empty and decode-only ranks must also participate in this decision.
    if (!reduce_any_prefill(local_prefill)) {
        return false;
    }
    streams.remove_if([](const StreamPtr& stream) { return !stream->isContextStream(); });
    if (streams.empty()) {
        streams.emplace_back(make_fake_prefill());
    }
    return true;
}

}  // namespace rtp_llm
