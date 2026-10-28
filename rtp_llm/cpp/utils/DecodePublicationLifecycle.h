#pragma once

#include <cstdint>
#include <optional>

namespace rtp_llm {

// Stream-lock-owned state for one logical Decode request. No allocation,
// sampling thread, source labels or metrics are attached to ordinary requests.
class DecodePublicationLifecycle {
public:
    bool recordPrefillComputedEnd(int64_t end) const {
        if (end < 0 || (prefill_computed_end_ && *prefill_computed_end_ != end)) {
            return false;
        }
        prefill_computed_end_ = end;
        return true;
    }
    const std::optional<int64_t>& prefillComputedEnd() const {
        return prefill_computed_end_;
    }
    void reset() {
        prefill_computed_end_.reset();
    }

private:
    mutable std::optional<int64_t> prefill_computed_end_;
};

}  // namespace rtp_llm
