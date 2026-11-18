#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace rtp_llm {

// One completed engine timing sample. All token/time pairs are incremental and
// owned by one RPC stream, so a batch never multiplies the timing denominator.
struct FrontendMetricUpdate {
    int64_t               context_tokens            = 0;
    int64_t               context_tokens_with_cache = 0;
    int64_t               generate_tokens           = 0;
    int64_t               execute_time_us           = 0;
    std::optional<double> sp_tpot_us;
};

// Reserve only streams that can still deliver this step. The reservation keeps
// their terminal frame pending, without delaying older token frames or errors.
// Completion publishes the already measured engine sample; no RPC future can
// accidentally wait for a later iteration. RAII releases reservations on errors.
template<typename StreamPtr>
class FrontendMetricsStep {
public:
    template<typename Range>
    explicit FrontendMetricsStep(const Range& streams, bool enabled = true) {
        if (!enabled) {
            return;
        }
        streams_.reserve(streams.size());
        for (const auto& stream : streams) {
            if (stream->beginFrontendMetricsStep()) {
                streams_.push_back(stream);
            }
        }
    }
    FrontendMetricsStep(const FrontendMetricsStep&)            = delete;
    FrontendMetricsStep& operator=(const FrontendMetricsStep&) = delete;
    ~FrontendMetricsStep() {
        for (const auto& stream : streams_) {
            stream->endFrontendMetricsStep();
        }
    }
    bool publish(const FrontendMetricUpdate& update) {
        for (const auto& stream : streams_) {
            // A request may fail during dispatch. Try another reserved owner;
            // a newly finished request is still eligible until its final frame.
            if (stream->publishFrontendMetrics(update)) {
                return true;
            }
        }
        return false;
    }
    bool empty() const {
        return streams_.empty();
    }

private:
    std::vector<StreamPtr> streams_;
};

}  // namespace rtp_llm
