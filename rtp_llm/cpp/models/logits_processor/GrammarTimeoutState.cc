#include "rtp_llm/cpp/models/logits_processor/GrammarTimeoutState.h"

#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

GrammarTimeoutState::GrammarTimeoutState(int                          timeout_ms,
                                         kmonitor::MetricsReporterPtr reporter,
                                         std::string                  stream_log_tag):
    timeout_ms_(timeout_ms), reporter_(std::move(reporter)), stream_log_tag_(std::move(stream_log_tag)) {
    if (timeout_ms <= 0) {
        throw std::invalid_argument("grammar update_timeout_ms must be greater than 0");
    }
}

bool GrammarTimeoutState::record(Stage stage, std::chrono::steady_clock::duration elapsed) {
    if (disabled()) {
        return false;
    }
    auto& streak = stage == Stage::UPDATE ? consecutive_update_timeouts_ : consecutive_verify_timeouts_;
    streak       = elapsed > std::chrono::milliseconds(timeout_ms_) ? streak + 1 : 0;
    return disabled();
}

void GrammarTimeoutState::report(Stage stage, int64_t elapsed_us, int token_count) const {
    RTP_LLM_LOG_WARNING("xgrammar timeout; grammar disabled for stream, stage=%s, consecutive_timeouts=%d, "
                        "budget_ms=%d, elapsed_us=%ld, token_count=%d, %s",
                        stage == Stage::UPDATE ? "update/updateStatus" : "verify/tryAcceptAndFillBitmask",
                        kTimeoutLimit,
                        timeout_ms_,
                        static_cast<long>(elapsed_us),
                        token_count,
                        stream_log_tag_.c_str());
    if (reporter_) {
        RtpLLMGrammarUpdateMetricsCollector collector;
        collector.update_timeout_qps = true;
        reporter_->report<RtpLLMGrammarMetrics, RtpLLMGrammarUpdateMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
