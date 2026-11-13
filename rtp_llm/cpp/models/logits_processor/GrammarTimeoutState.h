#pragma once

#include <chrono>
#include <string>

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

// Owned by one grammar processor and protected by that processor's state mutex.
class GrammarTimeoutState {
public:
    enum class Stage {
        UPDATE,
        VERIFY
    };

    GrammarTimeoutState(int timeout_ms, kmonitor::MetricsReporterPtr reporter, std::string stream_log_tag);

    bool disabled() const {
        return consecutive_update_timeouts_ == kTimeoutLimit || consecutive_verify_timeouts_ == kTimeoutLimit;
    }

    // Record only successful, nonempty CPU work. Returns true only on the transition
    // to disabled; each stage resets only its own streak. Excludes D2H and lock wait.
    bool record(Stage stage, std::chrono::steady_clock::duration elapsed);
    // Called after releasing the processor mutex, once when record() returns true.
    void report(Stage stage, int64_t elapsed_us, int token_count) const;

private:
    static constexpr int         kTimeoutLimit = 3;
    int                          timeout_ms_;
    int                          consecutive_update_timeouts_ = 0;
    int                          consecutive_verify_timeouts_ = 0;
    kmonitor::MetricsReporterPtr reporter_;
    const std::string            stream_log_tag_;
};

}  // namespace rtp_llm
