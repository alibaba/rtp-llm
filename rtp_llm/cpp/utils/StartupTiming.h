#pragma once

#include <chrono>
#include <exception>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Measure individual startup operations. Never carry a timer through the SCR
// barrier, where the elapsed duration would include the frozen interval.
class StartupTiming {
public:
    explicit StartupTiming(const char* stage):
        stage_(stage), started_(std::chrono::steady_clock::now()), exceptions_(std::uncaught_exceptions()) {
        RTP_LLM_LOG_INFO("[RTPLLM_STARTUP] stage=%s event=begin", stage_);
    }

    ~StartupTiming() noexcept {
        try {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_).count();
            RTP_LLM_LOG_INFO("[RTPLLM_STARTUP] stage=%s event=%s elapsed_ms=%.3f",
                             stage_,
                             std::uncaught_exceptions() > exceptions_ ? "failed" : "end",
                             elapsed_ms);
        } catch (...) {
            // Diagnostics must not replace a startup exception during unwind.
        }
    }

    StartupTiming(const StartupTiming&)            = delete;
    StartupTiming& operator=(const StartupTiming&) = delete;

private:
    const char*                           stage_;
    std::chrono::steady_clock::time_point started_;
    int                                   exceptions_;
};

}  // namespace rtp_llm
