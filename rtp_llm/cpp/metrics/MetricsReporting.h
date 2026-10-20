#pragma once

#include <cstdint>
#include <chrono>

namespace rtp_llm {

// Process-wide publication fence. Odd epochs discard; even epochs report.
// Metric registrations stay alive across sleep/wake.
uint64_t kmonitorReportingEpoch();
struct KmonitorReportingState {
    uint64_t                              epoch;
    std::chrono::steady_clock::time_point resumed_at;
};
KmonitorReportingState kmonitorReportingState();
bool                   setKmonitorReportingEnabled(bool enabled);

}  // namespace rtp_llm
