#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "absl/status/status.h"

namespace rtp_llm {

inline constexpr char    kCoordinatedStopTimeoutEnv[]     = "RTP_LLM_COORDINATED_STOP_TIMEOUT_MS";
inline constexpr int64_t kDefaultCoordinatedStopTimeoutMs = 60000;

// Read on each shutdown operation so tests and embedding applications can set
// the value before initiating shutdown. Invalid values fall back to the safe
// default instead of turning a configuration typo into an immediate timeout.
int64_t coordinatedStopTimeoutMs();

// Wait for the engine loop to observe a coordinated-stop control operation.
// This is kept independent of NormalEngine so the long-step shutdown path can
// be exercised without constructing a GPU engine.
absl::Status waitForCoordinatedStopAck(const std::atomic<bool>&     running,
                                       const std::function<bool()>& acknowledged,
                                       int64_t                      timeout_ms,
                                       const std::string&           operation);

}  // namespace rtp_llm
