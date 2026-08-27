#include "rtp_llm/cpp/utils/CoordinatedStopUtil.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

int64_t coordinatedStopTimeoutMs() {
    const char* raw = std::getenv(kCoordinatedStopTimeoutEnv);
    if (raw == nullptr) {
        RTP_LLM_LOG_INFO(
            "%s is unset, using default %ldms", kCoordinatedStopTimeoutEnv, kDefaultCoordinatedStopTimeoutMs);
        return kDefaultCoordinatedStopTimeoutMs;
    }

    errno                  = 0;
    char*           end    = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed <= 0) {
        RTP_LLM_LOG_WARNING(
            "invalid %s=%s, using default %ldms", kCoordinatedStopTimeoutEnv, raw, kDefaultCoordinatedStopTimeoutMs);
        return kDefaultCoordinatedStopTimeoutMs;
    }

    RTP_LLM_LOG_INFO("%s=%s -> %lldms", kCoordinatedStopTimeoutEnv, raw, parsed);
    return static_cast<int64_t>(parsed);
}

absl::Status waitForCoordinatedStopAck(const std::atomic<bool>&     running,
                                       const std::function<bool()>& acknowledged,
                                       int64_t                      timeout_ms,
                                       const std::string&           operation) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (running.load(std::memory_order_acquire) && !acknowledged()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return absl::DeadlineExceededError("engine loop did not acknowledge coordinated stop " + operation);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
