#pragma once
#include <atomic>
#include <cstdint>
#include <random>
#include <string>

namespace rtp_llm {
inline std::string nextTransferTaskId() {
    // Random process identity prevents sequence reuse across coordinator restarts.
    static const std::string instance = [] {
        std::random_device random;
        std::string value;
        for (int i = 0; i < 4; ++i) {
            value += std::to_string(random()) + "-";
        }
        return value;
    }();
    static std::atomic<uint64_t> sequence{0};
    return instance + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}
}  // namespace rtp_llm
