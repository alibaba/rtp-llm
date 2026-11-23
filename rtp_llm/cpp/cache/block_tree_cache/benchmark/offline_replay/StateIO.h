#pragma once
#include <stdexcept>

namespace rtp_llm::benchmark {
inline void stateRequire(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
}  // namespace rtp_llm::benchmark
