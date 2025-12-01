#pragma once

namespace rtp_llm {

struct WarmUpResult {
    size_t device_reserved_bytes = 0;
    size_t max_used_memory       = 0;
};

}  // namespace rtp_llm
