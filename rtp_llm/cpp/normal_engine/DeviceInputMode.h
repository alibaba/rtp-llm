#pragma once

#include <cstdlib>
#include <cstring>

namespace rtp_llm {

// Async dispatch requires device inputs and next-step state publication in
// both normal decode and MTP, even when DEVICE_INPUT is explicitly disabled.
inline bool deviceInputEnabled() {
    const char* stream_async = std::getenv("RTP_LLM_STREAM_ASYNC");
    const char* device_input = std::getenv("RTP_LLM_DEVICE_INPUT");
    return (stream_async != nullptr && std::strcmp(stream_async, "1") == 0)
           || (device_input != nullptr && std::strcmp(device_input, "1") == 0);
}

}  // namespace rtp_llm
