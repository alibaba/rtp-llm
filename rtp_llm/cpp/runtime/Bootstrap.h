#pragma once

#include <cstddef>

namespace rtp_llm::process {

struct BootstrapConfig {
    std::size_t device_id           = 0;
    bool        enable_stack_tracer = false;
};

// One-time process-startup side effects: setlinebuf(stdout), cudaSetDevice/hipSetDevice,
// default CUDA stream selection, optional autil stack tracer enablement.
// Idempotent via std::call_once. Holds NO readable state.
void bootstrap(const BootstrapConfig& cfg);

}  // namespace rtp_llm::process
