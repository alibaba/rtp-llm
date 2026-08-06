#pragma once

#include <cstddef>

namespace rtp_llm {

struct MemoryStatus {
    size_t used_bytes      = 0;
    size_t free_bytes      = 0;
    size_t available_bytes = 0;  // free GPU memory available for allocation
    size_t allocated_bytes = 0;  // memory allocated via current device
};

// runtime device status, such as available memory.
struct ExecStatus {
    MemoryStatus device_memory_status;
    MemoryStatus host_memory_status;
};

}  // namespace rtp_llm
