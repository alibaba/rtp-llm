#pragma once

#include <cstdlib>
#include <string>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace transfer {
namespace tcp_test_internal {

/// No-op after TrackerAllocator removal — PyTorch native allocator manages GPU memory.
/// Kept as a stub so callers don't need to be changed.
inline void apply_device_reserve_from_env(DeviceResourceConfig& cfg) {
    // device_reserve_memory_bytes was removed; nothing to configure.
}

}  // namespace tcp_test_internal
}  // namespace transfer
}  // namespace rtp_llm
