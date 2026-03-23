#pragma once

#include <cstdlib>
#include <string>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace transfer {
namespace tcp_test_internal {

/// Fills device_reserve_memory_bytes for tests that call DeviceFactory::initDevices with a
/// default-constructed DeviceResourceConfig (otherwise the struct default is negative).
inline void apply_device_reserve_from_env(DeviceResourceConfig& cfg) {
    constexpr int64_t k_default_reserve_bytes = 512000000;
    cfg.device_reserve_memory_bytes           = k_default_reserve_bytes;
    if (const char* p = std::getenv("DEVICE_RESERVE_MEMORY_BYTES")) {
        try {
            const auto v = std::stoll(p);
            if (v > 0) {
                cfg.device_reserve_memory_bytes = v;
            }
        } catch (...) {
        }
    }
}

}  // namespace tcp_test_internal
}  // namespace transfer
}  // namespace rtp_llm
