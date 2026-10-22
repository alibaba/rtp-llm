#pragma once

#include <cstdint>

namespace rtp_llm {

enum class Tier : int8_t {
    DEVICE = 0,  // L1: GPU
    HOST   = 1,  // L2: CPU memory
    DISK   = 2,  // L3: Local disk
    REMOTE = 3,  // L4: Remote storage
    NONE   = 4,  // No tier (direct release)
};

inline const char* tierName(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            return "DEVICE";
        case Tier::HOST:
            return "HOST";
        case Tier::DISK:
            return "DISK";
        case Tier::REMOTE:
            return "REMOTE";
        case Tier::NONE:
            return "NONE";
    }
    return "UNKNOWN";
}

}  // namespace rtp_llm
