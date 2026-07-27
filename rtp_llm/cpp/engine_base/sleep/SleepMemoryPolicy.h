#pragma once

#include <cstdlib>
#include <cstring>

namespace rtp_llm::sleep_memory_policy {

// Level 1/2 preserve captured graphs across sleep, so their graph allocations
// need stable VMM addresses. Level 3 destroys and recaptures graphs instead.
// Processes without sleep mode do not need graph allocations in a TMS region.
inline bool useCudaGraphVmmRegionFromEnvironment() {
    const char* enabled = std::getenv("ENABLE_SLEEP_MODE");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
        return false;
    }

    const char* level = std::getenv("SLEEP_MODE_LEVEL");
    return level == nullptr || std::strcmp(level, "1") == 0 || std::strcmp(level, "2") == 0;
}

inline bool manageCudaGraphVmmBacking(int sleep_level) {
    return sleep_level != 3;
}

}  // namespace rtp_llm::sleep_memory_policy
