#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/MemoryLayout.h"

namespace rtp_llm {

struct BlockPoolConfig {
    // all memory layouts share the same block id space
    uint32_t block_num = 0;

    size_t total_size_bytes = 0;

    std::vector<MemoryLayoutConfig> memory_layouts;
};

}  // namespace rtp_llm