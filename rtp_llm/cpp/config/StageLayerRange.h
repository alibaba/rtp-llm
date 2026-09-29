#pragma once

#include <cstdint>

namespace rtp_llm {

/** Global layer range [begin, begin + size) owned by one PP stage. */
struct StageLayerRange {
    uint32_t begin = 0;
    uint32_t size  = 0;

    uint32_t end() const {
        return begin + size;
    }
};

}  // namespace rtp_llm
