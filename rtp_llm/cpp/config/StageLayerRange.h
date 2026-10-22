#pragma once

#include <cstddef>
#include <cstdint>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

/** Global layer range [begin, begin + size) owned by one PP stage. */
struct StageLayerRange {
    uint32_t begin = 0;
    uint32_t size  = 0;

    uint32_t end() const {
        return begin + size;
    }
};

/** Use shared main-model counts so cache and execution agree on layer ownership. */
inline StageLayerRange stageLayerRange(const ParallelismConfig& config, int64_t stage, int64_t total_layers) {
    if (config.pp_size <= 1) {
        return {0, static_cast<uint32_t>(total_layers)};
    }
    const auto& counts = config.pp_stage_layer_counts;
    int64_t     begin  = 0;
    for (int64_t s = 0; s < stage; ++s) {
        begin += counts[static_cast<std::size_t>(s)];
    }
    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(counts[static_cast<std::size_t>(stage)])};
}

}  // namespace rtp_llm
