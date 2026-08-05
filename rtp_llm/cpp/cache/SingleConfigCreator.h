#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class SingleConfigCreator {
public:
    static CacheConfig createSingleConfig(const ModelConfig&       model_config,
                                          const ParallelismConfig& parallelism_config,
                                          bool                     is_mtp,
                                          int                      gen_num_per_cycle);
};

}  // namespace rtp_llm