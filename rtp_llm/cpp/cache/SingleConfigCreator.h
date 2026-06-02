#pragma once

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
                                          bool                     is_mtp = false);
};

}  // namespace rtp_llm