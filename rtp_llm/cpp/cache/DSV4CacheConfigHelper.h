#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include <cstdint>

namespace rtp_llm {

class DSV4CacheConfigHelper {
public:
    static void applyConfig(CacheConfig&             config,
                            const ModelConfig&       model_config,
                            const ParallelismConfig& parallelism_config,
                            const KVCacheConfig&     kv_cache_config,
                            int                      gen_num_per_cycle);
};

}  // namespace rtp_llm
