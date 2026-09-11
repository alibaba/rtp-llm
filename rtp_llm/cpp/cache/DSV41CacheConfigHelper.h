#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class DSV41CacheConfigHelper {
public:
    static void applyConfig(CacheConfig&             config,
                            const ModelConfig&       model_config,
                            const ParallelismConfig& parallelism_config,
                            const KVCacheConfig&     kv_cache_config,
                            bool                     is_draft,
                            int                      gen_num_per_cycle);
    static void populateOwnerMappings(CacheConfig& config);
};

}  // namespace rtp_llm
