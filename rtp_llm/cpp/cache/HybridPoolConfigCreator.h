#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class HybridPoolConfigCreator {
public:
    static CacheConfig createConfig(const ModelConfig&       model_config,
                                    const ParallelismConfig& parallelism_config,
                                    const KVCacheConfig&     kv_cache_config = KVCacheConfig{},
                                    bool                     is_mtp          = false);
};

}  // namespace rtp_llm
