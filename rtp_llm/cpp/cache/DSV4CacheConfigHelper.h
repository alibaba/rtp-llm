#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class DSV4CacheConfigHelper {
public:
    static void applyConfig(CacheConfig& config, const ModelConfig& model_config, const KVCacheConfig& kv_cache_config);
};

}  // namespace rtp_llm
