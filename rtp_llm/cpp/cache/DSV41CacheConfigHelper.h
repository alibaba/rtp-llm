#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV41CacheState.h"
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

// Physical layout fingerprint over the descriptor (spec bytes/owners). The
// string format is frozen: it feeds DSV41CacheIdentity::cacheKeySeed().
std::string dsv41LayoutFingerprint(const CacheConfig& config);

// Assembles the model cache identity from the opaque model-identity payload
// plus the physical fingerprint. Validates the model-side policy strings.
DSV41CacheIdentity dsv41CacheIdentity(const CacheConfig& config);

}  // namespace rtp_llm
