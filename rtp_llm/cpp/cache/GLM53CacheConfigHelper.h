#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

// Appends GLM-5.3-Flash KPool INDEXER_KV/INDEXER_STATE side regions to
// the ordinary hybrid MLA/KDA cache groups.
class GLM53CacheConfigHelper {
public:
    static void appendIndexerPools(CacheConfig&             config,
                                   const ModelConfig&       model_config,
                                   const ParallelismConfig& parallelism_config,
                                   const KVCacheConfig&     kv_cache_config,
                                   int                      gen_num_per_cycle);
};

}  // namespace rtp_llm
