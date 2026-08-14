#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

// Adds the GLM-5.4 compressed-indexer side pools to an existing MLA/KDA
// hybrid cache layout. The DEFAULT MLA cache and LINEAR/KDA cache are created
// by HybridPoolConfigCreator; this helper only appends INDEXER_KV and
// INDEXER_STATE.
class GLM54CacheConfigHelper {
public:
    static void appendIndexerPools(CacheConfig&             config,
                                   const ModelConfig&       model_config,
                                   const ParallelismConfig& parallelism_config,
                                   const KVCacheConfig&     kv_cache_config,
                                   int                      gen_num_per_cycle);
};

}  // namespace rtp_llm
