#pragma once

#include <memory>
#include <vector>
#include <utility>
#include <string>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class HybridConfigCreator {
public:
    static CacheConfig                   createHybridConfig(const ModelConfig&       model_config,
                                                            const ParallelismConfig& parallelism_config,
                                                            bool                     is_mtp            = false,
                                                            int                      gen_num_per_cycle = 0);
    static std::vector<std::vector<int>> splitIntoGroups(const std::vector<int>& ids, int group_layer_num);

    // Calculate the number of layers per group based on linear and full layers count
    static int calculateGroupLayerNum(int linear_layer_count, int full_layer_count);

private:
    static void
    setupPhysicalSizes(CacheConfig& config, const KVCacheSpecPtr& full_spec, const KVCacheSpecPtr& linear_spec);
};

}  // namespace rtp_llm
