#pragma once

#include <cstdint>
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
    static CacheConfig createHybridConfig(const ModelConfig&       model_config,
                                          const ParallelismConfig& parallelism_config,
                                          bool                     is_mtp,
                                          int                      gen_num_per_cycle);

private:
    static std::pair<std::vector<int>, std::vector<int>> splitLayersByAttentionType(const ModelConfig& model_config);
    static CacheConfig                                   initializeConfig(const ModelConfig&      model_config,
                                                                          const std::vector<int>& linear_layers,
                                                                          const std::vector<int>& full_layers,
                                                                          rtp_llm::DataType       dtype);
    static void
    setupPhysicalSizes(CacheConfig& config, const KVCacheSpecPtr& full_spec, const KVCacheSpecPtr& linear_spec);
};

}  // namespace rtp_llm
