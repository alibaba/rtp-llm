#pragma once

#include <memory>
#include <vector>
#include <utility>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class HybridConfigCreator {
public:
    static CacheConfig                   createHybridConfig(const ModelConfig&       model_config,
                                                            const ParallelismConfig& parallelism_config,
                                                            bool                     is_mtp = false);
    static std::vector<std::vector<int>> splitIntoGroups(const std::vector<int>& ids, int group_layer_num);

    // Calculate the number of layers per group based on linear and full layers count
    static int calculateGroupLayerNum(int linear_layer_count, int full_layer_count);

private:
    // Helper functions for creating hybrid config in the order they appear in the main flow
    static std::pair<std::vector<int>, std::vector<int>> splitLayersByAttentionType(const ModelConfig& model_config);
    static CacheConfig                                   initializeConfig(const ModelConfig&      model_config,
                                                                          const std::vector<int>& linear_layers,
                                                                          const std::vector<int>& full_layers,
                                                                          rtp_llm::DataType       dtype);
    static KVCacheSpecPtr                                createFullAttentionSpec(const ModelConfig&       model_config,
                                                                                 const ParallelismConfig& parallelism_config,
                                                                                 rtp_llm::DataType        dtype);
    static KVCacheSpecPtr                                createLinearAttentionSpec(const ModelConfig&       model_config,
                                                                                   const ParallelismConfig& parallelism_config,
                                                                                   rtp_llm::DataType        dtype);
    static std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
    createLayerGroups(const std::vector<int>& linear_layers, const std::vector<int>& full_layers, int& group_layer_num);
    static void setupCacheConfigSpecs(CacheConfig&                         config,
                                      const std::vector<std::vector<int>>& linear_groups,
                                      const std::vector<std::vector<int>>& full_groups,
                                      const KVCacheSpecPtr&                linear_spec,
                                      const KVCacheSpecPtr&                full_spec);
    static void
    setupPhysicalSizes(CacheConfig& config, const KVCacheSpecPtr& full_spec, const KVCacheSpecPtr& linear_spec);
    static void setupLayerToGroupMapping(CacheConfig& config);
};

}  // namespace rtp_llm