#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV4CacheConfig.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class DSV4ConfigCreator {
public:
    static CacheConfig
    createConfig(const ModelConfig& model_config, const ParallelismConfig& parallelism_config, bool is_mtp = false);

    // Build DSV4-specific config from model config
    static DSV4CacheConfig buildDSV4Config(const ModelConfig& model_config);

private:
    static void classifyLayers(const std::vector<int>& compress_ratios, DSV4CacheConfig& dsv4_config);

    static void buildPoolSpecs(DSV4CacheConfig& dsv4_config, const ModelConfig& model_config);

    static void populateCacheConfig(CacheConfig&             config,
                                    const DSV4CacheConfig&   dsv4_config,
                                    const ModelConfig&       model_config,
                                    const ParallelismConfig& parallelism_config);
};

}  // namespace rtp_llm
