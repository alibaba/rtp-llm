#pragma once

#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

namespace rtp_llm {

// Samples the current device and delegates the sizing decision to MemoryEvaluationHelper.
// Provided by //rtp_llm/cpp/cache:memory_evaluation_sizing_device.
size_t getKVCacheMemorySizeFromDevice(
    const RuntimeConfig&                             runtime_config,
    const KVCacheConfig&                             kv_cache_config,
    const ModelConfig&                               model_config,
    const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
    const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);

}  // namespace rtp_llm
