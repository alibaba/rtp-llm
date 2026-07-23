#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

inline constexpr double  kDefaultRuntimeMemorySafetyRatio = 0.05;
inline constexpr int64_t kDefaultRuntimeNoWarmupFloorMiB  = 2048;

struct RuntimeMemorySizingInput {
    bool   has_warmup               = false;
    size_t configured_reserve_bytes = 0;
    size_t warmup_required_bytes    = 0;
    size_t sampler_required_bytes   = 0;
    size_t total_gpu_bytes          = 0;
    double safety_ratio             = kDefaultRuntimeMemorySafetyRatio;
    size_t no_warmup_floor_bytes    = kDefaultRuntimeNoWarmupFloorMiB * 1024 * 1024;
};

struct RuntimeMemorySizingResult {
    size_t safety_headroom_bytes  = 0;
    size_t runtime_required_bytes = 0;
};

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input);
std::optional<double>      parseRuntimeMemorySafetyRatio(std::string_view value);
std::optional<int64_t>     parseRuntimeMemoryNoWarmupFloorMiB(std::string_view value);

class MemoryEvaluationHelper {
public:
    static size_t
                  getDefaultRuntimeMemorySize(const RuntimeConfig&                             runtime_config,
                                              const ParallelismConfig&                         parallelism_config,
                                              const ModelConfig&                               model_config,
                                              const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    static size_t getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                       const KVCacheConfig&                             kv_cache_config,
                                       const ModelConfig&                               model_config,
                                       const ParallelismConfig&                         parallelism_config,
                                       const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                       const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);

    // Helper function to update memory size if below minimum requirement
    static void updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario);

    static rtp_llm::DataType getDataTypeForCache(const ModelConfig& model_config);
};

}  // namespace rtp_llm
