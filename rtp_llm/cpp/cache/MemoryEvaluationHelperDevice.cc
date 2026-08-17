#include "rtp_llm/cpp/cache/MemoryEvaluationHelperDevice.h"

#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

size_t getKVCacheMemorySizeFromDevice(const RuntimeConfig&                             runtime_config,
                                      const KVCacheConfig&                             kv_cache_config,
                                      const ModelConfig&                               model_config,
                                      const std::optional<WarmUpResult>&               warm_up_result,
                                      const std::optional<SpeculativeExecutionConfig>& sp_config) {
    return MemoryEvaluationHelper::getKVCacheMemorySize(runtime_config,
                                                        kv_cache_config,
                                                        model_config,
                                                        getGpuExecStatus().device_memory_status,
                                                        warm_up_result,
                                                        sp_config);
}

}  // namespace rtp_llm
