#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

namespace rtp_llm {

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const ft::GptInitParameter& param);
    static CacheConfig createConfig(
        const ft::GptInitParameter& param,
        const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
    static std::tuple<CacheConfig, CacheConfig> createSpConfig(
        const ft::GptInitParameter& score_param,
        const ft::GptInitParameter& propose_param,
        const std::optional<WarmUpResult>& warm_up_result);

private:
    static size_t getDefaultRuntimeMemorySize(const ft::GptInitParameter& param);
    static size_t getKVCacheMemorySize(const ft::GptInitParameter& param,
                                       const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
};

}  // namespace rtp_llm
