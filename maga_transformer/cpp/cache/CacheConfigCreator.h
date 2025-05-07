#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"
#include "maga_transformer/cpp/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

namespace rtp_llm {

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const rtp_llm::GptInitParameter& param, bool is_mtp=false);
    static CacheConfig createConfig(
        const rtp_llm::GptInitParameter& param,
        const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
    static std::tuple<CacheConfig, CacheConfig> createSpConfig(
        const rtp_llm::GptInitParameter& score_param,
        const rtp_llm::GptInitParameter& propose_param,
        const std::optional<WarmUpResult>& warm_up_result,
        bool is_mtp);

private:
    static size_t getDefaultRuntimeMemorySize(const rtp_llm::GptInitParameter& param);
    static size_t getKVCacheMemorySize(const rtp_llm::GptInitParameter& param,
                                       const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
};

}  // namespace rtp_llm
