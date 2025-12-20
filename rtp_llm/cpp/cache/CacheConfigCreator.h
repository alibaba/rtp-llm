#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"

namespace rtp_llm {

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const rtp_llm::GptInitParameter& param, bool is_mtp = false);
    static CacheConfig createConfig(const rtp_llm::GptInitParameter&   param,
                                    const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
    static std::tuple<CacheConfig, CacheConfig> createSpConfig(const rtp_llm::GptInitParameter&   score_param,
                                                               const rtp_llm::GptInitParameter&   propose_param,
                                                               const std::optional<WarmUpResult>& warm_up_result,
                                                               bool                               is_mtp,
                                                               bool                               is_eagle);

private:
    static size_t getDefaultRuntimeMemorySize(const rtp_llm::GptInitParameter& params);
    static size_t getKVCacheMemorySize(const rtp_llm::GptInitParameter&   params,
                                       const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
};

}  // namespace rtp_llm
