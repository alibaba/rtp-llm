#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace rtp_llm {

class CacheConfigCreator {
public:
    static absl::StatusOr<CacheConfig> createConfig(const ft::GptInitParameter& param);

    static absl::StatusOr<std::tuple<CacheConfig, CacheConfig>> createSpConfig(const ft::GptInitParameter& score_param, const ft::GptInitParameter& propose_param);

private:
    static CacheConfig createBasicConfig(const ft::GptInitParameter& param);
    static absl::StatusOr<int64_t> getKVCacheMemorySize(const ft::GptInitParameter& param);
};

}  // namespace rtp_llm
