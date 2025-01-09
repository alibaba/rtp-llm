#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class SystemPromptConstructor {
public:
    static absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> construct(
        const ft::GptInitParameter& params, EngineBase* engine, CacheManager* cache_manager, bool insert_kv_cache);
};

}  // namespace rtp_llm
