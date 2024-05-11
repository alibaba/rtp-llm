#pragma once
#include <assert.h>
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
    static std::unordered_map<int, SystemPromptParams> construct(const ft::GptInitParameter& params, EngineBase* engine, CacheManager* cache_manager);
};

} // namespace rtp_llm
