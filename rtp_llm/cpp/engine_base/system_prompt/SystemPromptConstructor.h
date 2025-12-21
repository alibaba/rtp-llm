#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class SystemPromptConstructor {
public:
    static absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> construct(
        const KVCacheConfig& kv_cache_config, EngineBase* engine, KVCacheManager* cache_manager, bool insert_kv_cache);
};

}  // namespace rtp_llm
