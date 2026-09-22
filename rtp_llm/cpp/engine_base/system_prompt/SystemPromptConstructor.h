#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <optional>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class SystemPromptConstructor {
public:
    static absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> construct(
        const KVCacheConfig& kv_cache_config, EngineBase* engine, KVCacheManager* cache_manager, bool insert_kv_cache);

    static std::shared_ptr<GenerateInput> makeBuildInput(const std::vector<int>& tokens_id, int64_t request_id);

    static absl::StatusOr<SystemPromptParams> buildAndCommitOne(EngineBase*                           engine,
                                                                KVCacheManager*                       cache_manager,
                                                                const std::shared_ptr<GenerateInput>& generate_input,
                                                                const std::vector<int>&               tokens_id,
                                                                bool                                  insert_kv_cache);

    // Commits an already-executed build stream's blocks as resident KV. Shared by the non-PP
    // preRun path and the PP direct-build driver, which differ only in how the stream is run.
    static absl::StatusOr<SystemPromptParams> commitResident(const GenerateStreamPtr& stream,
                                                             KVCacheManager*          cache_manager,
                                                             const std::vector<int>&  tokens_id,
                                                             bool                     insert_kv_cache);
};

}  // namespace rtp_llm
