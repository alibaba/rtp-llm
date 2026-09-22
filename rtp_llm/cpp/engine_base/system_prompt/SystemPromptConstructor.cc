#include <string>
#include <vector>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include <torch/python.h>

namespace rtp_llm {

std::shared_ptr<GenerateInput> SystemPromptConstructor::makeBuildInput(const std::vector<int>& tokens_id,
                                                                       int64_t                 request_id) {
    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    generate_config->max_new_tokens = 1;
    generate_input->request_id      = request_id;
    generate_input->input_ids =
        torch::from_blob(const_cast<int*>(tokens_id.data()), {(int64_t)tokens_id.size()}, torch::kInt32).clone();
    generate_input->generate_config = generate_config;
    // TODO(chanyin): last partial block will be wasted when need_release_resource is false
    generate_input->need_release_resource = false;
    return generate_input;
}

absl::StatusOr<SystemPromptParams>
SystemPromptConstructor::buildAndCommitOne(EngineBase*                           engine,
                                           KVCacheManager*                       cache_manager,
                                           const std::shared_ptr<GenerateInput>& generate_input,
                                           const std::vector<int>&               tokens_id,
                                           bool                                  insert_kv_cache) {
    CHECK_AND_RETURN_REF(stream, engine->preRun(generate_input, preRunMode::build_system_prompt));
    return commitResident(stream, cache_manager, tokens_id, insert_kv_cache);
}

absl::StatusOr<SystemPromptParams> SystemPromptConstructor::commitResident(const GenerateStreamPtr& stream,
                                                                           KVCacheManager*          cache_manager,
                                                                           const std::vector<int>&  tokens_id,
                                                                           bool                     insert_kv_cache) {
    if (!insert_kv_cache) {
        return SystemPromptParams();
    }
    auto& kv_cache = stream->kvCacheMutable();
    auto& blocks   = kv_cache.blocks(0, 0);
    RTP_LLM_CHECK(blocks.size() > 0);
    rtp_llm::InsertInfo insert_info{
        stream->kvCachePtr(),
        stream->completeTokenIdsPtr(),
        true  // is_resident for system prompt
    };
    cache_manager->insertIntoCache(insert_info);
    return SystemPromptParams(tokens_id, blocks);
}

absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> SystemPromptConstructor::construct(
    const KVCacheConfig& kv_cache_config, EngineBase* engine, KVCacheManager* cache_manager, bool insert_kv_cache) {
    std::unordered_map<std::string, SystemPromptParams> multi_task_prompt_args;
    for (const auto& item : kv_cache_config.multi_task_prompt_tokens) {
        const auto& task_id   = item.first;
        const auto& tokens_id = item.second;

        auto generate_input = makeBuildInput(tokens_id, /*request_id=*/0);
        CHECK_AND_RETURN_REF(build_result,
                             buildAndCommitOne(engine, cache_manager, generate_input, tokens_id, insert_kv_cache));
        if (insert_kv_cache) {
            multi_task_prompt_args[task_id] = build_result;
        }
    }
    return multi_task_prompt_args;
}

}  // namespace rtp_llm
