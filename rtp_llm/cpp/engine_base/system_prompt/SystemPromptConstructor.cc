#include <string>
#include <vector>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> SystemPromptConstructor::construct(
    const rtp_llm::GptInitParameter& params, EngineBase* engine, KVCacheManager* cache_manager, bool insert_kv_cache) {
    std::unordered_map<std::string, SystemPromptParams> multi_task_prompt_args;
    for (const auto& item : params.multi_task_prompt_tokens_) {
        const auto& task_id   = item.first;
        const auto& tokens_id = item.second;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->max_new_tokens = 1;
        std::vector<size_t> shape       = {tokens_id.size()};
        generate_input->request_id      = 0;
        generate_input->input_ids       = std::make_unique<rtp_llm::Buffer>(
            rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(tokens_id.data()));
        generate_input->generate_config = generate_config;
        // TODO(chanyin): last partial block will be wasted when need_release_resource is false
        generate_input->need_release_resource = false;

        CHECK_AND_RETURN_REF(stream, engine->preRun(generate_input, preRunMode::build_system_prompt));

        if (insert_kv_cache) {
            auto& kv_cache = stream->kvCacheMutable();
            auto& blocks   = kv_cache.blocks(0, 0);
            RTP_LLM_CHECK(blocks.size() > 0);
            rtp_llm::InsertInfo insert_info(stream->kvCachePtr(),
                                            stream->completeTokenIdsPtr(),
                                            true  // is_resident for system prompt
            );
            cache_manager->insertIntoCache(insert_info);
            multi_task_prompt_args[task_id] = SystemPromptParams(tokens_id, blocks);
        }
    }
    return multi_task_prompt_args;
}

}  // namespace rtp_llm
