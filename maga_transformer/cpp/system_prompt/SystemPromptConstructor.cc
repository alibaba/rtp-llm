#include <string>
#include <vector>
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/core/Buffer.h"

namespace ft = fastertransformer;

namespace rtp_llm {

absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> SystemPromptConstructor::construct(const ft::GptInitParameter& params, EngineBase* engine, CacheManager* cache_manager) {
    std::unordered_map<std::string, SystemPromptParams> multi_task_prompt_args;
    for (const auto& item: params.multi_task_prompt_tokens_) {
        const auto& task_id = item.first;
        const auto& tokens_id = item.second;

        std::shared_ptr<GenerateInput> generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->max_new_tokens = 1;
        std::vector<size_t> shape = {tokens_id.size()};
        generate_input->input_ids = std::make_unique<ft::Buffer>(ft::MEMORY_CPU, ft::TYPE_INT32, shape, (void *)(tokens_id.data()));
        generate_input->generate_config = generate_config;
        generate_input->need_release_resource = false;

        CHECK_AND_RETURN_REF(stream, engine->preRun(generate_input, preRunMode::build_system_prompt));
       
        const auto& kv_cache = stream->kvCache();
        FT_CHECK(kv_cache.batch_offset.size() == 1);
        FT_CHECK(kv_cache.batch_offset[0].size() > 0);
        cache_manager->insertResidentCache(kv_cache.batch_offset[0], tokens_id);
        multi_task_prompt_args[task_id] = SystemPromptParams(tokens_id, kv_cache.batch_offset[0]);
    }
    return multi_task_prompt_args;
}

} // namespace rtp_llm
