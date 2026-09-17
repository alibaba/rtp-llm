#include <string>
#include <vector>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include <torch/python.h>

namespace rtp_llm {

absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> SystemPromptConstructor::construct(
    const KVCacheConfig& kv_cache_config, EngineBase* engine, KVCacheManager* cache_manager, bool insert_kv_cache) {
    std::unordered_map<std::string, SystemPromptParams> multi_task_prompt_args;
    std::vector<std::shared_ptr<GenerateStream>>        prepared_streams;
    prepared_streams.reserve(kv_cache_config.multi_task_prompt_tokens.size());
    for (const auto& item : kv_cache_config.multi_task_prompt_tokens) {
        const auto& task_id   = item.first;
        const auto& tokens_id = item.second;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->max_new_tokens = 1;
        generate_input->request_id      = 0;
        generate_input->input_ids =
            torch::from_blob(const_cast<int*>(tokens_id.data()), {(int64_t)tokens_id.size()}, torch::kInt32).clone();
        generate_input->generate_config = generate_config;
        CHECK_AND_RETURN_REF(stream, engine->preRun(generate_input, preRunMode::build_system_prompt));

        if (insert_kv_cache) {
            auto& kv_cache = stream->kvCacheMutable();
            auto& blocks   = kv_cache.blocks(0, 0);
            RTP_LLM_CHECK(blocks.size() > 0);
            rtp_llm::InsertInfo insert_info{stream->kvCachePtr(),
                                            stream->completeTokenIdsPtr(),
                                            /*is_resident=*/true,
                                            /*target_tier=*/rtp_llm::Tier::DEVICE};
            size_t              resident_prefix_length = 0;
            cache_manager->insertIntoCache(insert_info, resident_prefix_length);
            size_t                              expected_prefix_length = kv_cache.cacheKeys(0).size();
            const std::shared_ptr<CPSlotMapper> mapper                 = cache_manager->cpSlotMapper();
            const CacheConfig&                  config                 = cache_manager->cacheConfig();
            if (mapper && mapper->isSharded()
                && (config.use_independent_block_pools || config.groupNums() > 1
                    || mapper->usesCpCanonicalKeys(config, 0))) {
                expected_prefix_length /= static_cast<size_t>(mapper->cpSize());
            }
            if (resident_prefix_length != expected_prefix_length) {
                return absl::FailedPreconditionError(
                    "system prompt resident cache insertion incomplete: task_id=" + task_id
                    + " resident_prefix_length=" + std::to_string(resident_prefix_length)
                    + " expected_prefix_length=" + std::to_string(expected_prefix_length));
            }
            multi_task_prompt_args[task_id] = SystemPromptParams(tokens_id, blocks);
        }
        prepared_streams.push_back(std::move(stream));
    }

    if (insert_kv_cache) {
        for (const auto& stream : prepared_streams) {
            stream->setNeedReleaseResource(false);
        }
    }
    return multi_task_prompt_args;
}

}  // namespace rtp_llm
