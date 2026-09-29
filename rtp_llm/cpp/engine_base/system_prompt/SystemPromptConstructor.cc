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

std::shared_ptr<GenerateInput> SystemPromptConstructor::makeBuildInput(const std::vector<int>& tokens_id,
                                                                       int64_t                 request_id) {
    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    generate_config->max_new_tokens = 1;
    generate_input->request_id      = request_id;
    generate_input->input_ids =
        torch::from_blob(const_cast<int*>(tokens_id.data()), {(int64_t)tokens_id.size()}, torch::kInt32).clone();
    generate_input->generate_config = generate_config;
    return generate_input;
}

absl::StatusOr<SystemPromptParams> SystemPromptConstructor::commitResident(const GenerateStreamPtr& stream,
                                                                          KVCacheManager*          cache_manager,
                                                                          const std::vector<int>&  tokens_id,
                                                                          bool                     insert_kv_cache,
                                                                          const std::string&       task_id) {
    if (!insert_kv_cache) {
        return SystemPromptParams();
    }
    auto& kv_cache = stream->kvCacheMutable();
    std::unordered_map<std::string, std::vector<int>> blocks_by_group;
    for (const auto& tag : kv_cache.cacheResource().groupTags()) {
        blocks_by_group.emplace(tag, kv_cache.blocks(0, tag));
    }
    RTP_LLM_CHECK(kv_cache.curBlocksNum() > 0);
    rtp_llm::InsertInfo insert_info{stream->kvCachePtr(),
                                    stream->completeTokenIdsPtr(),
                                    /*is_resident=*/true,
                                    /*target_tier=*/rtp_llm::Tier::DEVICE};
    size_t resident_prefix_length = 0;
    cache_manager->insertIntoCache(insert_info, resident_prefix_length);
    size_t expected_prefix_length = kv_cache.cacheKeys(0).size();
    const auto mapper = cache_manager->cpSlotMapper();
    if (mapper && mapper->isSharded()) {
        expected_prefix_length /= static_cast<size_t>(mapper->cpSize());
    }
    if (resident_prefix_length != expected_prefix_length) {
        return absl::FailedPreconditionError(
            "system prompt resident cache insertion incomplete: task_id=" + task_id
            + " resident_prefix_length=" + std::to_string(resident_prefix_length)
            + " expected_prefix_length=" + std::to_string(expected_prefix_length));
    }
    return SystemPromptParams(tokens_id, blocks_by_group);
}

absl::StatusOr<std::unordered_map<std::string, SystemPromptParams>> SystemPromptConstructor::construct(
    const KVCacheConfig& kv_cache_config, EngineBase* engine, KVCacheManager* cache_manager, bool insert_kv_cache) {
    std::unordered_map<std::string, SystemPromptParams> multi_task_prompt_args;
    std::vector<GenerateStreamPtr> prepared_streams;
    prepared_streams.reserve(kv_cache_config.multi_task_prompt_tokens.size());
    for (const auto& item : kv_cache_config.multi_task_prompt_tokens) {
        const auto& task_id   = item.first;
        const auto& tokens_id = item.second;
        auto generate_input = makeBuildInput(tokens_id, /*request_id=*/0);
        CHECK_AND_RETURN_REF(stream, engine->preRun(generate_input, preRunMode::build_system_prompt));
        CHECK_AND_RETURN_REF(params, commitResident(stream, cache_manager, tokens_id, insert_kv_cache, task_id));
        if (insert_kv_cache) {
            multi_task_prompt_args[task_id] = params;
        }
        prepared_streams.push_back(std::move(stream));
    }

    /** Keep request ownership releasable until every task succeeds. An error releases
     * request refs and partial tails; earlier resident prefixes retain their CACHE refs. */
    if (insert_kv_cache) {
        for (const auto& stream : prepared_streams) {
            stream->setNeedReleaseResource(false);
        }
    }
    return multi_task_prompt_args;
}

}  // namespace rtp_llm
