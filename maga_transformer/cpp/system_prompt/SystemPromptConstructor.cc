#pragma once
#include <assert.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

std::unordered_map<int, SystemPromptParams> SystemPromptConstructor::construct(const ft::GptInitParameter& params, EngineBase* engine, CacheManager* cache_manager) {
    std::unordered_map<int, SystemPromptParams> multi_task_prompt_args;
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

        // TODO(xinfei.sxf) consider tp, consider sp engine
        GenerateStreamPtr stream = engine->enqueue(generate_input);

        auto output1 = stream->nextOutput();
        assert(output1.ok());
        assert(output1.value().generate_outputs[0].aux_info.output_len == 1);
        assert(stream->finished());

        const auto& kv_cache = stream->kvCache();
        assert(kv_cache.k_ptr.size() == 1);
        assert(kv_cache.k_ptr[0].size() > 0);
        auto block_indices = cache_manager->convertAddrToIndex(kv_cache.k_ptr[0][0]);
        cache_manager->insertResidentCache(block_indices, tokens_id);
        multi_task_prompt_args[task_id] = SystemPromptParams(tokens_id, block_indices);
    }
    return multi_task_prompt_args;
}

} // namespace rtp_llm
