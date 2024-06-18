#pragma once

#include <assert.h>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "maga_transformer/cpp/dataclass/Query.h"

namespace rtp_llm {

struct SystemPromptParams {
    SystemPromptParams() {}
    SystemPromptParams(const std::vector<int>& _prompt_token, const std::vector<int>& _block_cache) {
        prompt_token = _prompt_token;
        prompt_length = prompt_token.size();
        block_cache = _block_cache;
    }
    
    std::vector<int>                prompt_token;
    size_t                          prompt_length;
    std::vector<int>                block_cache;
};

class SystemPrompt {
public:
    SystemPrompt(const std::unordered_map<std::string, SystemPromptParams>& prompt_map) : prompt_map_(prompt_map) {}

    SystemPromptParams getPromptParams(const GenerateConfig& generate_config) {
        const auto& task_id = generate_config.task_id;
        if (task_id != std::nullopt) {
            auto it = prompt_map_.find(task_id.value());
            if (it == prompt_map_.end()) {
                return {};
            }
            return it->second;
        } else {
            return {};
        }
    }

private:
    std::unordered_map<std::string, SystemPromptParams> prompt_map_;
};

}  // namespace rtp_llm
