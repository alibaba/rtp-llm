#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

struct SystemPromptParams {
    SystemPromptParams() {}
    SystemPromptParams(const std::vector<int>& prompt_tokens, const std::vector<int>& block_ids):
        prompt_tokens(prompt_tokens), block_ids(block_ids) {}

    std::vector<int> prompt_tokens;
    std::vector<int> block_ids;
};

class SystemPrompt {
public:
    SystemPrompt(const std::unordered_map<std::string, SystemPromptParams>& prompt_map): prompt_map_(prompt_map) {}

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
