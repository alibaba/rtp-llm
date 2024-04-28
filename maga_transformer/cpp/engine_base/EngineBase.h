#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"

namespace rtp_llm {

class EngineBase {
public:
    virtual ~EngineBase() {}
    virtual void addLoRA(const int64_t                                                           lora_id,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) = 0;

    virtual void removeLoRA(const int64_t lora_id) = 0;

    virtual absl::Status enqueue(std::shared_ptr<GenerateStream>& stream) = 0;
    virtual absl::Status stop()                                           = 0;
    virtual const ResourceContext& resourceContext() const                = 0;
};

}  // namespace rtp_llm
