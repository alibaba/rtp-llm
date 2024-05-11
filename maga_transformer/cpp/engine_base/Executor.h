#pragma once

#include "absl/status/statusor.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include <memory>

namespace rtp_llm {

class Executor {
public:
    Executor(){};
    virtual absl::Status addLoRA(const int64_t                                                           lora_id,
                                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) = 0;

    virtual absl::Status removeLoRA(const int64_t lora_id) = 0;
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams) = 0;
    ~Executor(){};

public:
    ft::DeviceBase*                       device_;
};

}  // namespace rtp_llm
