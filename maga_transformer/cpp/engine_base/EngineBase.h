#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class EngineBase {
public:
    EngineBase(const EngineInitParams& params);
    virtual ~EngineBase() {}

    static void initDevices(const EngineInitParams& params);
    virtual absl::Status addLoRA(const int64_t                                                           lora_id,
                                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) = 0;

    virtual absl::Status removeLoRA(const int64_t lora_id) = 0;

    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;
    virtual absl::Status stop()                                           = 0;

protected:
    ft::DeviceBase*                       device_;
};

}  // namespace rtp_llm
