#pragma once

#include "absl/status/statusor.h"
#include "maga_transformer/cpp/batch_stream_processor/BatchStreamProcessor.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include <memory>

namespace rtp_llm {

class Executor {
public:
    Executor(){};
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams) = 0;
    ~Executor(){};

public:
    ft::DeviceBase*                       device_;
    std::unique_ptr<GptModel>             model_;
    std::unique_ptr<Sampler>              sampler_;
    std::unique_ptr<BatchStreamProcessor> batch_stream_processor_;
};

}  // namespace rtp_llm
