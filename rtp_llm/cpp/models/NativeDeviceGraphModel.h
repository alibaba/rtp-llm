#pragma once
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/devices/NativeGraphRunnerBase.h"

namespace rtp_llm {
class NativeDeviceGraphModel: public GptModel {
public:
    NativeDeviceGraphModel(const GptModelInitParams& params):
        GptModel(params), graph_runner_(device_->getNativeGraphRunner()) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override;

private:
    std::shared_ptr<NativeGraphRunner> graph_runner_;
};
}  // namespace rtp_llm