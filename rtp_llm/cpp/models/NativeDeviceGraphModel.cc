#include "rtp_llm/cpp/models/NativeDeviceGraphModel.h"

namespace rtp_llm {
GptModelOutputs NativeDeviceGraphModel::forward(const GptModelInputs& inputs) {
    return graph_runner_->run(
        inputs.input_lengths->shape()[0] - inputs.sequence_lengths->shape()[0],
        inputs.sequence_lengths->shape()[0],
        inputs,
        std::function<GptModelOutputs(GptModelInputs)>([&](auto inputs) { return GptModel::forward(inputs); }));
}
}  // namespace rtp_llm