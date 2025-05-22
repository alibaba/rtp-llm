
#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/embedding_engine/handlers/HandlerBase.h"

namespace rtp_llm {

class LinearSoftmaxHandlerImpl: public IHandlerImpl {
public:
    LinearSoftmaxHandlerImpl(const rtp_llm::GptInitParameter& params);
    ~LinearSoftmaxHandlerImpl();
    void loadTensor(std::unordered_map<std::string, rtp_llm::ConstBufferPtr>& tensors) override;
    th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) override;
private:
    rtp_llm::DeviceBase*      device_;
    rtp_llm::ConstBufferPtr   weight_;
    rtp_llm::ConstBufferPtr   bias_;
    bool                 is_initalized_        = false;
};

class LinearSoftmaxHandler: public HandlerBase {
public:
    LinearSoftmaxHandler(const rtp_llm::GptInitParameter& params);
};

} // namespace rtp_llm
