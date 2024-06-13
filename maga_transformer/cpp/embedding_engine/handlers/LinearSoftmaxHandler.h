
#pragma once

#include "absl/status/statusor.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"

namespace rtp_llm {

class LinearSoftmaxHandlerImpl: public IHandlerImpl {
public:
    LinearSoftmaxHandlerImpl(const ft::GptInitParameter& params);
    ~LinearSoftmaxHandlerImpl();
    void loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) override;
    th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) override;
private:
    ft::DeviceBase*      device_;
    ft::ConstBufferPtr   weight_;
    ft::ConstBufferPtr   bias_;
    bool                 is_initalized_        = false;
};

class LinearSoftmaxHandler: public HandlerBase {
public:
    LinearSoftmaxHandler(const ft::GptInitParameter& params);
};

} // namespace rtp_llm
