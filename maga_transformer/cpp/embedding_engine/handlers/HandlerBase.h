#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include <memory>

namespace rtp_llm {

class IHandlerImpl {
public:
    IHandlerImpl(const ft::GptInitParameter& params) : params_(params) {}
    virtual ~IHandlerImpl() {}
    virtual void loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) = 0;
    virtual th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) = 0;
protected:
    const ft::GptInitParameter params_;
};

class HandlerBase {
public:
    HandlerBase(const ft::GptInitParameter& params) : params_(params) {}
    virtual ~HandlerBase() {}    
    virtual void loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) { return handler_impl_->loadTensor(tensors); }
    virtual th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) {return handler_impl_->forward(hidden_states, input_lengths); }
protected:
    const ft::GptInitParameter params_;
    std::unique_ptr<IHandlerImpl> handler_impl_;
};
}  // namespace rtp_llm
