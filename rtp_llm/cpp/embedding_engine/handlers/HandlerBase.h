#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include <memory>

namespace th = torch;
namespace rtp_llm {

class IHandlerImpl {
public:
    IHandlerImpl(const rtp_llm::GptInitParameter& params): params_(params) {}
    virtual ~IHandlerImpl() {}
    virtual void       loadTensor(std::unordered_map<std::string, rtp_llm::ConstBufferPtr>& tensors) = 0;
    virtual th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths)                   = 0;

protected:
    const rtp_llm::GptInitParameter params_;
};

class HandlerBase {
public:
    HandlerBase(const rtp_llm::GptInitParameter& params): params_(params) {}
    virtual ~HandlerBase() {}
    virtual void loadTensor(std::unordered_map<std::string, rtp_llm::ConstBufferPtr>& tensors) {
        return handler_impl_->loadTensor(tensors);
    }
    virtual th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) {
        return handler_impl_->forward(hidden_states, input_lengths);
    }

protected:
    const rtp_llm::GptInitParameter params_;
    std::unique_ptr<IHandlerImpl>   handler_impl_;
};

}  // namespace rtp_llm
