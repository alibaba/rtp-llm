#pragma once

#include <vector>
#include "rtp_llm/cpp/embedding_engine/handlers/HandlerBase.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace torch_ext {

class EmbeddingHandlerOp {
private:
    std::unique_ptr<rtp_llm::HandlerBase> handler_ = nullptr;
public:
    EmbeddingHandlerOp() {}
    void setHandler(std::unique_ptr<rtp_llm::HandlerBase>& handler) {
        handler_ = std::move(handler);
    }

    void loadTensor(std::unordered_map<std::string, th::Tensor> weights) {
        if (!handler_) {
            throw std::runtime_error("handler is not set!");
        }

        std::unordered_map<std::string, rtp_llm::ConstBufferPtr> weights_buffer;
        for (auto& it : weights) {
            weights_buffer.emplace(it.first, rtp_llm::torchTensor2Buffer(it.second));
        }
        handler_->loadTensor(weights_buffer);
    }

    th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) {
        if (!handler_) {
            throw std::runtime_error("handler is not set!");
        }
        return handler_->forward(hidden_states, input_lengths);
    }

};

void registerEmbeddingHandler(py::module& m);

} // namespace ft
