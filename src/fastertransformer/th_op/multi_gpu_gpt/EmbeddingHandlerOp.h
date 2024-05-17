#pragma once

#include <vector>
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"


namespace torch_ext {

class EmbeddingHandlerOp: public th::jit::CustomClassHolder {
private: 
    std::unique_ptr<rtp_llm::HandlerBase> handler_ = nullptr;
public:
    EmbeddingHandlerOp() {}
    void setHandler(std::unique_ptr<rtp_llm::HandlerBase>& handler) {
        handler_ = std::move(handler);
    }

    rtp_llm::HandlerBase& getHandler() {
        if (handler_.get() == nullptr) {
            throw std::runtime_error("handler ptr should not be nullptr");
        }
        return *handler_;
    }
};

} // namespace ft
