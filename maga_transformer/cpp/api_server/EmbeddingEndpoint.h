#pragma once

#include <torch/python.h>
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class EmbeddingEndpoint {
public:
    EmbeddingEndpoint(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                      std::shared_ptr<MultimodalProcessor> mm_processor,
                      py::object                           py_render):
        py_render_(py_render), embedding_engine_(embedding_engine), mm_processor_(mm_processor) {}

private:
    py::object                           py_render_;
    std::shared_ptr<EmbeddingEngine>     embedding_engine_;
    std::shared_ptr<MultimodalProcessor> mm_processor_ = nullptr;
};

}  // namespace rtp_llm
