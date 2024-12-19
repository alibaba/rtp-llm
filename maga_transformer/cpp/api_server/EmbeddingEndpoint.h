#pragma once

#include "maga_transformer/cpp/utils/PyUtils.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class EmbeddingEndpoint {
public:
    enum EmbeddingType {
        DENSE,
        SPARSE,
        COLBERT
    };
    EmbeddingEndpoint(const std::shared_ptr<EmbeddingEngine>&     embedding_engine,
                      const std::shared_ptr<MultimodalProcessor>& mm_processor,
                      py::object                                  custom_module):
        custom_module_(custom_module), embedding_engine_(embedding_engine), mm_processor_(mm_processor) {}
    virtual ~EmbeddingEndpoint() = default;

public:
    // `virtual` for test
    virtual std::pair<std::string, std::optional<std::string>>
    handle(const std::string& body, std::optional<EmbeddingEndpoint::EmbeddingType> type);

private:
    std::string embeddingTypeToString(EmbeddingType type);
    std::optional<MultimodalFeature> getMultimodalFeature(py::object py_mm_inputs,
                                                          th::Tensor& token_ids);
    std::string getAsyncResult(py::object loop, py::object coro);
private:
    py::object                           custom_module_;
    std::shared_ptr<EmbeddingEngine>     embedding_engine_;
    std::shared_ptr<MultimodalProcessor> mm_processor_ = nullptr;
};

}  // namespace rtp_llm
