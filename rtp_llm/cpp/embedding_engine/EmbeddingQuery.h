#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace rtp_llm {

class EmbeddingInput {
public:
    explicit EmbeddingInput(const std::shared_ptr<rtp_llm::Buffer>&         token_ids,
                            const std::shared_ptr<rtp_llm::Buffer>&         token_type_ids,
                            const std::shared_ptr<rtp_llm::Buffer>&         input_lengths,
                            const int64_t                                   total_length,
                            int64_t                                         request_id,
                            std::optional<MultimodalFeature>                multimodal_features = std::nullopt,
                            std::optional<std::shared_ptr<rtp_llm::Buffer>> input_embeddings    = std::nullopt);

    explicit EmbeddingInput(const torch::Tensor&             token_ids,
                            const torch::Tensor&             token_type_ids,
                            const torch::Tensor&             input_lengths,
                            int                              request_id,
                            std::optional<MultimodalFeature> multimodal_features = std::nullopt,
                            std::optional<torch::Tensor>     input_embeddings    = std::nullopt);

    std::shared_ptr<rtp_llm::Buffer>                token_ids;
    std::shared_ptr<rtp_llm::Buffer>                token_type_ids;
    std::shared_ptr<rtp_llm::Buffer>                input_lengths;
    int64_t                                         total_length;
    int64_t                                         request_id;
    std::optional<MultimodalFeature>                multimodal_features;
    std::optional<std::shared_ptr<rtp_llm::Buffer>> input_embeddings;

    void        checkVaild();
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateInput {"
                     << ", input_ids: " << token_ids->debugString()
                     << ", token_type_ids: " << token_type_ids->debugString()
                     << ", input_lengths: " << input_lengths->debugString() << ", total_length: " << total_length
                     << "}";
        if (input_embeddings.has_value()) {
            debug_string << ", input_embeddings: " << input_embeddings.value()->debugString();
        }
        return debug_string.str();
    }
};

class TypedOutput {
public:
    void setTensorOuput(torch::Tensor t) {
        isTensor = true;
        this->t  = t;
    }
    void setMapOutput(std::vector<std::map<std::string, at::Tensor>>& m) {
        isTensor  = false;
        this->map = std::move(m);
    }

    bool                                                          isTensor;
    std::optional<at::Tensor>                                     t;
    std::optional<std::vector<std::map<std::string, at::Tensor>>> map;
};

class EmbeddingOutput {
public:
    void setTensorOutput(torch::Tensor t) {
        output.setTensorOuput(t);
    }
    void setMapOutput(std::vector<std::map<std::string, torch::Tensor>>& m) {
        output.setMapOutput(m);
    }
    void setError(const std::string& error_msg) {
        error_info = ErrorInfo(ErrorCode::UNKNOWN_ERROR, error_msg);
    }

    TypedOutput output;
    ErrorInfo   error_info;
};

}  // namespace rtp_llm
