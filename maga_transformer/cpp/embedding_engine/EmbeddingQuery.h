#pragma once
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace ft = fastertransformer;

namespace rtp_llm {

class EmbeddingInput {
public:
    explicit EmbeddingInput(const std::shared_ptr<ft::Buffer>&   token_ids,
                            const std::shared_ptr<ft::Buffer>&   token_type_ids,
                            const std::shared_ptr<ft::Buffer>&   input_lengths,
                            const int64_t                        total_length,
                            int64_t                              request_id,
                            std::optional<MultimodalFeature> multimodal_features = std::nullopt);

    std::shared_ptr<ft::Buffer>         token_ids;
    std::shared_ptr<ft::Buffer>         token_type_ids;
    std::shared_ptr<ft::Buffer>         input_lengths;
    int64_t                             total_length;
    int64_t                             request_id;
    std::optional<MultimodalFeature>    multimodal_features;

    void checkVaild();
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateInput {"
                     << ", input_ids: " << token_ids->debugString()
                     << ", token_type_ids: " << token_type_ids->debugString()
                     << "}";
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

    bool                                             isTensor;
    std::optional<at::Tensor>                        t;
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
    void setError(const std::string& error) {
        this->error_info.has_error     = true;
        this->error_info.error_message = error;
    }

    TypedOutput output;
    ErrorInfo   error_info;
};

}  // namespace rtp_llm
