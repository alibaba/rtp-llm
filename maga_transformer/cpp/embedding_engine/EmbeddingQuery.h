#pragma once
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

#include <assert.h>
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
                            int64_t                              request_id);

    std::shared_ptr<ft::Buffer>         token_ids;
    std::shared_ptr<ft::Buffer>         token_type_ids;
    std::shared_ptr<ft::Buffer>         input_lengths;
    int64_t                             request_id;    
    int64_t                             total_length;

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

class EmbeddingOutput {
public:
    void setOutput(fastertransformer::BufferPtr& model_outputs, int64_t length);
    void setError(const std::string& error);

    ft::ConstBufferPtr output;
    int64_t            input_length;
    ErrorInfo          error_info;
};


}  // namespace rtp_llm
