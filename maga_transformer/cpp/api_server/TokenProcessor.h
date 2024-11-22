#pragma once

#include <torch/python.h>

#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/api_server/TokenizerEncodeResponse.h"

namespace rtp_llm {

class TokenProcessor {
public:
    TokenProcessor(py::object token_processor);
    virtual ~TokenProcessor() = default;

public:
    // virtual for test
    virtual std::string                              decode(const std::vector<int>& token_ids);
    virtual std::vector<int>                         encode(const std::string& prompt);
    virtual std::shared_ptr<TokenizerEncodeResponse> tokenizer(const std::string& prompt);
    static std::string formatResponse(const std::string& generate_texts, const GenerateOutputs* generate_outputs);

private:
    // TODO: change to tokenizer wrapper
    py::object token_processor_;
};

}  // namespace rtp_llm
