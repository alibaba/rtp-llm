#pragma once

#include <string>
#include <vector>

#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace py = pybind11;

namespace rtp_llm {

class Tokenizer {
public:
    Tokenizer(py::object tokenizer): tokenizer_(tokenizer) {}
    virtual ~Tokenizer() {}

public:
    std::string toString();
    // `virtual` for test
    virtual bool               isPreTrainedTokenizer();
    virtual std::optional<int> getEosTokenId();
    virtual std::string        decode(const std::vector<int>& ids);
    virtual std::vector<int>   encode(const std::string& tokens);

    virtual std::vector<int> convertSelectTokens(const std::vector<std::string>& select_tokens_str, int vocab_size);
private:
    py::object tokenizer_;
};

}  // namespace rtp_llm
