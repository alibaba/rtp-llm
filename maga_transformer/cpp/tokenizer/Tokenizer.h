#pragma once

#include <string>
#include <vector>

#include "maga_transformer/cpp/utils/PyUtils.h"

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

private:
    py::object tokenizer_;
};

}  // namespace rtp_llm
