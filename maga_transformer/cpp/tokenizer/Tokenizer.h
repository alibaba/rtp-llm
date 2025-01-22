#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace rtp_llm {

class Tokenizer {
public:
    Tokenizer(py::object tokenizer): tokenizer_(tokenizer) {}
    virtual ~Tokenizer() {}

public:
    // `virtual` for test
    virtual bool             isPreTrainedTokenizer();
    virtual int              getEosTokenId();
    virtual std::string      decode(const std::vector<int>& ids);
    virtual std::vector<int> encode(const std::string& tokens);

private:
    py::object tokenizer_;
};

}  // namespace rtp_llm
