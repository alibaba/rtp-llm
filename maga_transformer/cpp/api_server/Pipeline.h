#pragma once

#include <torch/python.h>

#include "maga_transformer/cpp/dataclass/Query.h"

namespace rtp_llm {

class Pipeline {
public:
    Pipeline(py::object token_processor): token_processor_(token_processor) {}

private:
    // TODO: change to tokenizer wrapper
    py::object token_processor_;
};

}  // namespace rtp_llm
