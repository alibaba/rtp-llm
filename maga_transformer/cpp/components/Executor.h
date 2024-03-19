#pragma once

#include "maga_transformer/cpp/models/GptModel.h"

namespace rtp_llm {

class Executor {
public:
    Executor();
    virtual ~Executor();

private:
    GptModel gpt_model_;
};

} // namespace rtp_llm
