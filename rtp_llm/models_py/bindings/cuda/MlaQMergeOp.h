#pragma once

#include <torch/extension.h>

namespace rtp_llm {

void MlaQMerge(torch::Tensor a, torch::Tensor b, torch::Tensor out);

}  // namespace rtp_llm
