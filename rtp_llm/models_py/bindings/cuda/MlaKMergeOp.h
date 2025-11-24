#pragma once

#include <torch/extension.h>

namespace rtp_llm {

void MlaKMerge(torch::Tensor k_out, torch::Tensor k_nope, torch::Tensor k_pe);

}  // namespace rtp_llm
