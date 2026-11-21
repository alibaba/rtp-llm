#pragma once

#include <ATen/core/Tensor.h>
#include <vector>

namespace rtp_llm {

std::vector<at::Tensor> packPreprocessCpuTensors(const std::vector<at::Tensor>& tensors);

}  // namespace rtp_llm
