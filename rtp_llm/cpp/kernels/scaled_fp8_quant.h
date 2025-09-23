#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#include "rtp_llm/cpp/cuda/cuda_utils.h"

namespace rtp_llm {
void per_tensor_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s, bool is_static);
}  // namespace rtp_llm
