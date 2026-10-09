#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor gemma4_softmax_8192_bf16(const at::Tensor& input, int64_t query_start, int64_t window_left);

}  // namespace torch_ext
