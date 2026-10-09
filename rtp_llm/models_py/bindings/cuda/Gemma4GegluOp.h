#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor gemma4_geglu_tanh_bf16(const at::Tensor& gate_up);

}  // namespace torch_ext
