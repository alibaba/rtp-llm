#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

// GEMM函数声明
void gemm(at::Tensor& output, at::Tensor& input, at::Tensor& weight);