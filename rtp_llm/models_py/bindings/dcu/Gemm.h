#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {
// GEMM函数声明
void gemm(at::Tensor& output, at::Tensor& input, at::Tensor& weight);

}  // namespace rtp_llm
