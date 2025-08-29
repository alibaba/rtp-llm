#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/MoETopkSoftmax.h"
#include "rtp_llm/cpp/kernels/moe_topk_softmax_kernels.h"
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
using namespace std;
namespace th = torch;
using namespace rtp_llm;
namespace torch_ext {
void moe_topk_softmax(at::Tensor& topk_weights,
                      at::Tensor& topk_indices,
                      at::Tensor& token_expert_indices,
                      at::Tensor& gating_output) {
    topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output);
}
}  // namespace torch_ext