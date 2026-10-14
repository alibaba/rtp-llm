#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/no_aux_tc_kernels.h"

namespace rtp_llm {

class GroupTopKOp {
public:
    GroupTopKOp();
    // router_logits: raw (pre-sigmoid) gate output in fp32/fp16/bf16; the fp32
    // sigmoid and the fp32 correction-bias add are fused into the CUDA kernels.
    // correction_bias: per-expert bias, fp32 (checkpoint-stored, e.g. Xing4 /
    // DeepSeek family) or the router_logits dtype.
    void forward(torch::Tensor&       topk_values,
                 torch::Tensor&       topk_indices,
                 torch::Tensor const& router_logits,
                 torch::Tensor const& correction_bias,
                 int64_t              n_group,
                 int64_t              topk_group,
                 int64_t              topk,
                 bool                 renormalize,
                 double               routed_scaling_factor);
};

void registerGroupTopKOp(const pybind11::module& m);
}  // namespace rtp_llm