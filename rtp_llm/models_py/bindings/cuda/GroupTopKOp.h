#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/kernels/no_aux_tc_kernels.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"

namespace rtp_llm {

class GroupTopKOp {
public:
    GroupTopKOp();
    void forward(torch::Tensor&       topk_values,
                 torch::Tensor&       topk_indices,
                 torch::Tensor const& scores,
                 torch::Tensor const& scores_with_bias,
                 int64_t              n_group,
                 int64_t              topk_group,
                 int64_t              topk,
                 bool                 renormalize,
                 double               routed_scaling_factor);
};

void registerGroupTopKOp(const pybind11::module& m);
}  // namespace rtp_llm