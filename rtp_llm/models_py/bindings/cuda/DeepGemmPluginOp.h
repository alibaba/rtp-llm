#pragma once

#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace rtp_llm {

torch::Tensor
deep_gemm_fp8(torch::Tensor lhs_bf16, torch::Tensor rhs_data, torch::Tensor rhs_scale, int user_deep_gemm_num_sm);

void deep_gemm_grouped_fp8_masked(torch::Tensor lhs_data,
                                  torch::Tensor lhs_scale,
                                  torch::Tensor rhs_data,
                                  torch::Tensor rhs_scale,
                                  torch::Tensor output,
                                  torch::Tensor masked_m,
                                  int           expected_m,
                                  int           user_deep_gemm_num_sm);

void registerDeepGemmPluginOp(py::module& m);

}  // namespace rtp_llm
