#pragma once

#include <torch/all.h>

namespace rtp_llm {

void fused_rmsnorm_fp8_quant(torch::Tensor input,
                             torch::Tensor weight,
                             torch::Tensor output_q,
                             torch::Tensor output_s,
                             double        norm_eps,
                             double        quant_eps,
                             double        min_8bit,
                             double        max_8bit);

void fused_rmsnorm_bf16_fp8_quant(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor output_y,
                                  torch::Tensor output_q,
                                  torch::Tensor output_s,
                                  double        norm_eps,
                                  double        quant_eps,
                                  double        min_8bit,
                                  double        max_8bit);

}  // namespace rtp_llm
