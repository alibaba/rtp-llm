#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace torch_ext {

void fused_rmsnorm_fp8_quant(at::Tensor& input,
                             at::Tensor& weight,
                             at::Tensor& output_q,
                             at::Tensor& output_s,
                             double      norm_eps,
                             double      quant_eps,
                             double      fp8_min,
                             double      fp8_max);

void fused_rmsnorm_bf16_fp8_quant(at::Tensor& input,
                                  at::Tensor& weight,
                                  at::Tensor& output_y,
                                  at::Tensor& output_q,
                                  at::Tensor& output_s,
                                  double      norm_eps,
                                  double      quant_eps,
                                  double      fp8_min,
                                  double      fp8_max);

}  // namespace torch_ext
