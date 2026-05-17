#include "rtp_llm/models_py/bindings/cuda/FusedRmsnormFp8QuantOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/fused_rmsnorm_fp8_quant.h"

namespace torch_ext {

void fused_rmsnorm_fp8_quant(at::Tensor& input,
                             at::Tensor& weight,
                             at::Tensor& output_q,
                             at::Tensor& output_s,
                             double      norm_eps,
                             double      quant_eps,
                             double      fp8_min,
                             double      fp8_max) {
    rtp_llm::fused_rmsnorm_fp8_quant(input, weight, output_q, output_s, norm_eps, quant_eps, fp8_min, fp8_max);
}

void fused_rmsnorm_bf16_fp8_quant(at::Tensor& input,
                                  at::Tensor& weight,
                                  at::Tensor& output_y,
                                  at::Tensor& output_q,
                                  at::Tensor& output_s,
                                  double      norm_eps,
                                  double      quant_eps,
                                  double      fp8_min,
                                  double      fp8_max) {
    rtp_llm::fused_rmsnorm_bf16_fp8_quant(
        input, weight, output_y, output_q, output_s, norm_eps, quant_eps, fp8_min, fp8_max);
}

}  // namespace torch_ext
