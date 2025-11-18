#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/PerTokenGroupQuantFp8.h"
#include "rtp_llm/cpp/kernels/per_token_group_quant_8bit.h"
#include "rtp_llm/cpp/kernels/per_token_group_quant_8bit_v2.h"
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

void per_token_group_quant_int8(at::Tensor& input,
                                at::Tensor& output_q,
                                at::Tensor& output_s,
                                int64_t     group_size,
                                double      eps,
                                double      int8_min,
                                double      int8_max,
                                bool        scale_ue8m0) {
    per_token_group_quant_8bit(input, output_q, output_s, group_size, eps, int8_min, int8_max, scale_ue8m0);
}

void per_token_group_quant_fp8(at::Tensor& input,
                               at::Tensor& output_q,
                               at::Tensor& output_s,
                               int64_t     group_size,
                               double      eps,
                               double      fp8_min,
                               double      fp8_max,
                               bool        scale_ue8m0) {
    per_token_group_quant_8bit(input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0);
}

void per_token_group_quant_fp8_v2(at::Tensor&                         input,
                                  at::Tensor&                         output_q,
                                  at::Tensor&                         output_s,
                                  int64_t                             group_size,
                                  double                              eps,
                                  double                              fp8_min,
                                  double                              fp8_max,
                                  bool                                scale_ue8m0,
                                  bool                                fuse_silu_and_mul,
                                  const std::optional<torch::Tensor>& masked_m) {
    sgl_per_token_group_quant_8bit_v2(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0, fuse_silu_and_mul, masked_m);
}

}  // namespace torch_ext
