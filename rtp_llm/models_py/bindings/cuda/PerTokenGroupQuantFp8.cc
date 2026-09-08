#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/PerTokenGroupQuantFp8.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_8bit.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_8bit_checked.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_8bit_v2.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_8bit_v2_checked.h"
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

namespace {
void validate_quant_status(const at::Tensor&                         input,
                           const at::Tensor&                         output_q,
                           const at::Tensor&                         output_s,
                           const std::optional<torch::Tensor>& masked_m,
                           const at::Tensor&                         status) {
    TORCH_CHECK(status.is_cuda(), "status must be a CUDA tensor");
    TORCH_CHECK(status.scalar_type() == at::ScalarType::Int, "status must have dtype torch.int32");
    TORCH_CHECK(status.is_contiguous(), "status must be contiguous");
    TORCH_CHECK(status.dim() == 1 && status.size(0) == 1, "status must have shape [1]");
    TORCH_CHECK(status.device() == input.device(), "status must be on the same device as input");
    TORCH_CHECK(!status.is_alias_of(input), "status must not share storage with input");
    TORCH_CHECK(!status.is_alias_of(output_q), "status must not share storage with output_q");
    TORCH_CHECK(!status.is_alias_of(output_s), "status must not share storage with output_s");
    TORCH_CHECK(!masked_m.has_value() || !status.is_alias_of(*masked_m), "status must not share storage with masked_m");
}
}  // namespace

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

void per_token_group_quant_fp8_checked(at::Tensor& input,
                                       at::Tensor& output_q,
                                       at::Tensor& output_s,
                                       int64_t     group_size,
                                       double      eps,
                                       double      fp8_min,
                                       double      fp8_max,
                                       bool        scale_ue8m0,
                                       at::Tensor& status) {
    validate_quant_status(input, output_q, output_s, std::nullopt, status);
    per_token_group_quant_8bit_checked(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0, status);
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

void per_token_group_quant_fp8_v2_checked(at::Tensor&                         input,
                                          at::Tensor&                         output_q,
                                          at::Tensor&                         output_s,
                                          int64_t                             group_size,
                                          double                              eps,
                                          double                              fp8_min,
                                          double                              fp8_max,
                                          bool                                scale_ue8m0,
                                          bool                                fuse_silu_and_mul,
                                          const std::optional<torch::Tensor>& masked_m,
                                          at::Tensor&                         status) {
    validate_quant_status(input, output_q, output_s, masked_m, status);
    sgl_per_token_group_quant_8bit_v2_checked(input,
                                               output_q,
                                               output_s,
                                               group_size,
                                               eps,
                                               fp8_min,
                                               fp8_max,
                                               scale_ue8m0,
                                               fuse_silu_and_mul,
                                               masked_m,
                                               status);
}

}  // namespace torch_ext
