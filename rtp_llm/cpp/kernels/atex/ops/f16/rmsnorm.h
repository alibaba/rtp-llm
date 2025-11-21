#include "rtp_llm/cpp/kernels/atex/common.cuh"

namespace atex {
namespace impl {

Tensor launch_rmsnorm_fp16(const Tensor& x, const Tensor& weight, const float eps);

Tensor launch_rmsnorm_bf16(const Tensor& x, const Tensor& weight, const float eps);

std::tuple<Tensor, Tensor>
launch_skiprmsnorm_fp16(const Tensor& x, const Tensor& skip, const Tensor& weight, const float eps);

std::tuple<Tensor, Tensor>
launch_skiprmsnorm_bf16(const Tensor& x, const Tensor& skip, const Tensor& weight, const float eps);

}  // namespace impl
}  // namespace atex