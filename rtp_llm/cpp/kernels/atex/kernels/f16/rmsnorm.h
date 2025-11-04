#include "common.cuh"

namespace atex {
namespace impl {

Tensor launch_rmsnorm_fp16(const Tensor& x, const Tensor& weight, const float eps);

}
}  // namespace atex