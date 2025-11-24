#include "rtp_llm/cpp/kernels/atex/common.cuh"

namespace atex {
namespace impl {

Tensor launch_gate_silu_bf16(const Tensor& x);

Tensor launch_gate_silu_fp16(const Tensor& x);

}  // namespace impl
}  // namespace atex