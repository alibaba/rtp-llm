#include "rtp_llm/cpp/kernels/atex/common.cuh"

namespace atex {

const static fp32_t FP8_E4M3_MAX = +448.0f;
const static fp32_t FP8_E4M3_MIN = -448.0f;
const static fp32_t SCALE_MIN    = 1e-7f;

namespace impl {

std::tuple<Tensor, Tensor> launch_minmax_pertensor_quant_fp16_fp8e4m3(const Tensor& x);
std::tuple<Tensor, Tensor> launch_minmax_pertensor_quant_bf16_fp8e4m3(const Tensor& x);

}  // namespace impl
}  // namespace atex