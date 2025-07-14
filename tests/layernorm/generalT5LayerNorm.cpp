#include "rtp_llm/cpp/kernels/rmsnormKernels.h"

#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

namespace unittest {

class T5LayerNormOp: public torch::jit::CustomClassHolder {
public:
    T5LayerNormOp(double eps): eps(eps) {};

    torch::Tensor forward(torch::Tensor input, torch::Tensor gamma);
    torch::Tensor forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma);
    torch::Tensor
    stride_forward(torch::Tensor input, torch::Tensor gamma, int64_t d_model, int64_t offset, int64_t stride);

private:
    int64_t eps;
};

torch::Tensor T5LayerNormOp::forward(torch::Tensor input, torch::Tensor gamma) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    torch::Tensor output = torch::zeros_like(input);
    rtp_llm::invokeGeneralRmsNorm<float, int8_t>((float*)output.data_ptr(),
                                                 (float*)input.data_ptr(),
                                                 (float*)gamma.data_ptr(),
                                                 (float*)nullptr,
                                                 (float)eps,
                                                 (int)batch_size,
                                                 (int)d_model,
                                                 stream);
    return output;
}

torch::Tensor T5LayerNormOp::forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    float* scale = input_scale.data_ptr<float>();

    torch::Tensor output = torch::zeros_like(input);
#ifdef ENABLE_FP8
    __nv_fp8_e4m3* quant_output = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    rtp_llm::invokeGeneralRmsNorm<float, __nv_fp8_e4m3>((float*)output.data_ptr(),
                                                        (float*)input.data_ptr(),
                                                        (float*)gamma.data_ptr(),
                                                        (float*)nullptr,
                                                        (float)eps,
                                                        (int)batch_size,
                                                        (int)d_model,
                                                        stream,
                                                        scale,
                                                        nullptr,
                                                        quant_output);
#endif
    return output;
}

torch::Tensor T5LayerNormOp::stride_forward(
    torch::Tensor input, torch::Tensor gamma, int64_t d_model, int64_t offset, int64_t stride) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto norm_size  = gamma.size(0);
    rtp_llm::invokeRmsNormWithStride((float*)input.data_ptr() + offset,
                                     (int)stride,
                                     (float*)input.data_ptr() + offset,
                                     (int)stride,
                                     (float*)gamma.data_ptr(),
                                     (float*)nullptr,
                                     (float)eps,
                                     (int)batch_size,
                                     (int)d_model,
                                     (int)norm_size,
                                     stream);
    auto input_slice = input.slice(1, offset, offset + d_model);
    return input_slice;
}

}  // namespace unittest

static auto T5LayerNormTHS = torch::jit::class_<unittest::T5LayerNormOp>("unittest", "T5LayerNormOp")
                                 .def(torch::jit::init<double>())
                                 .def("forward", &unittest::T5LayerNormOp::forward)
                                 .def("forward_fp8", &unittest::T5LayerNormOp::forward_fp8)
                                 .def("stride_forward", &unittest::T5LayerNormOp::stride_forward);