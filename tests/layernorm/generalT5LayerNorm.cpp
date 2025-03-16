#include "src/fastertransformer/kernels/rmsnormKernels.h"

#include "src/fastertransformer/cuda/cuda_fp8_utils.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

namespace unittest {


class T5LayerNormOp: public torch::jit::CustomClassHolder {
public:
    T5LayerNormOp(double eps):eps(eps){};

    torch::Tensor forward(torch::Tensor input, torch::Tensor gamma);
    torch::Tensor forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma);
    torch::Tensor stride_forward(torch::Tensor input, torch::Tensor gamma,
                                 int64_t d_model, int64_t offset, int64_t stride);

private:
    int64_t eps;
};

torch::Tensor T5LayerNormOp::forward(torch::Tensor input, torch::Tensor gamma) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    torch::Tensor output = torch::zeros_like(input);
    fastertransformer::invokeGeneralRmsNorm<float, int8_t>((float*)output.data_ptr(), 
            (float*)input.data_ptr(), (float*)gamma.data_ptr(), (float*)nullptr, (float)eps, (int)batch_size, (int)d_model, stream);
    return output;
}

torch::Tensor T5LayerNormOp::forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    float *scale = input_scale.data_ptr<float>();

    torch::Tensor output = torch::zeros_like(input);
#ifdef ENABLE_FP8
    __nv_fp8_e4m3 *quant_output = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    fastertransformer::invokeGeneralRmsNorm<float, __nv_fp8_e4m3>((float*)output.data_ptr(), 
            (float*)input.data_ptr(), (float*)gamma.data_ptr(), (float*)nullptr, (float)eps, (int)batch_size, (int)d_model, stream, scale, nullptr, quant_output);
#endif
    return output;
}

torch::Tensor T5LayerNormOp::stride_forward(torch::Tensor input, torch::Tensor gamma,
                                            int64_t d_model, int64_t offset, int64_t stride) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto norm_size = gamma.size(0);
    torch::Tensor output     = torch::zeros_like(input);
    fastertransformer::invokeRmsNormWithStride((float*)input.data_ptr(),
                                                 (float*)gamma.data_ptr(),
                                                 (float*)nullptr,
                                                 (float)eps,
                                                 (int)batch_size,
                                                 (int)d_model,
                                                 (int)norm_size,
                                                 (int)stride,
                                                 (int)offset,
                                                 stream);
    auto input_slice = input.slice(1, offset, offset + d_model);
    return input_slice;
}

}  // namespace unittest

static auto T5LayerNormTHS =
    torch::jit::class_<unittest::T5LayerNormOp>("unittest", "T5LayerNormOp")
        .def(torch::jit::init<double>())
        .def("forward", &unittest::T5LayerNormOp::forward)
        .def("forward_fp8", &unittest::T5LayerNormOp::forward_fp8)
        .def("stride_forward",  &unittest::T5LayerNormOp::stride_forward);