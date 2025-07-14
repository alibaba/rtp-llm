#include "rtp_llm/cpp/kernels/layernorm_kernels.h"

#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

namespace unittest {

class LayerNormOp: public torch::jit::CustomClassHolder {
public:
    LayerNormOp(double eps): eps(eps) {};

    torch::Tensor forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor bias);
    torch::Tensor forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma, torch::Tensor bias);
    torch::Tensor stride_forward(
        torch::Tensor input, torch::Tensor gamma, torch::Tensor bias, int64_t d_model, int64_t offset, int64_t stride);

private:
    int64_t eps;
};

torch::Tensor LayerNormOp::forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor bias) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto          batch_size = input.size(0);
    auto          d_model    = input.size(1);
    torch::Tensor output     = torch::zeros_like(input);
    rtp_llm::invokeGeneralLayerNorm((float*)nullptr,
                                    (float*)output.data_ptr(),
                                    (float*)input.data_ptr(),
                                    (float*)gamma.data_ptr(),
                                    (float*)bias.data_ptr(),
                                    (float)eps,
                                    (int)batch_size,
                                    (int)d_model,
                                    stream);
    return output;
}

torch::Tensor
LayerNormOp::forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma, torch::Tensor bias) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    float* scale = input_scale.data_ptr<float>();

    torch::Tensor output = torch::zeros_like(input);
    // auto output = torch::empty(input.dims(), torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
#ifdef ENABLE_FP8
    __nv_fp8_e4m3* quant_output = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    rtp_llm::invokeGeneralLayerNorm((float*)nullptr,
                                    (float*)output.data_ptr(),
                                    (float*)input.data_ptr(),
                                    (float*)gamma.data_ptr(),
                                    (float*)bias.data_ptr(),
                                    (float)eps,
                                    (int)batch_size,
                                    (int)d_model,
                                    stream,
                                    true,
                                    scale,
                                    nullptr,
                                    quant_output,
                                    false);
#endif
    return output;
}

torch::Tensor LayerNormOp::stride_forward(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor bias, int64_t d_model, int64_t offset, int64_t stride) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto norm_size  = gamma.size(0);
    rtp_llm::invokeLayerNormWithStride((float*)input.data_ptr() + offset,
                                       (int)stride,
                                       (float*)input.data_ptr() + offset,
                                       (int)stride,
                                       (float*)gamma.data_ptr(),
                                       (float*)bias.data_ptr(),
                                       (float)eps,
                                       (int)batch_size,
                                       (int)d_model,
                                       (int)norm_size,
                                       stream);
    // std::cout << "after layernorm: " << input << std::endl;
    auto input_slice = input.slice(1, offset, offset + d_model);
    return input_slice;
}

}  // namespace unittest

static auto LayerNormTHS = torch::jit::class_<unittest::LayerNormOp>("unittest", "LayerNormOp")
                               .def(torch::jit::init<double>())
                               .def("forward", &unittest::LayerNormOp::forward)
                               .def("forward_fp8", &unittest::LayerNormOp::forward_fp8)
                               .def("stride_forward", &unittest::LayerNormOp::stride_forward);
