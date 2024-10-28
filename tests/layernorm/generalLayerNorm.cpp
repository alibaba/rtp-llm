#include "src/fastertransformer/kernels/layernorm_kernels.h"

#include "src/fastertransformer/cuda/cuda_fp8_utils.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

namespace unittest {


class LayerNormOp: public torch::jit::CustomClassHolder {
public:
    LayerNormOp(double eps):eps(eps){};

    torch::Tensor forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor bias);
  torch::Tensor forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma, torch::Tensor bias);

private:
    int64_t eps;
};

  torch::Tensor LayerNormOp::forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor bias) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);
    torch::Tensor output     = torch::zeros_like(input);
    fastertransformer::invokeGeneralLayerNorm((float*)nullptr,
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

  torch::Tensor LayerNormOp::forward_fp8(torch::Tensor input, torch::Tensor input_scale, torch::Tensor gamma, torch::Tensor bias) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    float *scale = input_scale.data_ptr<float>();

    torch::Tensor output = torch::zeros_like(input);
    // auto output = torch::empty(input.dims(), torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
#ifdef ENABLE_FP8
    __nv_fp8_e4m3 *quant_output = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr());
    fastertransformer::invokeGeneralLayerNorm(
            (float*)nullptr,
            (float*)output.data_ptr(), 
            (float*)input.data_ptr(), 
            (float*)gamma.data_ptr(), 
            (float*)bias.data_ptr(), 
            (float)eps, 
            (int)batch_size, (int)d_model, stream, true, scale, nullptr, quant_output, false);
#endif
    return output;
}

}  // namespace unittest

static auto LayerNormTHS =
    torch::jit::class_<unittest::LayerNormOp>("unittest", "LayerNormOp")
        .def(torch::jit::init<double>())
        .def("forward", &unittest::LayerNormOp::forward)
        .def("forward_fp8", &unittest::LayerNormOp::forward_fp8);
