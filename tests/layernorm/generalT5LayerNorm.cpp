#include "src/fastertransformer/kernels/rmsnormKernels.h"

#include "src/fastertransformer/cuda/cuda_fp8_utils.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>

namespace unittest {


class T5LayerNormOp: public torch::jit::CustomClassHolder {
public:
    T5LayerNormOp(double eps):eps(eps){};

    torch::Tensor forward(torch::Tensor input, torch::Tensor gamma);

private:
    int64_t eps;
};

torch::Tensor T5LayerNormOp::forward(torch::Tensor input, torch::Tensor gamma) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto batch_size = input.size(0);
    auto d_model    = input.size(1);

    torch::Tensor output = torch::zeros_like(input);
    fastertransformer::invokeGeneralRmsNorm((float*)output.data_ptr(), 
            (float*)input.data_ptr(), (float*)gamma.data_ptr(), (float*)nullptr, (float)eps, (int)batch_size, (int)d_model, stream);
    return output;
}

}  // namespace unittest

static auto T5LayerNormTHS =
    torch::jit::class_<unittest::T5LayerNormOp>("unittest", "T5LayerNormOp")
        .def(torch::jit::init<double>())
        .def("forward", &unittest::T5LayerNormOp::forward);