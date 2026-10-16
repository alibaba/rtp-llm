#include <torch/extension.h>

void sm90_g32_gemm_cuda(const at::Tensor& a, const at::Tensor& a_scale,
                       const at::Tensor& b, const at::Tensor& b_scale,
                       const at::Tensor& out, int config);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("fp8_gemm_nt", &sm90_g32_gemm_cuda,
               "Isolated SGLang DeepGEMM SM90 FP8 group32 dense candidate");
}
