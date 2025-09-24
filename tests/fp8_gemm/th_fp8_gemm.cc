#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/memory_utils.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"
#include "rtp_llm/cpp/cuda/cutlass/interface.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/cuda/cublas/cublasAlgoMap.h"
#include "rtp_llm/cpp/cuda/cublas/cublasMMWrapper.h"
#include "rtp_llm/cpp/cuda/allocator_torch.h"
#include "rtp_llm/cpp/core/allocator.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace torch_ext {


template<typename T>
T getCudaValue(const T* ptr, int index) {
    T tmp;
    check_cuda_value(cudaMemcpy(&tmp, ptr + index, sizeof(T), cudaMemcpyDeviceToHost));
    return tmp;
}

namespace tkc = tensorrt_llm::kernels::cutlass_kernels;
namespace tc  = tensorrt_llm::cutlass_extensions;

template<typename T>
Tensor fp8_quant_gemm_helper(Tensor A, Tensor B, Tensor act_scale, Tensor w_scale) {
    auto stream        = at::cuda::getCurrentCUDAStream().stream();
    auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasLtHandle_t cublaslt_handle;
    cublasLtCreate(&cublaslt_handle);
    rtp_llm::Allocator<rtp_llm::AllocatorType::TH>* allocator = new rtp_llm::Allocator<rtp_llm::AllocatorType::TH>();
    allocator->setStream(stream);

    rtp_llm::cublasAlgoMap*   cublas_algo_map = new rtp_llm::cublasAlgoMap(GEMM_CONFIG);
    rtp_llm::cublasMMWrapper* cublas_wrapper  = new rtp_llm::cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map, new std::mutex(), allocator);
    cublas_wrapper->setFP8GemmConfig(CUDA_R_32F);

    int m = A.size(0);
    int n = B.size(0);
    int k = A.size(1);

    __nv_fp8_e4m3* input_tensor  = get_ptr<__nv_fp8_e4m3>(A);
    __nv_fp8_e4m3* weight_tensor = get_ptr<__nv_fp8_e4m3>(B);
    float*         a_scale       = get_ptr<float>(act_scale);
    float*         b_scale       = get_ptr<float>(w_scale);

    float input_scale  = getCudaValue<float>(a_scale, 0);
    float weight_scale = getCudaValue<float>(b_scale, 0);
    float alpha        = input_scale * weight_scale;

    auto output_tensor = torch::empty({m, n}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    float* output_buffer = get_ptr<float>(output_tensor);

    cublas_wrapper->Gemm(CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         n,
                         m,
                         k,
                         reinterpret_cast<const void*>(weight_tensor),
                         k,
                         reinterpret_cast<const void*>(input_tensor),
                         k,
                         output_buffer,
                         n,
                         alpha,
                         0.0f);

    return output_tensor;
}

template<typename T>
Tensor fp8_gemm_helper(Tensor A, Tensor B, Tensor act_scale, Tensor w_scale) {
    auto                                            stream    = at::cuda::getCurrentCUDAStream().stream();
    rtp_llm::Allocator<rtp_llm::AllocatorType::TH>* allocator = new rtp_llm::Allocator<rtp_llm::AllocatorType::TH>();
    allocator->setStream(stream);

    int m = A.size(0);
    int n = B.size(0);
    int k = A.size(1);

    T*     input_tensor = get_ptr<T>(A);
    Tensor quanted_input_tensor =
        torch::empty({m, k}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    __nv_fp8_e4m3* quanted_input_fp8 = get_ptr<__nv_fp8_e4m3>(quanted_input_tensor);
    float*         a_scale           = get_ptr<float>(act_scale);
  rtp_llm::invokeQuantizeMatrix(
        quanted_input_fp8, a_scale, input_tensor, m * k, m, tensorrt_llm::common::QuantizeMode::PER_TENSOR, stream);
    return fp8_quant_gemm_helper<T>(quanted_input_tensor, B, act_scale, w_scale);
}

Tensor fp8_gemm(Tensor A, Tensor B, Tensor act_scale, Tensor w_scale) {
    return fp8_gemm_helper<half>(A, B, act_scale, w_scale);
}

Tensor fp8_quant_gemm(Tensor A, Tensor B, Tensor act_scale, Tensor w_scale) {
    return fp8_quant_gemm_helper<half>(A, B, act_scale, w_scale);
}

TORCH_LIBRARY(fp8_gemm_ops, m) {
    m.def("fp8_gemm", fp8_gemm);
    m.def("fp8_quant_gemm", fp8_quant_gemm);
}

}  // namespace torch_ext
