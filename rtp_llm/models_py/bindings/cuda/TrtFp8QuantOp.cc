#include "rtp_llm/models_py/bindings/cuda/TrtFp8QuantOp.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <c10/cuda/CUDAGuard.h>

namespace torch_ext {

namespace {
void validate_input_tensor(const at::Tensor& input) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    // not support 3d tensor yet
    CHECK_DIM(2, input);
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16, "Input tensor must be BF16 type, got ", input.scalar_type());
    // currently not support padding for tma
    int64_t first_dim = input.size(0);
    int64_t last_dim  = input.size(1);
    TORCH_CHECK(last_dim % 128 == 0, "Last dimension must be divisible by 128, got ", last_dim);
    // TORCH_CHECK(first_dim % 4 == 0, "First dimension must be divisible by 4, got ", first_dim);
    TORCH_CHECK(input.numel() > 0, "Input tensor cannot be empty");
}

std::pair<at::Tensor, at::Tensor> create_output_tensors(const at::Tensor& input, bool col_major_scale) {

    int64_t dim0 = input.size(0);
    int64_t dim1 = input.size(1);

    // Create FP8 output tensor with same shape as input
    auto output_q = torch::empty({dim0, dim1},
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat8_e4m3fn)
                                     .device(input.device())
                                     .memory_format(torch::MemoryFormat::Contiguous));

    // Create scale factor tensor
    at::Tensor output_s;
    if (col_major_scale) {
        // Column-major: shape is (dim1/128, dim0)
        output_s = torch::empty({dim1 / 128, dim0},
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(input.device())
                                    .memory_format(torch::MemoryFormat::Contiguous));
    } else {
        // Row-major: shape is (dim0, dim1/128)
        output_s = torch::empty({dim0, dim1 / 128},
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(input.device())
                                    .memory_format(torch::MemoryFormat::Contiguous));
    }

    return {output_q, output_s};
}
}  // namespace

std::pair<at::Tensor, at::Tensor> trt_fp8_quantize_128(const at::Tensor& input, bool col_major_scale) {
    validate_input_tensor(input);

    auto [output_q, output_s] = create_output_tensors(input, col_major_scale);

    trt_fp8_quantize_128_inplace(input, output_q, output_s, col_major_scale);

    return {output_q, output_s};
}

#ifdef ENABLE_FP8
void trt_fp8_quantize_128_inplace(const at::Tensor& input,
                                  at::Tensor&       output_q,
                                  at::Tensor&       output_s,
                                  bool              col_major_scale) {

    validate_input_tensor(input);

    TORCH_CHECK(output_q.is_cuda(), "Output quantized tensor must be CUDA tensor");
    TORCH_CHECK(output_s.is_cuda(), "Output scale tensor must be CUDA tensor");
    TORCH_CHECK(output_q.is_contiguous(), "Output quantized tensor must be contiguous");
    TORCH_CHECK(output_s.is_contiguous(), "Output scale tensor must be contiguous");

    int64_t dim0 = input.size(0);
    int64_t dim1 = input.size(1);

    TORCH_CHECK(output_q.size(0) == dim0 && output_q.size(1) == dim1,
                "Output quantized tensor shape mismatch. Expected (",
                dim0,
                ", ",
                dim1,
                "), got (",
                output_q.size(0),
                ", ",
                output_q.size(1),
                ")");

    if (col_major_scale) {
        TORCH_CHECK(output_s.size(0) == dim1 / 128 && output_s.size(1) == dim0,
                    "Output scale tensor shape mismatch for col_major_scale. Expected (",
                    dim1 / 128,
                    ", ",
                    dim0,
                    "), got (",
                    output_s.size(0),
                    ", ",
                    output_s.size(1),
                    ")");
    } else {
        TORCH_CHECK(output_s.size(0) == dim0 && output_s.size(1) == dim1 / 128,
                    "Output scale tensor shape mismatch for row_major_scale. Expected (",
                    dim0,
                    ", ",
                    dim1 / 128,
                    "), got (",
                    output_s.size(0),
                    ", ",
                    output_s.size(1),
                    ")");
    }

    TORCH_CHECK(output_q.scalar_type() == torch::kFloat8_e4m3fn,
                "Output quantized tensor must be FP8 E4M3 type, got ",
                output_q.scalar_type());
    TORCH_CHECK(output_s.scalar_type() == torch::kFloat32,
                "Output scale tensor must be float32 type, got ",
                output_s.scalar_type());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    try {
        rtp_llm::invokeComputeFP8Quantize128(reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                                                          reinterpret_cast<float*>(output_s.data_ptr()),
                                                          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                                                          dim0,
                                                          dim1,
                                                          input.numel(),
                                                          col_major_scale,
                                                          stream);

        // Check CUDA errors
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in invokeComputeFP8Quantize128: ")
                                     + cudaGetErrorString(cuda_error));
        }

    } catch (const std::exception& e) {
        TORCH_CHECK(false, "Error in trt_fp8_quantize_128_inplace: ", e.what());
    }
}
#else
void trt_fp8_quantize_128_inplace(const at::Tensor& input,
                                  at::Tensor&       output_q,
                                  at::Tensor&       output_s,
                                  bool              col_major_scale) {
    TORCH_CHECK(false, "FP8 quantization is not enabled");
}
#endif

void registerTrtFp8QuantOp(py::module& m) {
    m.def("trt_fp8_quantize_128",
          &trt_fp8_quantize_128,
          "Quantize BF16 weight matrix to FP8 format using 128-element block processing",
          py::arg("input"),
          py::arg("col_major_scale") = false);

    m.def("trt_fp8_quantize_128_inplace",
          &trt_fp8_quantize_128_inplace,
          "Quantize BF16 weight matrix to FP8 format using 128-element block processing (in-place version)",
          py::arg("input"),
          py::arg("output_q"),
          py::arg("output_s"),
          py::arg("col_major_scale") = false);
}

}  // namespace torch_ext
