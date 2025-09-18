#pragma once

#include <torch/extension.h>
#include <torch/all.h>

namespace torch_ext {

/**
 * @brief Quantize BF16 weight matrix to FP8 format using 128-element block processing
 *
 * @param input Input BF16 weight tensor, must be 2D tensor with last dimension divisible by 128
 * @param col_major_scale Whether to use column-major scale mode
 * @return std::pair<at::Tensor, at::Tensor> Returns quantized FP8 tensor and scale factor tensor
 */
std::pair<at::Tensor, at::Tensor> trt_fp8_quantize_128(const at::Tensor& input, bool col_major_scale = false);

/**
 * @brief Quantize BF16 weight matrix to FP8 format using 128-element block processing (in-place version)
 *
 * @param input Input BF16 weight tensor, must be 2D tensor with last dimension divisible by 128
 * @param output_q Output FP8 quantized tensor
 * @param output_s Output scale factor tensor
 * @param col_major_scale Whether to use column-major scale mode
 */
void trt_fp8_quantize_128_inplace(const at::Tensor& input,
                                  at::Tensor&       output_q,
                                  at::Tensor&       output_s,
                                  bool              col_major_scale = false);

/**
 * @brief Register TrtFp8QuantOp functions to Python module
 *
 * @param m Python module to register functions to
 */
void registerTrtFp8QuantOp(py::module& m);

}  // namespace torch_ext
