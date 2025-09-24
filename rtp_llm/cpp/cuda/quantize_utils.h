#pragma once

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "trt_plugins/common/trtPluginsInterface.h"

namespace trt_cutlass = tensorrt_llm::kernels::cutlass_kernels;

namespace rtp_llm {

inline tensorrt_llm::kernels::cutlass_kernels::QuantType get_ft_quant_type(torch::ScalarType quant_type) {
    if (quant_type == torch::kInt8) {
        return trt_cutlass::QuantType::INT8_WEIGHT_ONLY;
    } else if (quant_type == at::ScalarType::QUInt4x2) {
        return trt_cutlass::QuantType::PACKED_INT4_WEIGHT_ONLY;
    } else {
        TORCH_CHECK(false, "Invalid quantization type");
    }
}

inline void check_quant_type_allowed(torch::ScalarType quant_type) {
    TORCH_CHECK(quant_type == torch::kInt8 || quant_type == at::ScalarType::QUInt4x2,
                "Must be int4 or int8 quantization");
}

inline std::vector<torch::Tensor> symmetric_quantize_helper(torch::Tensor     weight,
                                                            torch::ScalarType quant_type,
                                                            bool              return_unprocessed_quantized_tensor,
                                                            const int         arch) {
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

    auto _st = weight.scalar_type();
    TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
                "Invalid datatype. Weight must be FP16 or BF16");
    check_quant_type_allowed(quant_type);
    trt_cutlass::QuantType ft_quant_type = get_ft_quant_type(quant_type);

    const size_t num_experts = weight.dim() == 2 ? 1 : weight.size(0);
    const size_t num_rows    = weight.size(-2);
    const size_t num_cols    = weight.size(-1);

    const size_t bits_in_type      = get_bits_in_quant_type(ft_quant_type);
    const size_t bytes_per_out_col = num_cols * bits_in_type / 8;

    std::vector<long int> quantized_weight_shape;
    std::vector<long int> scale_shape;
    if (weight.dim() == 2) {
        quantized_weight_shape = {long(num_rows), long(bytes_per_out_col)};
        scale_shape            = {long(num_cols)};
    } else if (weight.dim() == 3) {
        quantized_weight_shape = {long(num_experts), long(num_rows), long(bytes_per_out_col)};
        scale_shape            = {long(num_experts), long(num_cols)};
    } else {
        TORCH_CHECK(false, "Invalid weight dimension. Weight must have dim 2 or 3");
    }

    torch::Tensor unprocessed_quantized_weight =
        torch::empty(quantized_weight_shape, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    torch::Tensor processed_quantized_weight = torch::empty_like(unprocessed_quantized_weight);

    torch::Tensor scales =
        torch::empty(scale_shape, torch::dtype(weight.dtype()).device(torch::kCPU).requires_grad(false));

    int8_t* unprocessed_quantized_weight_ptr = torch_ext::get_ptr<int8_t>(unprocessed_quantized_weight);
    int8_t* processed_quantized_weight_ptr   = torch_ext::get_ptr<int8_t>(processed_quantized_weight);

    if (weight.scalar_type() == at::ScalarType::Float) {
        trt_cutlass::symmetric_quantize<float, float>(processed_quantized_weight_ptr,
                                                      unprocessed_quantized_weight_ptr,
                                                      torch_ext::get_ptr<float>(scales),
                                                      torch_ext::get_ptr<const float>(weight),
                                                      {num_experts, num_rows, num_cols},
                                                      ft_quant_type,
                                                      arch);
    } else if (weight.scalar_type() == at::ScalarType::Half) {
        trt_cutlass::symmetric_quantize<half, half>(processed_quantized_weight_ptr,
                                                    unprocessed_quantized_weight_ptr,
                                                    torch_ext::get_ptr<half>(scales),
                                                    torch_ext::get_ptr<const half>(weight),
                                                    {num_experts, num_rows, num_cols},
                                                    ft_quant_type,
                                                    arch);
    } else if (weight.scalar_type() == at::ScalarType::BFloat16) {
        trt_cutlass::symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(processed_quantized_weight_ptr,
                                                                      unprocessed_quantized_weight_ptr,
                                                                      torch_ext::get_ptr<__nv_bfloat16>(scales),
                                                                      torch_ext::get_ptr<const __nv_bfloat16>(weight),
                                                                      {num_experts, num_rows, num_cols},
                                                                      ft_quant_type,
                                                                      arch);
    } else {
        TORCH_CHECK(false, "Invalid datatype. Weight must be BF16/FP16");
    }

    if (return_unprocessed_quantized_tensor) {
        return std::vector<torch::Tensor>{unprocessed_quantized_weight, processed_quantized_weight, scales};
    }

    return std::vector<torch::Tensor>{processed_quantized_weight, scales};
}

}  // namespace rtp_llm
