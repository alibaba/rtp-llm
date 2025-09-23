#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/cuda/quantize_utils.h"

using namespace std;
using namespace torch_ext;
using torch::Tensor;

namespace rtp_llm {

torch::Tensor CudaDevice::packInt8TensorToPackedInt4(torch::Tensor weight) {
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be a int8 tensor");

    std::vector<long int> packed_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
        packed_tensor_size[i] = weight.size(i);
    }
    packed_tensor_size[weight.dim() - 1] = (packed_tensor_size[weight.dim() - 1] + 1) / 2;

    Tensor packed_weight =
        torch::zeros(packed_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* unpacked_ptr = get_ptr<int8_t>(weight);
    int8_t* packed_ptr   = get_ptr<int8_t>(packed_weight);

    for (int packed_idx = 0; packed_idx < packed_weight.numel(); ++packed_idx) {
        int8_t packed_int4s = 0;
        int8_t elt_0        = unpacked_ptr[2 * packed_idx + 0];
        int8_t elt_1        = unpacked_ptr[2 * packed_idx + 1];

        TORCH_CHECK(elt_0 >= -8 && elt_0 <= 7, "Value in unpacked tensor not in int4 range");
        TORCH_CHECK(elt_1 >= -8 && elt_1 <= 7, "Value in unpacked tensor not in int4 range");

        packed_int4s |= ((elt_0 & 0x0F));
        packed_int4s |= int8_t(elt_1 << 4);

        packed_ptr[packed_idx] = packed_int4s;
    }
    return packed_weight;
}

torch::Tensor CudaDevice::preprocessWeightsForMixedGemm(torch::Tensor     row_major_quantized_weight,
                                                        torch::ScalarType quant_type,
                                                        const string&     arch) {
    auto _st = row_major_quantized_weight.scalar_type();
    CHECK_CPU(row_major_quantized_weight);
    CHECK_CONTIGUOUS(row_major_quantized_weight);
    TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");
    check_quant_type_allowed(quant_type);
    TORCH_CHECK(row_major_quantized_weight.dim() == 2 || row_major_quantized_weight.dim() == 3,
                "Invalid dim. The dim of weight should be 2 or 3");

    trt_cutlass::QuantType ft_quant_type      = get_ft_quant_type(quant_type);
    const size_t           bits_in_quant_type = get_bits_in_quant_type(ft_quant_type);

    const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
    const size_t num_rows    = row_major_quantized_weight.size(-2);
    const size_t num_cols    = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

    Tensor  processed_tensor = torch::zeros_like(row_major_quantized_weight);
    int8_t* input_byte_ptr   = get_ptr<int8_t>(row_major_quantized_weight);
    int8_t* output_byte_ptr  = get_ptr<int8_t>(processed_tensor);

    int sm_version = arch.empty() ? get_sm() : atoi(arch.c_str());

    trt_cutlass::preprocess_weights_for_mixed_gemm(
        output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, ft_quant_type, sm_version);

    return processed_tensor;
}

std::vector<torch::Tensor> CudaDevice::symmetricQuantizeLastAxisOfBatchedMatrix(torch::Tensor     weight,
                                                                                torch::ScalarType quant_type,
                                                                                const string&     arch) {
    int sm_version = arch.empty() ? get_sm() : atoi(arch.c_str());
    return symmetric_quantize_helper(weight, quant_type, false, sm_version);
}

}  // namespace rtp_llm
