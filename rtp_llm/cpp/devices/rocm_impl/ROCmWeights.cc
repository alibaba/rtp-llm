#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/rocm/quantizePreprocessors.h"
#include "rtp_llm/cpp/pybind/th_utils.h"

using namespace std;
using namespace torch_ext;
using torch::Tensor;

namespace rtp_llm {

#if USING_CK_INT4
// column major
torch::Tensor ROCmDevice::packInt8TensorToPackedInt4(torch::Tensor weight) {
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "Weight must be a int8 tensor");

    std::vector<long int> packed_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
        packed_tensor_size[i] = weight.size(i);
    }
    // packed_tensor_size[weight.dim() - 1] = (packed_tensor_size[weight.dim() - 1] + 1) / 2;
    packed_tensor_size[weight.dim() - 2] = (packed_tensor_size[weight.dim() - 2] + 1) / 2;
    std::reverse(packed_tensor_size.begin(), packed_tensor_size.end());

    Tensor packed_weight =
        torch::zeros(packed_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    Tensor  weight_transposed = weight.transpose(0, 1).contiguous();
    int8_t* unpacked_ptr      = get_ptr<int8_t>(weight_transposed);
    int8_t* packed_ptr        = get_ptr<int8_t>(packed_weight);

    for (int packed_idx = 0; packed_idx < packed_weight.numel(); ++packed_idx) {
        int8_t packed_int4s = 0;
        int8_t elt_0        = unpacked_ptr[2 * packed_idx + 0];
        int8_t elt_1        = unpacked_ptr[2 * packed_idx + 1];

        TORCH_CHECK(elt_0 >= -8 && elt_0 <= 7, "Value in unpacked tensor not in int4 range");
        TORCH_CHECK(elt_1 >= -8 && elt_1 <= 7, "Value in unpacked tensor not in int4 range");

        // packed_int4s |= ((elt_0 & 0x0F));
        // packed_int4s |= int8_t(elt_1 << 4);
        packed_int4s |= ((elt_1 & 0x0F));
        packed_int4s |= int8_t(elt_0 << 4);
        packed_int4s ^= (1 << 3);
        packed_int4s ^= (1 << 7);

        packed_ptr[packed_idx] = packed_int4s;
    }

    return packed_weight.transpose(0, 1);
}
#else
torch::Tensor ROCmDevice::packInt8TensorToPackedInt4(torch::Tensor weight) {
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
#endif

torch::Tensor ROCmDevice::preprocessWeightsForMixedGemm(torch::Tensor     column_major_quantized_weight,
                                                        torch::ScalarType quant_type,
                                                        const string&     arch) {
    auto _st = column_major_quantized_weight.scalar_type();
    CHECK_CPU(column_major_quantized_weight);
    TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");

    int columns = column_major_quantized_weight.size(1);
    int rows    = column_major_quantized_weight.size(0);

    std::vector<long int> permute_tensor_size(column_major_quantized_weight.dim());
    for (int i = 0; i < column_major_quantized_weight.dim(); ++i) {
        permute_tensor_size[i] = column_major_quantized_weight.size(i);
    }
    // permute_tensor_size[column_major_quantized_weight.dim() - 2] =
    // (permute_tensor_size[column_major_quantized_weight.dim() - 2] + 1) / 2;

    std::reverse(permute_tensor_size.begin(), permute_tensor_size.end());
    Tensor permute_weight =
        torch::zeros(permute_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* unpermute_ptr = get_ptr<int8_t>(column_major_quantized_weight);
    int8_t* permute_ptr   = get_ptr<int8_t>(permute_weight);

    for (int permute_idx = 0; permute_idx < permute_weight.numel(); permute_idx += 4) {
        int input[8];

        for (int k = 0; k < 4; k++) {
            int i4x2         = unpermute_ptr[permute_idx + k];
            input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
            input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
        }

        // permute 01234567->20643175
        {
            int hi   = input[2];
            int lo   = input[0];
            int i4x2 = (hi << 4) | lo;

            permute_ptr[permute_idx] = i4x2;
        }

        {
            int hi   = input[6];
            int lo   = input[4];
            int i4x2 = (hi << 4) | lo;

            permute_ptr[permute_idx + 1] = i4x2;
        }

        {
            int hi   = input[3];
            int lo   = input[1];
            int i4x2 = (hi << 4) | lo;

            permute_ptr[permute_idx + 2] = i4x2;
        }

        {
            int hi   = input[7];
            int lo   = input[5];
            int i4x2 = (hi << 4) | lo;

            permute_ptr[permute_idx + 3] = i4x2;
        }
    }
    return permute_weight.transpose(0, 1);
}

}  // namespace rtp_llm
