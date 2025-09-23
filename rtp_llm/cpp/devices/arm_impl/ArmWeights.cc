#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/devices/arm_impl/gemm_opt/ArmGemmKernel.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace torch_ext;
using torch::Tensor;

namespace rtp_llm {

torch::Tensor ArmCpuDevice::packInt8TensorToPackedInt4(torch::Tensor weight) {
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

torch::Tensor ArmCpuDevice::preprocessWeightsForMixedGemm(torch::Tensor     row_major_quantized_weight,
                                                          torch::ScalarType quant_type,
                                                          const string&     arch) {
    return row_major_quantized_weight;
}

torch::Tensor ArmCpuDevice::preprocessWeightScale(torch::Tensor qweight, torch::Tensor scales) {
    auto qweightBuffer = torchTensor2Buffer(qweight);
    auto scaleBuffer   = torchTensor2Buffer(scales);
    auto retBuffer     = prepareGemmOptForGPTQInt4(qweightBuffer, scaleBuffer, "");

    return Buffer2torchTensor(*retBuffer, false);
}

}  // namespace rtp_llm
