#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/pybind/th_utils.h"
#include "rtp_llm/cpp/devices/arm_impl/gemm_opt/ArmGemmKernel.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace torch_ext;
using torch::Tensor;

namespace rtp_llm {

torch::Tensor ArmCpuDevice::preprocessWeightScale(torch::Tensor qweight, torch::Tensor scales) {
    auto qweightBuffer = torchTensor2Buffer(qweight);
    auto scaleBuffer   = torchTensor2Buffer(scales);
    auto retBuffer     = prepareGemmOptForGPTQInt4(qweightBuffer, scaleBuffer, "");

    return Buffer2torchTensor(*retBuffer, false);
}

}  // namespace rtp_llm
