#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
#include "rtp_llm/cpp/kernels/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/add_residual_kernels.h"

using namespace std;

namespace rtp_llm {

AddBiasOutput CudaDevice::addbias(const AddBiasParams& params) {
    BufferPtr   input     = params.input;
    const auto& bias      = params.bias;
    const auto  data_type = input->type();
    RTP_LLM_CHECK_WITH_INFO(params.inplace == true, "bias only support inplace now");
    RTP_LLM_CHECK_WITH_INFO(bias.dim() == 1, "bias dim should be 1");
    RTP_LLM_CHECK_WITH_INFO(input->dim() == 2, "input dim should be 2");
    RTP_LLM_CHECK_WITH_INFO(input->shape()[1] == bias.shape()[0], "input and bias last dim should equal");
    const size_t m = input->shape()[0];
    const size_t n = input->shape()[1];
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAddBias, input->data(), bias.data(), m, n, stream_);
    return {input};
}

}  // namespace rtp_llm
