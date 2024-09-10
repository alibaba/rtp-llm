#include "src/fastertransformer/devices/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"

using namespace std;

namespace fastertransformer {

AddBiasOutput CudaDevice::addbias(const AddBiasParams& params) {
    BufferPtr input = params.input;
    const auto& bias = params.bias;
    const auto data_type = input->type();
    FT_CHECK_WITH_INFO(params.inplace == true, "bias only support inplace now");
    FT_CHECK_WITH_INFO(bias.dim() == 1, "bias dim should be 1");
    FT_CHECK_WITH_INFO(input->dim() == 2, "input dim should be 2");
    FT_CHECK_WITH_INFO(input->shape()[1] == bias.shape()[0], "input and bias last dim should equal");
    const size_t m = input->shape()[0];
    const size_t n = input->shape()[1];
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeAddBias, input->data(), bias.data(), m, n, stream_);
    return {input};
}


} // namespace fastertransformer
