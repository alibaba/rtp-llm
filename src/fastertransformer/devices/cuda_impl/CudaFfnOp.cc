#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"


using namespace std;

namespace fastertransformer {

/// @brief   feed forward neural network ops
/// @details output = Gemm(Act(Gemm(input, W1) + b1), W2) + b2
///          input(array) : [m, k]
///          W1(array) : [k, n]
///          b1(array) : [m, n]
///          W2(array) : [m, n]
///          b2(array)
///          output(array)
void CudaDevice::ffnLayer(const FfnLayerParams& params) {

    size_t token_num = params.input.shape()[0];
    size_t inter_size = params.gate_weight.shape()[1];
    auto gate_buf = allocateBuffer({
        params.input.type(), {token_num, inter_size}, AllocationType::DEVICE}, {});
    gemm({params.input, params.gate_weight, *gate_buf});

    auto up_buf = allocateBuffer({
        params.input.type(), {token_num, inter_size}, AllocationType::DEVICE}, {});
    
    gemm({params.input, params.up_weight, *up_buf});

    activation({params.atype, *gate_buf, std::nullopt, *up_buf, std::nullopt});

    gemm({*gate_buf, params.down_weight, params.output});
    
}

} // namespace fastertransformer

