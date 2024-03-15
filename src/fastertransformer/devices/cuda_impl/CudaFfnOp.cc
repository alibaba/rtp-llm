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
FfnLayerOutput CudaDevice::ffnLayer(const FfnLayerParams& params) {
    const auto& input = params.input;
    const auto& gate_weight = *(params.weights.gate_weight->kernel);
    const auto& up_weight = *(params.weights.up_weight->kernel);
    const auto& down_weight = *(params.weights.down_weight->kernel);

    const auto token_num = input.shape()[0];
    const auto inter_size = gate_weight.shape()[1];

    auto gate_buf = gemm({params.input, gate_weight});
    auto up_buf = gemm({params.input, up_weight});

    activation({params.activation_type, *gate_buf, std::nullopt, *up_buf, std::nullopt});
    auto output = gemm({*gate_buf, down_weight});

    return FfnLayerOutput({move(output)});
}

} // namespace fastertransformer

