#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
using namespace std;

namespace fastertransformer {

struct FFNDispatch {
    enum FFNType {
        NoGate,
        Gate,
        Moe,
    };

    static FFNType dispatch(const FfnLayerParams& params) {
        if (params.weights.moe_gating_weight != nullptr) {
            return Moe;
        } else if (isGatedActivation(params.activation_type)) {
            return Gate;
        } else {
            return NoGate;
        }
    }
};


/// @brief   feed forward neural network ops
/// @details output = Gemm(Act(Gemm(input, W1) + b1), W2) + b2
///          input(array) : [m, k]
///          W1(array) : [k, n]
///          b1(array) : [m, n]
///          W2(array) : [m, n]
///          b2(array)
///          output(array)
FfnLayerOutput DeviceBase::ffnLayer(const FfnLayerParams& params) {
    const auto& input = params.input;
    const auto& up_weight = *(params.weights.up_weight->kernel);
    const auto& down_weight = *(params.weights.down_weight->kernel);

    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");

    auto up_output = loraLinear({params.input,
                                 std::nullopt,
                                 *(params.weights.up_weight),
                                 std::nullopt});
    printBufferData(*up_output.output, "ffn_up");
    if (FFNDispatch::dispatch(params) == FFNDispatch::FFNType::Gate) {
        {
            auto gate_output = loraLinear({params.input,
                                        std::nullopt,
                                        *(params.weights.gate_weight),
                                        std::nullopt});

            activation({params.activation_type,
                        *(up_output.output),
                        mayGetRef(params.weights.up_weight->bias),
                        *(gate_output.output),
                        std::nullopt});
            gate_output.output.reset();
        }

        auto output = loraLinear({*(up_output.output),
                                  std::nullopt,
                                  *(params.weights.down_weight),
                                  std::nullopt});
        return FfnLayerOutput({move(output.output)});
    } else if (FFNDispatch::dispatch(params) == FFNDispatch::FFNType::NoGate) {
        activation({params.activation_type,
                    *(up_output.output),
                    mayGetRef(params.weights.up_weight->bias),
                    std::nullopt,
                    std::nullopt});
        printBufferData(*up_output.output, "ffn_act");
        auto output = loraLinear({*(up_output.output),
                                  std::nullopt,
                                  *(params.weights.down_weight),
                                  std::nullopt});
        printBufferData(*output.output, "ffn_out");
        return FfnLayerOutput({move(output.output)});
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

}; // namespace fastertransformer
