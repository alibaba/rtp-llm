#include "src/fastertransformer/devices/DeviceBase.h"

using namespace std;

namespace fastertransformer {

DeviceBase::DeviceBase() {}

void DeviceBase::init() {
    buffer_manager_.reset(new BufferManager(getAllocator(), getHostAllocator()));
}

unique_ptr<Buffer> DeviceBase::allocateBuffer(const BufferParams& params, const BufferHints& hints) {
    return buffer_manager_->allocate(params, hints);
}

struct FFNDispatch {
    enum FFNType {
        NoGate,
        Gate,
        Moe,
    };

    static FFNType dispatch(const FfnLayerParams& params) {
        if (params.weights.moe_gating_weight != nullptr) {
            return Moe;
        }
        else if (isGatedActivation(params.activation_type)) {
            return Gate;
        }
        else {
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
    const auto& gate_weight = *(params.weights.gate_weight->kernel);
    const auto& up_weight = *(params.weights.up_weight->kernel);
    const auto& down_weight = *(params.weights.down_weight->kernel);

    auto up_buf = gemm({params.input, up_weight});
    
    auto up_output = loraLinear({params.input,
                                 std::nullopt,
                                 *(params.weights.up_weight),
                                 std::nullopt});

    if (FFNDispatch::dispatch(params) == FFNDispatch::FFNType::Gate) {
        auto gate_output = loraLinear({params.input,
                                       std::nullopt,
                                       *(params.weights.gate_weight),
                                       std::nullopt});

        activation({params.activation_type, 
                    *(gate_output.output),
                    std::nullopt,
                    *(up_output.output),
                    std::nullopt});

        auto output = loraLinear({*(gate_output.output),
                                  std::nullopt,
                                  *(params.weights.down_weight),
                                  std::nullopt});

        return FfnLayerOutput({move(output.output)});
        // auto gate_buf = gemm({params.input, gate_weight});
        // activation({params.activation_type, *gate_buf, std::nullopt, *up_buf, std::nullopt});
        // auto output = gemm({*gate_buf, down_weight});
        // return FfnLayerOutput({move(output)});
    }

    else if (FFNDispatch::dispatch(params) == FFNDispatch::FFNType::NoGate) {
        activation({params.activation_type, 
                    *(up_output.output),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt});

        auto output = loraLinear({*(up_output.output),
                                  std::nullopt,
                                  *(params.weights.down_weight),
                                  std::nullopt});

        return FfnLayerOutput({move(output.output)});
        // activation({params.activation_type, *up_buf, std::nullopt, std::nullopt, std::nullopt});
        // auto output = gemm({*up_buf, down_weight});
        // return FfnLayerOutput({move(output)});
    }
    
    else if (FFNDispatch::dispatch(params) == FFNDispatch::FFNType::Moe) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

LoraLinearOutput DeviceBase::loraLinear(const LoraLinearParams& params) {
    auto output = gemm({params.input, *(params.weight.kernel)});

    if (params.lora_ids != std::nullopt) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    return LoraLinearOutput({std::move(output)});
}

}; // namespace fastertransformer

