#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"

using namespace std;

namespace fastertransformer {
    BufferPtr DeviceBase::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
        BufferPtr output = loraLinear(params.lora_linear_params).output;
        // create new activation from output;
        auto act_params = params.activation_params;
        act_params.states = output;
        return activation(act_params);
    }
} // namespace fastertransformer