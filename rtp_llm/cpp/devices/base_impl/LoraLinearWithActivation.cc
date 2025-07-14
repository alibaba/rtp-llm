#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"

using namespace std;

namespace rtp_llm {
BufferPtr DeviceBase::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    BufferPtr output = loraLinear(params.lora_linear_params).output;
    // create new activation from output;
    auto act_params   = params.activation_params;
    act_params.states = output;
    return activation(act_params);
}
}  // namespace rtp_llm