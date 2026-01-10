#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

using namespace std;

namespace rtp_llm {
BufferPtr DeviceBase::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    auto lora_output = loraLinear(params.lora_linear_params).output;

    if (initParams().profile_debug_logging_config.check_nan) {
        if (lora_output->isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(*lora_output);
            checkNAN(qbuffer.kernel(), "loraLinearWithAct_lora_output_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "loraLinearWithAct_lora_output_scales_dump", nullptr, true);
        } else {
            checkNAN(*lora_output, "loraLinearWithAct_lora_output_dump", nullptr, true);
        }
    }

    // create new activation from output;
    auto act_params   = params.activation_params;
    act_params.states = lora_output;
    return activation(act_params);
}
}  // namespace rtp_llm