#pragma once

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

class FMHARocmBase {
public:
    FMHARocmBase(const GptInitParameter& gpt_init_parameter):
        attn_configs_(gpt_init_parameter.getAttentionConfigs()),
        layer_num_(gpt_init_parameter.num_layers_),
        hw_kernel_config_(gpt_init_parameter.hw_kernel_config),
        device_(dynamic_cast<ROCmDevice*>(DeviceFactory::getDefaultDevice())) {}

protected:
    AttentionConfigs attn_configs_;
    int              layer_num_;
    HWKernelConfig   hw_kernel_config_;
    ROCmDevice*      device_;
};
}  // namespace rtp_llm