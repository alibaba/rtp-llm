#pragma once

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

class FMHACudaBase {
public:
    FMHACudaBase(const GptInitParameter& gpt_init_parameter):
        attn_configs_(gpt_init_parameter.getAttentionConfigs()),
        fmha_config_(gpt_init_parameter.fmha_config),
        device_(dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice())) {}

protected:
    AttentionConfigs attn_configs_;
    FMHAConfig       fmha_config_;
    CudaDevice*      device_;
};
}  // namespace rtp_llm
