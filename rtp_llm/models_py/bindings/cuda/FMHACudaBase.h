#pragma once

#include <cuda_runtime.h>
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"

namespace rtp_llm {

class FMHACudaBase {
public:
    FMHACudaBase(const GptInitParameter& gpt_init_parameter):
        attn_configs_(gpt_init_parameter.getAttentionConfigs()),
        fmha_config_(gpt_init_parameter.fmha_config),
        device_(nullptr) {
        if (!DeviceFactory::isAlreadyInit()) {
            DeviceFactory::initDevices(gpt_init_parameter);
        }
        device_ = dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice());
    }

protected:
    AttentionConfigs attn_configs_;
    FMHAConfig       fmha_config_;
    CudaDevice*      device_;
};
}  // namespace rtp_llm
