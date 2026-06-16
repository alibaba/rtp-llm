#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace rtp_llm {

int64_t getDeviceId();

torch::Device getTorchCudaDevice();

}  // namespace rtp_llm
