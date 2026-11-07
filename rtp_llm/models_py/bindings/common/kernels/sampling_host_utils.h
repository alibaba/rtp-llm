#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {
namespace sampler {

inline bool isSm120() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return properties->major == 12 && properties->minor == 0;
}

}  // namespace sampler
}  // namespace rtp_llm
