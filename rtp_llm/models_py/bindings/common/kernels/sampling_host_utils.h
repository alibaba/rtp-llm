#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cstring>

namespace rtp_llm {
namespace sampler {

inline bool optimizationEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool isSm120() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return properties->major == 12 && properties->minor == 0;
}

}  // namespace sampler
}  // namespace rtp_llm
