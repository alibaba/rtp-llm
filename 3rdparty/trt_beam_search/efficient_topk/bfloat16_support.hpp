#pragma once

#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include "hip_utils.hpp"

namespace HipKernels {
namespace numeric {

template<>
__inline__ amd_bfloat16 get_lower_bound<amd_bfloat16>() {
    return __float2bfloat16(-INFINITY);
}

template<>
__inline__ amd_bfloat16 get_upper_bound<amd_bfloat16>() {
    return __float2bfloat16(INFINITY);
}

}  // namespace numeric
}  // namespace HipKernels
