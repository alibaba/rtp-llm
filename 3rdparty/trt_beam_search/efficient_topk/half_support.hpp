#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_utils.hpp"

namespace HipKernels {
namespace numeric {

template<>
__inline__ __half get_lower_bound<__half>() {
    return __float2half(-INFINITY);
}

template<>
__inline__ __half get_upper_bound<__half>() {
    return __float2half(INFINITY);
}

}  // namespace numeric
}  // namespace HipKernels
