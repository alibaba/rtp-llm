#pragma once

#include <hipcub/backend/rocprim/util_type.hpp>
#include "rocprim_shims.h"

// hipcub only specializes FpLimits / NumericTraits for hip_bfloat16.
// amd_bfloat16 needs its own specializations for hipcub primitives
// (e.g. DeviceSegmentedSort) to work.

namespace hipcub {
template<>
struct FpLimits<amd_bfloat16> {
    static __host__ __device__ __forceinline__ amd_bfloat16 Max() {
        return amd_bfloat16(FpLimits<hip_bfloat16>::Max());
    }
    static __host__ __device__ __forceinline__ amd_bfloat16 Lowest() {
        return amd_bfloat16(FpLimits<hip_bfloat16>::Lowest());
    }
};
template<>
struct NumericTraits<amd_bfloat16>: BaseTraits<FLOATING_POINT, true, false, unsigned short, amd_bfloat16> {};
}  // namespace hipcub
