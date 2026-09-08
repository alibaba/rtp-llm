#pragma once

#include <rocprim/type_traits.hpp>
#include "amd_bfloat16.h"

// rocprim only specializes traits::define for rocprim::bfloat16 (== hip_bfloat16).
// amd_bfloat16 inherits from __hip_bfloat16 but is a distinct type, so it needs
// its own specialization for rocprim primitives (e.g. radix sort) to work.

namespace rocprim {
template<>
struct traits::define<amd_bfloat16>: traits::define<rocprim::bfloat16> {};
}  // namespace rocprim
