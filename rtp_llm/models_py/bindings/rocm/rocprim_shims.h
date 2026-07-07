#pragma once

#include <rocprim/type_traits.hpp>
#include "amd_bfloat16.h"
#if ROCM_VERSION < 70000
#include <rocprim/thread/radix_key_codec.hpp>
#endif

// rocprim only specializes traits::define for rocprim::bfloat16 (== hip_bfloat16).
// amd_bfloat16 inherits from __hip_bfloat16 but is a distinct type, so it needs
// its own specialization for rocprim primitives (e.g. radix sort) to work.

namespace rocprim {
template<>
struct traits::define<amd_bfloat16>: traits::define<rocprim::bfloat16> {};

#if ROCM_VERSION < 70000
namespace detail {
template<>
struct radix_key_codec_base<amd_bfloat16>: radix_key_codec_floating<amd_bfloat16, unsigned short> {};
}  // namespace detail
#endif
}  // namespace rocprim
