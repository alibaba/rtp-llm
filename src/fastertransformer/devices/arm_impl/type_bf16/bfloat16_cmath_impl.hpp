/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bfloat16_cmath_impl.hpp
 */

#ifndef UTILS_BFLOAT16_CMATH_IMPL_HPP_
#define UTILS_BFLOAT16_CMATH_IMPL_HPP_

#include <cmath>

#include "bfloat16_impl.hpp"

namespace __hie_buildin {

namespace __bf16_cmath_impl {

#ifndef __BF16_CMATH_IMPL
#define __BF16_CMATH_IMPL

#define __CMATH_DEVICE_FUNC inline

#define __FP32_FALLBACK1(func)                                                                                         \
    __CMATH_DEVICE_FUNC __Bf16Impl func(__Bf16Impl arg) {                                                              \
        float val = __Bf16Impl::bfloat162float(arg);                                                                   \
        return __Bf16Impl::float2bfloat16(std::func(val));                                                             \
    }

#define __FP32_FALLBACK2(func)                                                                                         \
    __CMATH_DEVICE_FUNC __Bf16Impl func(__Bf16Impl x, __Bf16Impl y) {                                                  \
        float valx = __Bf16Impl::bfloat162float(x);                                                                    \
        float valy = __Bf16Impl::bfloat162float(y);                                                                    \
        return __Bf16Impl::float2bfloat16(std::func(valx, valy));                                                      \
    }

// basic operators
__FP32_FALLBACK1(fabs)
__FP32_FALLBACK2(fmod)
__FP32_FALLBACK2(remainder)

__CMATH_DEVICE_FUNC __Bf16Impl fma(__Bf16Impl x, __Bf16Impl y, __Bf16Impl z) {
    float valx = __Bf16Impl::bfloat162float(x);
    float valy = __Bf16Impl::bfloat162float(y);
    float valz = __Bf16Impl::bfloat162float(z);
    return __Bf16Impl::float2bfloat16(std::fma(valx, valy, valz));
}

__FP32_FALLBACK2(fmax)
__FP32_FALLBACK2(fmin)
__FP32_FALLBACK2(fdim)

// exponential functions
__FP32_FALLBACK1(exp)
__FP32_FALLBACK1(exp2)
__FP32_FALLBACK1(expm1)
__FP32_FALLBACK1(log)
__FP32_FALLBACK1(log10)
__FP32_FALLBACK1(log2)
__FP32_FALLBACK1(log1p)

// power functions
__FP32_FALLBACK2(pow)
__FP32_FALLBACK1(sqrt)

__CMATH_DEVICE_FUNC __Bf16Impl rsqrt(__Bf16Impl arg) {
    float val = __Bf16Impl::bfloat162float(arg);
    return __Bf16Impl::float2bfloat16(1.f / std::sqrt(val));
}

__FP32_FALLBACK1(cbrt)
__FP32_FALLBACK2(hypot)

// trigonometric functions
__FP32_FALLBACK1(sin)
__FP32_FALLBACK1(cos)
__FP32_FALLBACK1(tan)
__FP32_FALLBACK1(asin)
__FP32_FALLBACK1(acos)
__FP32_FALLBACK1(atan)
__FP32_FALLBACK2(atan2)

// hyperbolic functions
__FP32_FALLBACK1(sinh)
__FP32_FALLBACK1(cosh)
__FP32_FALLBACK1(tanh)
__FP32_FALLBACK1(asinh)
__FP32_FALLBACK1(acosh)
__FP32_FALLBACK1(atanh)

// error and gamma functions
__FP32_FALLBACK1(erf)
__FP32_FALLBACK1(erfc)
__FP32_FALLBACK1(tgamma)
__FP32_FALLBACK1(lgamma)

// nearest integer floating point operations
__FP32_FALLBACK1(ceil)
__FP32_FALLBACK1(floor)
__FP32_FALLBACK1(trunc)
__FP32_FALLBACK1(round)
__FP32_FALLBACK1(nearbyint)
__FP32_FALLBACK1(rint)

// classification
__CMATH_DEVICE_FUNC bool isinf(__Bf16Impl arg) {
    return (arg.__x & 0x7fffU) == 0x7f80U;
}
__CMATH_DEVICE_FUNC bool isnan(__Bf16Impl arg) {
    return (arg.__x & 0x7fffU) > 0x7f80U;
}
__CMATH_DEVICE_FUNC bool isfinite(__Bf16Impl arg) {
    return (arg.__x & 0x7fffU) < 0x7f80U;
}
__CMATH_DEVICE_FUNC bool isnormal(__Bf16Impl arg) {
    uint16_t expo = reinterpret_cast<uint16_t&>(arg) & 0x7f80;
    return expo < 0x7f80 && expo != 0;
}
__CMATH_DEVICE_FUNC bool signbit(__Bf16Impl arg) {
    return (arg.__x & 0x8000) != 0;
}

#undef __FP32_FALLBACK1
#undef __FP32_FALLBACK2

#endif  //  __BF16_CMATH_IMPL

}  // namespace __bf16_cmath_impl

}  // namespace __hie_buildin

#endif  // UTILS_BFLOAT16_CMATH_IMPL_HPP_
