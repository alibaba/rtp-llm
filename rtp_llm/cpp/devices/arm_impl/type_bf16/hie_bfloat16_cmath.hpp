/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hie_bfloat16_cmath.hpp
 */

#ifndef UTILS_BFLOAT16_CMATH_HPP_
#define UTILS_BFLOAT16_CMATH_HPP_

#include "bfloat16_cmath_impl.hpp"
#include "hie_bfloat16.hpp"

// ------------------------------------
//  C++11 cmath API for BFP16
// ------------------------------------

#if !defined(__BF16_CMATH_IMPL)
#error __BF16_CMATH_IMPL not defined
#endif

#define __HIE_CMATH_API_BFBF(func, expr)                                                                               \
    __CMATH_DEVICE_FUNC                                                                                                \
    __hie_buildin::bfloat16 func(__hie_buildin::bfloat16 arg) {                                                        \
        return __hie_buildin::bfloat16(__hie_buildin::__bf16_cmath_impl::expr(arg.__x));                               \
    }

#define __HIE_CMATH_API_BFBFBF(func, expr)                                                                             \
    __CMATH_DEVICE_FUNC __hie_buildin::bfloat16 func(__hie_buildin::bfloat16 x, __hie_buildin::bfloat16 y) {           \
        return __hie_buildin::bfloat16(__hie_buildin::__bf16_cmath_impl::expr(x.__x, y.__x));                          \
    }

#define __HIE_CMATH_API_BBF(func)                                                                                      \
    __CMATH_DEVICE_FUNC bool func(::__hie_buildin::bfloat16 arg) {                                                     \
        return ::__hie_buildin::__bf16_cmath_impl::func(arg.__x);                                                      \
    }

// basic operators
__HIE_CMATH_API_BFBF(fabs_bf16, fabs)
__HIE_CMATH_API_BFBFBF(fmod_bf16, fmod)
__HIE_CMATH_API_BFBFBF(remainder_bf16, remainder)

__CMATH_DEVICE_FUNC __hie_buildin::bfloat16
                    fma_bf16(__hie_buildin::bfloat16 x, __hie_buildin::bfloat16 y, __hie_buildin::bfloat16 z) {
    return __hie_buildin::bfloat16(__hie_buildin::__bf16_cmath_impl::fma(x.__x, y.__x, z.__x));
}

__HIE_CMATH_API_BFBFBF(fmax_bf16, fmax)
__HIE_CMATH_API_BFBFBF(fmin_bf16, fmin)
__HIE_CMATH_API_BFBFBF(fdim_bf16, fdim)

// exponential functions
__HIE_CMATH_API_BFBF(exp_bf16, exp)
__HIE_CMATH_API_BFBF(exp2_bf16, exp2)
__HIE_CMATH_API_BFBF(expm1_bf16, expm1)
__HIE_CMATH_API_BFBF(log_bf16, log)
__HIE_CMATH_API_BFBF(log10_bf16, log10)
__HIE_CMATH_API_BFBF(log2_bf16, log2)
__HIE_CMATH_API_BFBF(log1p_bf16, log1p)

// power functions
__HIE_CMATH_API_BFBFBF(pow_bf16, pow)
__HIE_CMATH_API_BFBF(sqrt_bf16, sqrt)
__HIE_CMATH_API_BFBF(rsqrt_bf16, rsqrt)
__HIE_CMATH_API_BFBF(cbrt_bf16, cbrt)
__HIE_CMATH_API_BFBFBF(hypot_bf16, hypot)

// trigonometric functions
__HIE_CMATH_API_BFBF(sin_bf16, sin)
__HIE_CMATH_API_BFBF(cos_bf16, cos)
__HIE_CMATH_API_BFBF(tan_bf16, tan)
__HIE_CMATH_API_BFBF(asin_bf16, asin)
__HIE_CMATH_API_BFBF(acos_bf16, acos)
__HIE_CMATH_API_BFBF(atan_bf16, atan)
__HIE_CMATH_API_BFBFBF(atan2_bf16, atan2)

// hyperbolic functions
__HIE_CMATH_API_BFBF(sinh_bf16, sinh)
__HIE_CMATH_API_BFBF(cosh_bf16, cosh)
__HIE_CMATH_API_BFBF(tanh_bf16, tanh)
__HIE_CMATH_API_BFBF(asinh_bf16, asinh)
__HIE_CMATH_API_BFBF(acosh_bf16, acosh)
__HIE_CMATH_API_BFBF(atanh_bf16, atanh)

// error and gamma functions
__HIE_CMATH_API_BFBF(erf_bf16, erf)
__HIE_CMATH_API_BFBF(erfc_bf16, erfc)
__HIE_CMATH_API_BFBF(tgamma_bf16, tgamma)
__HIE_CMATH_API_BFBF(lgamma_bf16, lgamma)

// nearest integer floating point operations
__HIE_CMATH_API_BFBF(ceil_bf16, ceil)
__HIE_CMATH_API_BFBF(floor_bf16, floor)
__HIE_CMATH_API_BFBF(trunc_bf16, trunc)
__HIE_CMATH_API_BFBF(round_bf16, round)
__HIE_CMATH_API_BFBF(nearbyint_bf16, nearbyint)
__HIE_CMATH_API_BFBF(rint_bf16, rint)

// classification
namespace std {
__HIE_CMATH_API_BBF(isfinite)
__HIE_CMATH_API_BBF(isinf)
__HIE_CMATH_API_BBF(isnan)
__HIE_CMATH_API_BBF(isnormal)
__HIE_CMATH_API_BBF(signbit)
}  // namespace std

#undef __HIE_CMATH_API_BFBF
#undef __HIE_CMATH_API_BFBFBF
#undef __HIE_CMATH_API_BBF

#undef __BF16_CMATH_IMPL
#undef __CMATH_DEVICE_FUNC

#endif  // UTILS_BFLOAT16_CMATH_HPP_
