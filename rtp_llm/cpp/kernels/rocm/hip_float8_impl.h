#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#if defined(__HIPCC__) &&                                                      \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
#define __HIP__MI300__
#endif

#ifdef __HIPCC__
#define HIP_FP8_HOST_DEVICE __host__ __device__
#define HIP_FP8_HOST __host__
#define HIP_FP8_DEVICE __device__
#else
#define HIP_FP8_HOST_DEVICE
#define HIP_FP8_HOST
#define HIP_FP8_DEVICE
#endif

namespace hip_fp8_impl {

#ifdef __HIP__MI300__
HIP_FP8_DEVICE uint8_t to_fp8_from_fp32(float v) {
  uint8_t i8data;
  union {
    float fval;
    uint32_t i32val;
    uint8_t i8val[4];
  } val;

  uint32_t ival = 0;
  val.fval = v;

  if ((val.i32val & 0x7F800000) != 0x7F800000) {
    val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
  }

  ival = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false);
  val.i32val = ival;
  i8data = val.i8val[0];

  return i8data;
}
#endif

HIP_FP8_HOST inline int clz(uint32_t x) { return __builtin_clz(x); }
#if defined(__HIPCC__) || defined(__CUDA_ARCH__)
HIP_FP8_DEVICE inline int clz(uint32_t x) { return __clz(x); }
#endif

template <int we, int wm, typename T, bool negative_zero_nan, bool clip>
HIP_FP8_HOST_DEVICE uint8_t to_float8(T _x, bool stoch = false,
                                      uint32_t rng = 0) {
#ifdef __HIPCC__
  constexpr bool is_half = std::is_same<T, _Float16>::value;
#else
  constexpr bool is_half = false;
#endif
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(wm + we == 7, "wm+we==7");
  static_assert(is_half || is_float, "Only half and float can be cast to f8");

  const int mfmt = (sizeof(T) == 4) ? 23 : 10;
  uint32_t x;
  if (sizeof(T) == 4) {
    x = reinterpret_cast<uint32_t &>(_x);
  } else {
    x = reinterpret_cast<uint16_t &>(_x);
  }

  uint32_t head, mantissa;
  int exponent, bias;
  uint32_t sign;

  if (sizeof(T) == 4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign = head >> 31;
    bias = 127;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head >> 10) & 0x1F;
    sign = head >> 15;
    bias = 15;
  }

  uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

  if (negative_zero_nan) {
    if (sizeof(T) == 4) {
      if ((x & 0x7F800000) == 0x7F800000) {
        return 0x80;
      }
    } else {
      if ((x & 0x7C00) == 0x7C00) {
        return 0x80;
      }
    }
  } else {
    if (sizeof(T) == 4) {
      if ((x & 0x7F800000) == 0x7F800000) {
        return signed_inf + (mantissa != 0 ? 1 : 0);
      }
    } else {
      if ((x & 0x7C00) == 0x7C00) {
        return signed_inf + (mantissa != 0 ? 1 : 0);
      }
    }
  }
  if (x == 0) {
    return 0;
  }

  const int f8_bias = (1 << (we - 1)) - 1 + (negative_zero_nan ? 1 : 0);
  const int f8_denormal_act_exponent = 1 - f8_bias;
  int act_exponent, f8_exponent, exponent_diff;

  if (exponent == 0) {
    act_exponent = exponent - bias + 1;
    exponent_diff = f8_denormal_act_exponent - act_exponent;
  } else {
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) {
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {
      exponent_diff = 0;
    }
    mantissa += (1 << mfmt);
  }

  bool midpoint = (mantissa & ((1 << (mfmt - wm + exponent_diff)) - 1)) ==
                  static_cast<uint32_t>(1 << (mfmt - wm + exponent_diff - 1));

  if (exponent_diff > 0) {
    mantissa >>= exponent_diff;
  } else if (exponent_diff == -1) {
    mantissa <<= -exponent_diff;
  }
  bool implicit_one = mantissa & (1 << mfmt);
  f8_exponent = (act_exponent + exponent_diff) + f8_bias - (implicit_one ? 0 : 1);

  uint32_t drop_mask = (1 << (mfmt - wm)) - 1;
  bool odd = mantissa & (1 << (mfmt - wm));
  mantissa +=
      (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) &
      drop_mask;

  if (f8_exponent == 0) {
    if ((1 << mfmt) & mantissa) {
      f8_exponent = 1;
    }
  } else {
    if ((1 << (mfmt + 1)) & mantissa) {
      mantissa >>= 1;
      f8_exponent++;
    }
  }

  mantissa >>= (mfmt - wm);

  const int max_exp = (1 << we) - (negative_zero_nan ? 1 : 2);
  if (f8_exponent > max_exp) {
    if (clip) {
      mantissa = (1 << wm) - 1;
      f8_exponent = max_exp;
    } else {
      return signed_inf;
    }
  }

  if (f8_exponent == 0 && mantissa == 0) {
    return negative_zero_nan ? 0 : (sign << 7);
  }
  mantissa &= (1 << wm) - 1;
  return (sign << 7) | (f8_exponent << wm) | mantissa;
}

template <int we, int wm, typename T = float, bool negative_zero_nan = true>
inline HIP_FP8_HOST_DEVICE T from_float8(uint8_t x) {
#ifdef __HIPCC__
  constexpr bool is_half = std::is_same<T, _Float16>::value;
#else
  constexpr bool is_half = false;
#endif
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(is_half || is_float, "only half and float are supported");

  constexpr int weo = is_half ? 5 : 8;
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

  T fInf, fNegInf, fNaN, fNeg0;

#ifdef __HIPCC__
  if (is_half) {
    const uint16_t ihInf = 0x7C00;
    const uint16_t ihNegInf = 0xFC00;
    const uint16_t ihNaN = 0x7C01;
    const uint16_t ihNeg0 = 0x8000;
    fInf = reinterpret_cast<const _Float16 &>(ihInf);
    fNegInf = reinterpret_cast<const _Float16 &>(ihNegInf);
    fNaN = reinterpret_cast<const _Float16 &>(ihNaN);
    fNeg0 = reinterpret_cast<const _Float16 &>(ihNeg0);
  } else
#endif
      if (is_float) {
    const uint32_t ifInf = 0x7F800000;
    const uint32_t ifNegInf = 0xFF800000;
    const uint32_t ifNaN = 0x7F800001;
    const uint32_t ifNeg0 = 0x80000000;
    fInf = reinterpret_cast<const float &>(ifInf);
    fNegInf = reinterpret_cast<const float &>(ifNegInf);
    fNaN = reinterpret_cast<const float &>(ifNaN);
    fNeg0 = reinterpret_cast<const float &>(ifNeg0);
  }

  if (x == 0) {
    return 0;
  }

  uint32_t sign = x >> 7;
  uint32_t mantissa = x & ((1 << wm) - 1);
  int exponent = (x & 0x7F) >> wm;
  if (negative_zero_nan) {
    if (x == 0x80) {
      return fNaN;
    }
  } else {
    if (x == 0x80) {
      return fNeg0;
    }
    if (exponent == ((1 << we) - 1)) {
      return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
    }
  }
  typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type retval;
  if (we == 5 && is_half && !negative_zero_nan) {
    retval = x << 8;
    return reinterpret_cast<const T &>(retval);
  }

  const int exp_low_cutoff =
      (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

  // subnormal input
  if (exponent == 0) {
    // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + clz(mantissa) - (32 - wm);
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1 << wm) - 1);
  }
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if (exponent <= 0) {
    mantissa |= 1 << wmo;
    mantissa >>= 1 - exponent;
    exponent = 0;
  }

  if (sizeof(T) == 2) {
    retval = (sign << 15) | (exponent << 10) | mantissa;
  } else {
    retval = (sign << 31) | (exponent << 23) | mantissa;
  }
  return reinterpret_cast<const T &>(retval);
}

} // namespace hip_fp8_impl
