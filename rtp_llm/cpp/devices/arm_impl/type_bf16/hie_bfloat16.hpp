/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hie_bfloat16.hpp
 */

#ifndef UTILS_BFLOAT16_HPP_
#define UTILS_BFLOAT16_HPP_

#include <cstdint>
#include <limits>
#include <ostream>

#include "bfloat16_impl.hpp"

namespace __hie_buildin {

constexpr size_t __BF16_SIZE_INBYTE = 2;

// extended types forward declaration
// #ifdef HIE_ENABLE_FLOAT16
// struct half;
// #endif

#if !defined(__BF16_IMPL)
#error __BF16_IMPL not defined
#endif

struct bfloat16 {
    __Bf16Impl __x;
    static_assert(sizeof(__Bf16Impl) == __BF16_SIZE_INBYTE, "invalid __Bf16Impl size");

    static uint32_t constexpr kFracSize = 7;
    static uint32_t constexpr kExpSize  = 8;
    static uint32_t constexpr kExpBias  = 127;

    /*
     * default rounding mode (float/double to bf16): nearest
     * Note we do avoid constructor init-list because of special host/device
     * compilation rules
     */

#if __cplusplus >= 201103L
    bfloat16() = default;
#else
    __BF16_DEVICE_FUNC bfloat16() {}
#endif  // __cplusplus >= 201103L

    __BF16_DEVICE_FUNC explicit bfloat16(__Bf16Impl v) {
        __x = v;
    }

    __BF16_DEVICE_FUNC static bfloat16 from_bits(uint16_t bits) {
        return bfloat16(__Bf16Impl::from_bits(bits));
    }

    // convert from other extended types
    // #ifdef HIE_ENABLE_FLOAT16
    //     __BF16_DEVICE_FUNC_DECL bfloat16(half v);
    // #endif

    // convert from build-in types
    __BF16_DEVICE_FUNC bfloat16(float v) {
        __x = __Bf16Impl::float2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(double v) {
        __x = __Bf16Impl::double2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(long long v) {
        __x = __Bf16Impl::ll2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned long long v) {
        __x = __Bf16Impl::ull2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(long v) {
        __x = __Bf16Impl::ll2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned long v) {
        __x = __Bf16Impl::ull2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(int v) {
        __x = __Bf16Impl::int2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned int v) {
        __x = __Bf16Impl::uint2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(short v) {
        __x = __Bf16Impl::short2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned short v) {
        __x = __Bf16Impl::ushort2bfloat16(v);
    }

    __BF16_DEVICE_FUNC bfloat16(char v) {
        __x = __Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    __BF16_DEVICE_FUNC bfloat16(signed char v) {
        __x = __Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    __BF16_DEVICE_FUNC bfloat16(unsigned char v) {
        __x = __Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    __BF16_DEVICE_FUNC bfloat16(bool v) {
        __x = __Bf16Impl::short2bfloat16(static_cast<short>(v));
    }

    // convert to extended types
    __BF16_DEVICE_FUNC bfloat16(float16_t v) {
        __x = __Bf16Impl::float162bfloat16(v);
    }

    // convert to build-in types
    __BF16_DEVICE_FUNC operator float() const {
        return __Bf16Impl::bfloat162float(__x);
    }

    __BF16_DEVICE_FUNC operator double() const {
        return __Bf16Impl::bfloat162double(__x);
    }

    __BF16_DEVICE_FUNC operator long long() const {
        return __Bf16Impl::bfloat162ll(__x);
    }

    __BF16_DEVICE_FUNC operator unsigned long long() const {
        return __Bf16Impl::bfloat162ull(__x);
    }

    __BF16_DEVICE_FUNC operator long() const {
        return __Bf16Impl::bfloat162ll(__x);
    }

    __BF16_DEVICE_FUNC operator unsigned long() const {
        return __Bf16Impl::bfloat162ull(__x);
    }

    __BF16_DEVICE_FUNC operator int() const {
        return __Bf16Impl::bfloat162int(__x);
    }

    __BF16_DEVICE_FUNC operator unsigned int() const {
        return __Bf16Impl::bfloat162uint(__x);
    }

    __BF16_DEVICE_FUNC operator short() const {
        return __Bf16Impl::bfloat162short(__x);
    }

    __BF16_DEVICE_FUNC operator unsigned short() const {
        return __Bf16Impl::bfloat162ushort(__x);
    }

    __BF16_DEVICE_FUNC operator char() const {
        return static_cast<char>(__Bf16Impl::bfloat162short(__x));
    }

    __BF16_DEVICE_FUNC operator signed char() const {
        return static_cast<signed char>(__Bf16Impl::bfloat162short(__x));
    }

    __BF16_DEVICE_FUNC operator unsigned char() const {
        return static_cast<unsigned char>(__Bf16Impl::bfloat162short(__x));
    }

    __BF16_DEVICE_FUNC operator bool() const {
        return (reinterpret_cast<const std::uint16_t&>(__x) & 0x7fff) != 0;
    }

    //convert from other extended types
    __BF16_DEVICE_FUNC operator float16_t() const {
        return __Bf16Impl::bfloat162float16(__x);
    }

    friend std::ostream& operator<<(std::ostream& out, const bfloat16& obj) {
        out << __Bf16Impl::bfloat162float(obj.__x);
        return out;
    }
};  // struct bfloat16

// positive & negative
__BF16_DEVICE_FUNC bfloat16 operator-(bfloat16 a) {
    std::uint16_t ret = reinterpret_cast<std::uint16_t&>(a) ^ 0x8000;
    return reinterpret_cast<bfloat16&>(ret);
}
__BF16_DEVICE_FUNC bfloat16 operator+(bfloat16 a) {
    return a;
}

#define __TO_BF16_BINARY_OPERATOR(op, type)                                                                            \
    __BF16_DEVICE_FUNC bfloat16 operator op(type a, bfloat16 b) {                                                      \
        return static_cast<bfloat16>(a) op b;                                                                          \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bfloat16 operator op(bfloat16 a, type b) {                                                      \
        return a op static_cast<bfloat16>(b);                                                                          \
    }

#define __BF16_BINARY_OPERATOR(op, impl_expr)                                                                          \
    __BF16_DEVICE_FUNC float operator op(float a, bfloat16 b) {                                                        \
        return a op static_cast<float>(b);                                                                             \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC float operator op(bfloat16 a, float b) {                                                        \
        return static_cast<float>(a) op b;                                                                             \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC double operator op(double a, bfloat16 b) {                                                      \
        return a op static_cast<double>(b);                                                                            \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC double operator op(bfloat16 a, double b) {                                                      \
        return static_cast<double>(a) op b;                                                                            \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bfloat16 operator op(bfloat16 a, bfloat16 b) {                                                  \
        return bfloat16(__Bf16Impl::impl_expr(a.__x, b.__x));                                                          \
    }                                                                                                                  \
    __TO_BF16_BINARY_OPERATOR(op, long long)                                                                           \
    __TO_BF16_BINARY_OPERATOR(op, unsigned long long)                                                                  \
    __TO_BF16_BINARY_OPERATOR(op, long)                                                                                \
    __TO_BF16_BINARY_OPERATOR(op, unsigned long)                                                                       \
    __TO_BF16_BINARY_OPERATOR(op, int)                                                                                 \
    __TO_BF16_BINARY_OPERATOR(op, unsigned int)                                                                        \
    __TO_BF16_BINARY_OPERATOR(op, short)                                                                               \
    __TO_BF16_BINARY_OPERATOR(op, unsigned short)                                                                      \
    __TO_BF16_BINARY_OPERATOR(op, char)                                                                                \
    __TO_BF16_BINARY_OPERATOR(op, signed char)                                                                         \
    __TO_BF16_BINARY_OPERATOR(op, unsigned char)                                                                       \
    __TO_BF16_BINARY_OPERATOR(op, bool)

// + - * /
__BF16_BINARY_OPERATOR(+, bf16add)
__BF16_BINARY_OPERATOR(-, bf16sub)
__BF16_BINARY_OPERATOR(*, bf16mul)
__BF16_BINARY_OPERATOR(/, bf16div)
#undef __BF16_BINARY_OPERATOR
#undef __TO_BF16_BINARY_OPERATOR

// += -= *= /=
__BF16_DEVICE_FUNC
bfloat16& operator+=(bfloat16& a, const bfloat16& b) {
    a = a + b;
    return a;
}
__BF16_DEVICE_FUNC
bfloat16& operator-=(bfloat16& a, const bfloat16& b) {
    a = a - b;
    return a;
}
__BF16_DEVICE_FUNC
bfloat16& operator*=(bfloat16& a, const bfloat16& b) {
    a = a * b;
    return a;
}
__BF16_DEVICE_FUNC
bfloat16& operator/=(bfloat16& a, const bfloat16& b) {
    a = a / b;
    return a;
}

// ++ --
__BF16_DEVICE_FUNC bfloat16& operator++(bfloat16& v) {
    std::uint16_t one = 0x3f80;
    v += reinterpret_cast<const bfloat16&>(one);
    return v;
}
__BF16_DEVICE_FUNC bfloat16& operator--(bfloat16& v) {
    std::uint16_t one = 0x3f80;
    v -= reinterpret_cast<const bfloat16&>(one);
    return v;
}
__BF16_DEVICE_FUNC bfloat16 operator++(bfloat16& v, int) {
    bfloat16      r   = v;
    std::uint16_t one = 0x3f80;
    v += reinterpret_cast<const bfloat16&>(one);
    return r;
}
__BF16_DEVICE_FUNC bfloat16 operator--(bfloat16& v, int) {
    bfloat16      r   = v;
    std::uint16_t one = 0x3f80;
    v -= reinterpret_cast<const bfloat16&>(one);
    return r;
}

#define __TO_BF16_CMP_OPERATOR(op, type)                                                                               \
    __BF16_DEVICE_FUNC bool operator op(type a, bfloat16 b) {                                                          \
        return static_cast<bfloat16>(a) op b;                                                                          \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bool operator op(bfloat16 a, type b) {                                                          \
        return a op static_cast<bfloat16>(b);                                                                          \
    }

#define __BF16_CMP_OPERATOR(op, impl_expr)                                                                             \
    __BF16_DEVICE_FUNC bool operator op(float a, bfloat16 b) {                                                         \
        return a op static_cast<float>(b);                                                                             \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bool operator op(bfloat16 a, float b) {                                                         \
        return static_cast<float>(a) op b;                                                                             \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bool operator op(double a, bfloat16 b) {                                                        \
        return a op static_cast<double>(b);                                                                            \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bool operator op(bfloat16 a, double b) {                                                        \
        return static_cast<double>(a) op b;                                                                            \
    }                                                                                                                  \
    __BF16_DEVICE_FUNC bool operator op(bfloat16 a, bfloat16 b) {                                                      \
        return __Bf16Impl::impl_expr(a.__x, b.__x);                                                                    \
    }                                                                                                                  \
    __TO_BF16_CMP_OPERATOR(op, long long)                                                                              \
    __TO_BF16_CMP_OPERATOR(op, unsigned long long)                                                                     \
    __TO_BF16_CMP_OPERATOR(op, long)                                                                                   \
    __TO_BF16_CMP_OPERATOR(op, unsigned long)                                                                          \
    __TO_BF16_CMP_OPERATOR(op, int)                                                                                    \
    __TO_BF16_CMP_OPERATOR(op, unsigned int)                                                                           \
    __TO_BF16_CMP_OPERATOR(op, short)                                                                                  \
    __TO_BF16_CMP_OPERATOR(op, unsigned short)                                                                         \
    __TO_BF16_CMP_OPERATOR(op, char)                                                                                   \
    __TO_BF16_CMP_OPERATOR(op, signed char)                                                                            \
    __TO_BF16_CMP_OPERATOR(op, unsigned char)                                                                          \
    __TO_BF16_CMP_OPERATOR(op, bool)

// == != > < >= <=
__BF16_CMP_OPERATOR(==, bf16eq)
__BF16_CMP_OPERATOR(!=, bf16ne)
__BF16_CMP_OPERATOR(>, bf16gt)
__BF16_CMP_OPERATOR(<, bf16lt)
__BF16_CMP_OPERATOR(>=, bf16ge)
__BF16_CMP_OPERATOR(<=, bf16le)
#undef __BF16_CMP_OPERATOR
#undef __TO_BF16_CMP_OPERATOR

}  // namespace __hie_buildin

namespace hie {
typedef __hie_buildin::bfloat16 bfloat16;
}

namespace std {
template<>
class numeric_limits<hie::bfloat16> {
    using bf16_impl_t = __hie_buildin::__Bf16Impl;

public:
    static const bool                    is_specialized    = std::numeric_limits<bf16_impl_t>::is_specialized;
    static const bool                    is_signed         = std::numeric_limits<bf16_impl_t>::is_signed;
    static const bool                    is_integer        = std::numeric_limits<bf16_impl_t>::is_integer;
    static const bool                    is_exact          = std::numeric_limits<bf16_impl_t>::is_exact;
    static const bool                    is_modulo         = std::numeric_limits<bf16_impl_t>::is_modulo;
    static const bool                    is_bounded        = std::numeric_limits<bf16_impl_t>::is_bounded;
    static const bool                    is_iec559         = std::numeric_limits<bf16_impl_t>::is_iec559;
    static const bool                    has_infinity      = std::numeric_limits<bf16_impl_t>::has_infinity;
    static const bool                    has_quiet_NaN     = std::numeric_limits<bf16_impl_t>::has_quiet_NaN;
    static const bool                    has_signaling_NaN = std::numeric_limits<bf16_impl_t>::has_signaling_NaN;
    static const std::float_denorm_style has_denorm        = std::numeric_limits<bf16_impl_t>::has_denorm;
    static const bool                    has_denorm_loss   = std::numeric_limits<bf16_impl_t>::has_denorm_loss;

    static const bool traps = std::numeric_limits<bf16_impl_t>::traps;

    static const bool tinyness_before = std::numeric_limits<bf16_impl_t>::tinyness_before;

    static const std::float_round_style round_style = std::numeric_limits<bf16_impl_t>::round_style;

    /// Significant digits.
    static const int digits = std::numeric_limits<bf16_impl_t>::digits;

    /// Significant decimal digits.
    static const int digits10 = std::numeric_limits<bf16_impl_t>::digits10;

    /// Required decimal digits to represent all possible values.
    static const int max_digits10 = std::numeric_limits<bf16_impl_t>::max_digits10;

    /// Number base.
    static const int radix = std::numeric_limits<bf16_impl_t>::radix;

    /// One more than smallest exponent.
    static const int min_exponent = std::numeric_limits<bf16_impl_t>::min_exponent;

    /// Smallest normalized representable power of 10.
    static const int min_exponent10 = std::numeric_limits<bf16_impl_t>::min_exponent10;

    /// One more than largest exponent
    static const int max_exponent = std::numeric_limits<bf16_impl_t>::max_exponent;

    /// Largest finitely representable power of 10.
    static const int max_exponent10 = std::numeric_limits<bf16_impl_t>::max_exponent10;

    /// Smallest positive normal value.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 min() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::min());
    }

    /// Smallest finite value.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 lowest() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::lowest());
    }

    /// Largest finite value.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 max() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::max());
    }

    /// Difference between 1 and next representable value.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 epsilon() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::epsilon());
    }

    /// Maximum rounding error in ULP (units in the last place).
    __BF16_DEVICE_FUNC
    static hie::bfloat16 round_error() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::round_error());
    }

    /// Positive infinity.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 infinity() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::infinity());
    }

    /// Quiet NaN.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 quiet_NaN() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::quiet_NaN());
    }

    /// Signaling NaN.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 signaling_NaN() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::signaling_NaN());
    }

    /// Smallest positive subnormal value.
    __BF16_DEVICE_FUNC
    static hie::bfloat16 denorm_min() {
        return hie::bfloat16(std::numeric_limits<bf16_impl_t>::denorm_min());
    }
};

}  // namespace std

#undef __BF16_IMPL
#undef __BF16_DEVICE_FUNC
#undef __BF16_DEVICE_FUNC_DECL

#endif  // UTILS_BFLOAT16_HPP_
