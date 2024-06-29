#pragma once

#include <hip/hip_bf16.h>

#if defined(__HIPCC_RTC__)
    #define __HOST_DEVICE_MEMBER__ __device__
#else
    #define __HOST_DEVICE_MEMBER__ __host__ __device__
#endif

struct amd_bfloat16 : public __hip_bfloat16 {
private:
    using amd_bfloat16__uint16_t = unsigned short;
public:
    __HOST_DEVICE_MEMBER__ amd_bfloat16() = default;

    __HOST_DEVICE_MEMBER__ amd_bfloat16(__hip_bfloat16 other) {
        this->data = other.data;
    }

    // round upper 16 bits of IEEE float to convert to bfloat16
    // explicit __HOST_DEVICE_MEMBER__ amd_bfloat16(float f)
    __HOST_DEVICE_MEMBER__ amd_bfloat16(float f)
    {
        this->data = float_to_bfloat16(f);
    }

#if 0
    explicit __HOST_DEVICE__ amd_bfloat16(float f, truncate_t)
        : data(truncate_float_to_bfloat16(f))
    {
    }
#endif

    __HOST_DEVICE_MEMBER__ operator float() const
    {
        union
        {
            uint32_t int32;
            float    fp32;
        } u = {uint32_t(data) << 16};
        return u.fp32;
    }

    __HOST_DEVICE_MEMBER__ amd_bfloat16 &operator=(const float& f)
    {
       data = float_to_bfloat16(f);
       return *this;
    }

    static  __HOST_DEVICE_MEMBER__ amd_bfloat16 round_to_bfloat16(float f)
    {
        amd_bfloat16 output;
        output.data = float_to_bfloat16(f);
        return output;
    }

#if 0
    static  __HOST_DEVICE_MEMBER__ amd_bfloat16 round_to_bfloat16(float f, truncate_t)
    {
        amd_bfloat16 output;
        output.data = truncate_float_to_bfloat16(f);
        return output;
    }
#endif

private:
    static __HOST_DEVICE_MEMBER__ amd_bfloat16__uint16_t float_to_bfloat16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
            // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
            // This causes the bfloat16's mantissa to be incremented by 1 if the 16
            // least significant bits of the float mantissa are greater than 0x8000,
            // or if they are equal to 0x8000 and the least significant bit of the
            // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
            // has the value 0x7f, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.
            u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
        }
        else if(u.int32 & 0xffff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 16 bits of the mantissa are 1, we set the least significant bit
            // of the bfloat16 mantissa, in order to preserve signaling NaN in case
            // the bloat16's mantissa bits are all 0.
            u.int32 |= 0x10000; // Preserve signaling NaN
        }
        return amd_bfloat16__uint16_t(u.int32 >> 16);
    }

    // Truncate instead of rounding, preserving SNaN
    static __HOST_DEVICE_MEMBER__ amd_bfloat16__uint16_t truncate_float_to_bfloat16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        return amd_bfloat16__uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
    }
};

__HOST_DEVICE__ amd_bfloat16 operator+(amd_bfloat16 a)
{
    return a;
}
__HOST_DEVICE__ amd_bfloat16 operator-(amd_bfloat16 a)
{
    a.data ^= 0x8000;
    return a;
}
__HOST_DEVICE__ amd_bfloat16 operator+(amd_bfloat16 a, amd_bfloat16 b)
{
    return amd_bfloat16(float(a) + float(b));
}
__HOST_DEVICE__ amd_bfloat16 operator-(amd_bfloat16 a, amd_bfloat16 b)
{
    return amd_bfloat16(float(a) - float(b));
}
__HOST_DEVICE__ amd_bfloat16 operator*(amd_bfloat16 a, amd_bfloat16 b)
{
    return amd_bfloat16(float(a) * float(b));
}
__HOST_DEVICE__ amd_bfloat16 operator/(amd_bfloat16 a, amd_bfloat16 b)
{
    return amd_bfloat16(float(a) / float(b));
}
__HOST_DEVICE__ bool operator<(amd_bfloat16 a, amd_bfloat16 b)
{
    return float(a) < float(b);
}
__HOST_DEVICE__ bool operator==(amd_bfloat16 a, amd_bfloat16 b)
{
    return float(a) == float(b);
}
__HOST_DEVICE__ bool operator>(amd_bfloat16 a, amd_bfloat16 b)
{
    return b < a;
}
__HOST_DEVICE__ bool operator<=(amd_bfloat16 a, amd_bfloat16 b)
{
    return !(a > b);
}
__HOST_DEVICE__ bool operator!=(amd_bfloat16 a, amd_bfloat16 b)
{
    return !(a == b);
}
__HOST_DEVICE__ bool operator>=(amd_bfloat16 a, amd_bfloat16 b)
{
    return !(a < b);
}
__HOST_DEVICE__ amd_bfloat16& operator+=(amd_bfloat16& a, amd_bfloat16 b)
{
    return a = a + b;
}
__HOST_DEVICE__ amd_bfloat16& operator-=(amd_bfloat16& a, amd_bfloat16 b)
{
    return a = a - b;
}
__HOST_DEVICE__ amd_bfloat16& operator*=(amd_bfloat16& a, amd_bfloat16 b)
{
    return a = a * b;
}
__HOST_DEVICE__ amd_bfloat16& operator/=(amd_bfloat16& a, amd_bfloat16 b)
{
    return a = a / b;
}
__HOST_DEVICE__ amd_bfloat16& operator++(amd_bfloat16& a)
{
    return a += amd_bfloat16(1.0f);
}
__HOST_DEVICE__ amd_bfloat16& operator--(amd_bfloat16& a)
{
    return a -= amd_bfloat16(1.0f);
}
__HOST_DEVICE__ amd_bfloat16 operator++(amd_bfloat16& a, int)
{
    amd_bfloat16 orig = a;
    ++a;
    return orig;
}
__HOST_DEVICE__ amd_bfloat16 operator--(amd_bfloat16& a, int)
{
    amd_bfloat16 orig = a;
    --a;
    return orig;
}

struct amd_bfloat162 {

  __HOST_DEVICE_MEMBER__ amd_bfloat162() = default;

  __HOST_DEVICE_MEMBER__ amd_bfloat162(__hip_bfloat162 other) {
    this->x = other.x;
    this->y = other.y;
  }

  __HOST_DEVICE_MEMBER__ amd_bfloat162(amd_bfloat16 _x, amd_bfloat16 _y) {
    this->x = _x;
    this->y = _y;
  }

  __HOST_DEVICE_MEMBER__ operator __hip_bfloat162() const {
    return {x, y};
  }

  amd_bfloat16 x;
  amd_bfloat16 y;
};
