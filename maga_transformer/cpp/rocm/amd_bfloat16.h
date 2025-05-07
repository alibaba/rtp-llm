#pragma once

#include <hip/hip_bf16.h>

#include "rocm/include/rocm-core/rocm_version.h"

#define ROCM_VERSION (ROCM_VERSION_MAJOR * 10000 + ROCM_VERSION_MINOR * 100 + ROCM_VERSION_PATCH)

#define __HOST_DEVICE_MEMBER__ __host__ __device__

struct amd_bfloat16: public __hip_bfloat16 {
    __HOST_DEVICE_MEMBER__ amd_bfloat16() = default;

    __HOST_DEVICE_MEMBER__ amd_bfloat16(__hip_bfloat16 other): __hip_bfloat16(other) {}

    __HOST_DEVICE_MEMBER__ amd_bfloat16(float f): amd_bfloat16(__float2bfloat16(f)) {}

#if ROCM_VERSION < 60300
    __HOST_DEVICE_MEMBER__ operator float() const {
        return __bfloat162float(*this);
    }

    __HOST_DEVICE_MEMBER__ amd_bfloat16& operator=(const float& f) {
        data = __float2bfloat16(f).data;
        return *this;
    }
#endif
};

#if ROCM_VERSION >= 60300
namespace __hip {
template<>
struct __numeric_type<amd_bfloat16> {
    static amd_bfloat16 __test(amd_bfloat16);

    typedef amd_bfloat16 type;
    static const bool    value = true;
};
}  // namespace __hip
#endif

#if ROCM_VERSION < 60300
__HOST_DEVICE__ amd_bfloat16 operator+(amd_bfloat16 a) {
    return a;
}
__HOST_DEVICE__ amd_bfloat16 operator-(amd_bfloat16 a) {
    a.data ^= 0x8000;
    return a;
}
__HOST_DEVICE__ amd_bfloat16 operator+(amd_bfloat16 a, amd_bfloat16 b) {
    return amd_bfloat16(float(a) + float(b));
}
__HOST_DEVICE__ amd_bfloat16 operator-(amd_bfloat16 a, amd_bfloat16 b) {
    return amd_bfloat16(float(a) - float(b));
}
__HOST_DEVICE__ amd_bfloat16 operator*(amd_bfloat16 a, amd_bfloat16 b) {
    return amd_bfloat16(float(a) * float(b));
}
__HOST_DEVICE__ amd_bfloat16 operator/(amd_bfloat16 a, amd_bfloat16 b) {
    return amd_bfloat16(float(a) / float(b));
}
__HOST_DEVICE__ bool operator<(amd_bfloat16 a, amd_bfloat16 b) {
    return float(a) < float(b);
}
__HOST_DEVICE__ bool operator==(amd_bfloat16 a, amd_bfloat16 b) {
    return float(a) == float(b);
}
__HOST_DEVICE__ bool operator>(amd_bfloat16 a, amd_bfloat16 b) {
    return b < a;
}
__HOST_DEVICE__ bool operator<=(amd_bfloat16 a, amd_bfloat16 b) {
    return !(a > b);
}
__HOST_DEVICE__ bool operator!=(amd_bfloat16 a, amd_bfloat16 b) {
    return !(a == b);
}
__HOST_DEVICE__ bool operator>=(amd_bfloat16 a, amd_bfloat16 b) {
    return !(a < b);
}
__HOST_DEVICE__ amd_bfloat16& operator++(amd_bfloat16& a) {
    return a = a + amd_bfloat16(1.0f);
}
__HOST_DEVICE__ amd_bfloat16& operator--(amd_bfloat16& a) {
    return a = a - amd_bfloat16(1.0f);
}
__HOST_DEVICE__ amd_bfloat16 operator++(amd_bfloat16& a, int) {
    amd_bfloat16 orig = a;
    ++a;
    return orig;
}
__HOST_DEVICE__ amd_bfloat16 operator--(amd_bfloat16& a, int) {
    amd_bfloat16 orig = a;
    --a;
    return orig;
}
#endif
template<typename T>
__HOST_DEVICE__ typename std::enable_if<std::is_base_of<__hip_bfloat16, T>::value, amd_bfloat16&>::type
operator+=(amd_bfloat16& a, T b) {
    return a = a + amd_bfloat16(b);
}
template<typename T>
__HOST_DEVICE__ typename std::enable_if<std::is_base_of<__hip_bfloat16, T>::value, amd_bfloat16&>::type
operator-=(amd_bfloat16& a, T b) {
    return a = a - amd_bfloat16(b);
}
template<typename T>
__HOST_DEVICE__ typename std::enable_if<std::is_base_of<__hip_bfloat16, T>::value, amd_bfloat16&>::type
operator*=(amd_bfloat16& a, T b) {
    return a = a * amd_bfloat16(b);
}
template<typename T>
__HOST_DEVICE__ typename std::enable_if<std::is_base_of<__hip_bfloat16, T>::value, amd_bfloat16&>::type
operator/=(amd_bfloat16& a, T b) {
    return a = a / amd_bfloat16(b);
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

#define make_bfloat162(a, b) amd_bfloat162(a, b)