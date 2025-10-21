#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <type_traits>

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE void fill(float_t val);

  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr);
};

template <>
struct vec_t<float, 1> {
  float data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE void fill(float val);

  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    data = (float)(*ptr);
  }

};

FLASHINFER_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE void fill(float val);

  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    data = make_float2((float)ptr[0], (float)ptr[1]);
  }

};

FLASHINFER_INLINE void vec_t<float, 2>::fill(float val) { data = make_float2(val, val); }

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }

  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4((float)ptr[i * 4],
                            (float)ptr[i * 4 + 1],
                            (float)ptr[i * 4 + 2],
                            (float)ptr[i * 4 + 3]);
    }
  }
};
