#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

namespace atex {

using fp16_t   = half;
using fp16x2_t = half2;

using bf16_t   = bfloat16;
using bf16x2_t = bfloat162;

using fp32_t   = float;
using fp32x2_t = float2;
using fp32x4_t = float4;

using int16_t = short int;

using int8_t = char using int8x2_t = char2;
using int8x4_t                     = char4;

using uint8x2_t = uchar2;
using uint8x4_t = uchar4;

__device__ inline bool operator>(const fp16_t& a, const fp16_t& b) {
    return __hgt(a, b);
}

__device__ inline fp16_t operator+(const fp16_t& a, const fp16_t& b) {
    return __hadd(a, b);
}

__device__ inline fp16_t operator-(const fp16_t& a, const fp16_t& b) {
    return __hsub(a, b);
}

__device__ inline fp16_t operator*(const fp16_t& a, const fp16_t& b) {
    return __hmul(a, b);
}

__device__ inline fp16_t operator/(const fp16_t& a, const fp16_t& b) {
    return __hdiv(a, b);
}

__device__ inline fp16x2_t operator+(const fp16x2_t& a, const fp16x2_t& b) {
    return __hadd2(a, b);
}

__device__ inline fp16x2_t operator-(const fp16x2_t& a, const fp16x2_t& b) {
    return __hsub2(a, b);
}

__device__ inline fp16x2_t operator*(const fp16x2_t& a, const fp16x2_t& b) {
    return __hmul2(a, b);
}

__device__ inline fp16x2_t operator/(const fp16x2_t& a, const fp16x2_t& b) {
    return __hdiv2(a, b);
}

template<typename ScalarType>
__host__ inline ScalarType* PTR(at::Tensor t) {
    return t.data_ptr<ScalarType>();
}

template<>
__host__ inline fp16_t* PTR(at::Tensor t) {
    return reinterpret_cast<fp16_t*>(t.data_ptr<at::Half>());
}

template<>
__host__ inline fp16x2_t* PTR(at::Tensor t) {
    return reinterpret_cast<fp16x2_t*>(t.data_ptr<at::Half>());
}

template<>
__host__ inline int8x4_t* PTR(at::Tensor t) {
    return reinterpret_cast<int8x4_t*>(t.data_ptr());
}

template<>
__host__ inline int8x2_t* PTR(at::Tensor t) {
    return reinterpret_cast<int8x2_t*>(t.data_ptr());
}

template<>
__host__ inline int8_t* PTR(at::Tensor t) {
    return reinterpret_cast<int8_t*>(t.data_ptr());
}

}  // namespace atex
