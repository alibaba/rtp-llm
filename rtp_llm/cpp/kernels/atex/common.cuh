#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>  // requires sm 80+
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

namespace atex {

const static uint32_t warpSize = 32;

using Tensor = at::Tensor;

using fp16_t   = __half;
using fp16x2_t = __half2;

using bf16_t   = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

using fp32_t   = float;
using fp32x2_t = float2;
using fp32x4_t = float4;

using int16_t = short int;

using int8_t   = char;
using int8x2_t = char2;
using int8x4_t = char4;

using uint8x2_t = uchar2;
using uint8x4_t = uchar4;

__device__ inline fp32_t cvt_f16_to_f32(fp16_t x) {
    return __half2float(x);
}

__device__ inline fp32_t cvt_f16_to_f32(bf16_t x) {
    return __bfloat162float(x);
}

__device__ inline fp32x2_t cvt_f16x2_to_f32x2(fp16x2_t x) {
    return __half22float2(x);
}

__device__ inline fp32x2_t cvt_f16x2_to_f32x2(bf16x2_t x) {
    return __bfloat1622float2(x);
}

template<typename dtype>
__device__ inline dtype cvt_f32x2_to_f16x2(fp32x2_t x) = delete;

template<>
__device__ inline fp16x2_t cvt_f32x2_to_f16x2<fp16x2_t>(fp32x2_t x) {
    return __float22half2_rn(x);
}
template<>
__device__ inline bf16x2_t cvt_f32x2_to_f16x2<bf16x2_t>(fp32x2_t x) {
    return __float22bfloat162_rn(x);
}

template<typename ScalarType>
__host__ inline ScalarType* PTR(at::Tensor t) = delete;

template<>
__host__ inline fp32_t* PTR(at::Tensor t) {
    return reinterpret_cast<fp32_t*>(t.data_ptr());
}

template<>
__host__ inline bf16_t* PTR(at::Tensor t) {
    return reinterpret_cast<bf16_t*>(t.data_ptr());
}

template<>
__host__ inline bf16x2_t* PTR(at::Tensor t) {
    return reinterpret_cast<bf16x2_t*>(t.data_ptr());
}

template<>
__host__ inline fp16_t* PTR(at::Tensor t) {
    return reinterpret_cast<fp16_t*>(t.data_ptr());
}

template<>
__host__ inline fp16x2_t* PTR(at::Tensor t) {
    return reinterpret_cast<fp16x2_t*>(t.data_ptr());
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
