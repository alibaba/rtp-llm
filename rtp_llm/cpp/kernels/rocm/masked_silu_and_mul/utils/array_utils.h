#pragma once

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bfloat16.h>
#else
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#endif

#include <cmath>
#include <cstdint>

namespace rtp_llm {
namespace utils {

constexpr float FP8_E4M3_MAX = 240.0f;
using fp8_e4m3_t = __hip_fp8_e4m3_fnuz;

template <typename T, int N>
struct Arr {
  using Element = T;
  static constexpr int kElements = N;
  T data[N];
  __host__ __device__       T& operator[](int i)       { return data[i]; }
  __host__ __device__ const T& operator[](int i) const { return data[i]; }
};

template <int N>
__host__ __device__ __forceinline__
Arr<float,N> operator*(const Arr<float,N>& a, const Arr<float,N>& b){
  Arr<float,N> o;
  #pragma unroll
  for (int i=0;i<N;++i) o[i]=a[i]*b[i];
  return o;
}

template <int N>
__host__ __device__ __forceinline__
Arr<float,N> operator*(const Arr<float,N>& a, float s){
  Arr<float,N> o;
  #pragma unroll
  for (int i=0;i<N;++i) o[i]=a[i]*s;
  return o;
}

template <int N>
struct SiLU {
  __host__ __device__ Arr<float,N> operator()(const Arr<float,N>& x) const {
    Arr<float,N> y;
    #pragma unroll
    for(int i=0;i<N;++i){ float v=x[i]; y[i]= v/(1.f+expf(-v)); }
    return y;
  }
};

template <class T, class U>
__host__ __device__ __forceinline__
U arrayConvert(T const& in) {
  static_assert(T::kElements==U::kElements,"kElements mismatch");
  using DType = typename U::Element;
  U u;
  #pragma unroll
  for (int i=0;i<U::kElements;++i) u[i]=static_cast<DType>(in[i]);
  return u;
}

template <int N>
__host__ __device__ __forceinline__
float max_abs(Arr<float,N> const& a){
  float m=0.f;
  #pragma unroll
  for (int i=0;i<N;++i){ float av=fabsf(a[i]); m = av>m? av: m; }
  return m;
}

enum class Fp8RoundMode { RNE, RNAZ, RTZ };

// 统一的舍入函数
template<Fp8RoundMode M>
__device__ __forceinline__ int qround(float x) {
  if constexpr (M == Fp8RoundMode::RTZ) {
    return (int)truncf(x);
  } else if constexpr (M == Fp8RoundMode::RNAZ) {
    float fx = floorf(fabsf(x));
    float frac = fabsf(x) - fx;
    int i = (int)fx + (frac > 0.5f || (frac == 0.5f)) ; // ties 远离 0
    return x < 0 ? -i : i;
  } else { // RNE
    float f = floorf(x);
    float frac = x - f;
    int i = (int)f;
    if (frac > 0.5f) return i + 1;
    if (frac < 0.5f) return i;
    // ties to even
    return (i & 1) ? (i + 1) : i;
  }
}

template<Fp8RoundMode M>
__device__ __forceinline__ uint8_t fp8_e4m3fnuz_encode(float x) {
  if (!isfinite(x)) return 0;
  uint32_t s = (x < 0.f) ? 1u : 0u;
  float ax = fabsf(x);

  const float SAT = 448.0f;
  if (ax >= SAT) return (uint8_t)((s << 7) | (0xFu << 3));

  const float MIN_SUB = 1.0f / 512.0f; // 2^-9
  if (ax < MIN_SUB) return 0;

  int exp2;
  float m = frexpf(ax, &exp2); // ax = m*2^exp2, m in [0.5,1)
  float k = m * 2.0f;
  int e = exp2 - 1;

  if (e < -6) {
    int mant = qround<M>(ax * 512.0f); // 2^9
    if (mant <= 0) return 0;
    if (mant > 7)  mant = 7;
    return (uint8_t)((s << 7) | (0u << 3) | (mant & 7));
  } else {
    int mant = qround<M>((k - 1.0f) * 8.0f);
    if (mant == 8) { mant = 0; ++e; }
    if (e > 7) return (uint8_t)((s << 7) | (0xFu << 3));
    uint8_t exp_field = (uint8_t)(e + 7);
    if (exp_field == 0) exp_field = 1;
    return (uint8_t)((s << 7) | (exp_field << 3) | (mant & 7));
  }
}

__device__ __forceinline__ fp8_e4m3_t to_e4m3fnuz(float x) {
  float c = fminf(fmaxf(x, -FP8_E4M3_MAX), FP8_E4M3_MAX);
  return static_cast<fp8_e4m3_t>(c);
}

template<int N>
__device__ __forceinline__ Arr<fp8_e4m3_t, N> pack_fp8_scaled(const Arr<float, N>& a, float inv_scale) {
  Arr<fp8_e4m3_t, N> out;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    out[i] = to_e4m3fnuz(a[i] * inv_scale);
    // out[i] = fp8_e4m3fnuz_encode<Fp8RoundMode::RNE>(a[i] * inv_scale);
  }
  return out;
}

template <typename T, int WARP_SIZE = 64>
__device__ __forceinline__ T blockReduceMax(T thread_val) {
  T warp_max = thread_val;
  #pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    T v = __shfl_xor(warp_max, offset, WARP_SIZE);
    warp_max = fmaxf(warp_max, v);
  }

  const int lane    = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  __shared__ T warp_red[1024 / WARP_SIZE];
  if (lane == 0) warp_red[warp_id] = warp_max;
  __syncthreads();

  T block_max = 0;
  if (warp_id == 0) {
    T val = (lane < num_warps) ? warp_red[lane] : 0;
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      T v = __shfl_xor(val, offset, WARP_SIZE);
      val = fmaxf(val, v);
    }
    if (lane == 0) warp_red[0] = val;
  }
  __syncthreads();
  return warp_red[0];
}

} // namespace utils
}  // namespace rtp_llm

