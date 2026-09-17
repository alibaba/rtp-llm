// Copyright 2026 Tencent
// Vendored utility subset for HY4 iHC kernels from hpc-ops (MIT).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

namespace rtp_llm {
namespace hy4_ihc {


// ============================
//    Debug Utility
// ============================

// print_type<T> _; // it will generate a compile time error with type T.
template <typename T>
struct print_type;

__device__ __forceinline__ void brkpt() { asm volatile("brkpt;" ::); }

// ============================
//    Load/Store(vectorized)
// ============================
template <typename T, int N>
struct vec_t {
  T data[N];

  using type = T;
  static constexpr int num = N;
  static constexpr int kNum = N;

  __device__ __forceinline__ constexpr T &operator[](int idx) { return data[idx]; }

  __device__ __forceinline__ constexpr const T &operator[](int idx) const { return data[idx]; }
};

template <typename T, int N, int... Dims>
struct traits_vec_t;

template <typename T, int N, int Dim>
struct traits_vec_t<T, N, Dim> {
  static_assert(N == Dim, "dimension mismatch");
  using type = vec_t<T, Dim>;
};

template <typename T, int N, int First, int... Rest>
struct traits_vec_t<T, N, First, Rest...> {
  static_assert(N % First == 0, "first dimension must divide total size");
  using inner_type = typename traits_vec_t<T, N / First, Rest...>::type;
  using type = vec_t<inner_type, First>;
};

template <typename T, int N>
__device__ __forceinline__ constexpr int size(vec_t<T, N> &v) {
  return N;
}

template <int... Dims, typename T, int N>
__device__ __forceinline__ constexpr auto &reshape(vec_t<T, N> &v) {
  constexpr int num_elements = (Dims * ...);

  using ResultType = typename traits_vec_t<T, N, Dims...>::type;
  return *reinterpret_cast<ResultType *>(&v);
}

template <typename U, typename T, int N>
__device__ __forceinline__ constexpr auto to(const vec_t<T, N> &v) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat16>) {
    using V = vec_t<__nv_bfloat16, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __float2bfloat16(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half2>) {
    using V = vec_t<U, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22half2_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __bfloat162float(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat162> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = __bfloat1622float2(v[i]);
      o[i * 2 + 0] = y.x;
      o[i * 2 + 1] = y.y;
    }
    return o;
  } else if constexpr (std::is_same_v<T, __half2> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = __half22float2(v[i]);
      o[i * 2 + 0] = y.x;
      o[i * 2 + 1] = y.y;
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_fp8x4_e4m3>) {
    static_assert(N % 4 == 0, "N % 4 must be 0");
    using V = vec_t<__nv_fp8x4_e4m3, N / 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 4; ++i) {
      o[i] = __nv_fp8x4_e4m3(*reinterpret_cast<const float4 *>(&v[4 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_fp8x4_e4m3> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = static_cast<float4>(v[i]);
      o[i * 4 + 0] = y.x;
      o[i * 4 + 1] = y.y;
      o[i * 4 + 2] = y.z;
      o[i * 4 + 3] = y.w;
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = static_cast<float>(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat162> && std::is_same_v<U, __nv_fp8x4_e4m3>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__nv_fp8x4_e4m3, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __nv_fp8x4_e4m3(*(reinterpret_cast<__nv_bfloat162 *>(&v[2 * i])),
                             *(reinterpret_cast<__nv_bfloat162 *>(&v[2 * i + 1])));
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat162>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__nv_bfloat162, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22bfloat162_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, U>) {
    return v;
  }
}

template <typename T, int N>
__device__ __forceinline__ constexpr auto load(const void *ptr) {
  using V = vec_t<T, N>;
  V v;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using L = uint8_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 2) {
    using L = uint16_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 4) {
    using L = uint32_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 8) {
    using L = uint64_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 16) {
    using L = uint4;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  }

  return v;
}

template <typename T, int N>
__device__ __forceinline__ constexpr void store(void *ptr, const vec_t<T, N> &v) {
  using V = vec_t<T, N>;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using S = uint8_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 2) {
    using S = uint16_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 4) {
    using S = uint32_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 8) {
    using S = uint64_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 16) {
    using S = uint4;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  }

  return;
}

template <typename T, typename... Args>
__device__ __forceinline__ constexpr void store(void *ptr, T val, Args... vals) {
  constexpr int N = sizeof...(Args);
  using V = vec_t<T, 1 + N>;

  static_assert((std::is_same_v<Args, T> && ...), "all vals must be type of T");

  V v;
  int idx = 0;
  (reinterpret_cast<T *>(&v))[idx++] = val;
  (((reinterpret_cast<T *>(&v))[idx++] = vals), ...);

  store(ptr, v);
}
// ============================
//       Fast Math API
// ============================

__device__ __forceinline__ float expf_ftz(float x) {
  // e^x = (2^m)^x
  // e = 2^m
  // m = lg2(e)
  // m = 1.4426950408889634

  const float m = 1.4426950408889634f;
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x * m));
  return r;
}

__device__ __forceinline__ float exp2f_ftz(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float log2f_ftz(float x) {
  float r;
  asm volatile("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float logf_ftz(float x) {
  // log(x) = lg2(x)log(2)
  // m = log(2) = 0.6931471805599453

  const float m = 0.6931471805599453f;
  float r;
  asm volatile("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r * m;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// y = 1 / (1 + e^(-x))
__device__ __forceinline__ float sigmoid(float x) { return rcpf_ftz(1.f + expf_ftz(-x)); }
__device__ __forceinline__ float sqrt_ftz(float x) {
  float r;
  asm volatile("sqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// y = max(0, x)
__device__ __forceinline__ float relu(float x) { return fmaxf(0, x); }

// y = x / (1 + e^(-x))
__device__ __forceinline__ float silu(float x) { return x * rcpf_ftz(1.f + expf_ftz(-x)); }

// y = log(1 + exp(x))
__device__ __forceinline__ float softplus(float x) { return logf_ftz(1.f + expf_ftz(x)); }

__device__ __forceinline__ float rsqrtf_ftz(float in) {
  float out;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(out) : "f"(in));
  return out;
}

__device__ __forceinline__ float warp_reduce_sum_down(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x += __shfl_down_sync(0xFFFFFFFF, x, ioffset);
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_max_down(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x = fmaxf(x, __shfl_down_sync(0xFFFFFFFF, x, ioffset));
  }

  return x;
}

__device__ __forceinline__ float half_warp_reduce_max_down(float x) {
  const int width = 16;

#pragma unroll
  for (int ioffset = width / 2; ioffset >= 1; ioffset /= 2) {
    x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFF, x, ioffset, width));
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_sum_xor(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x += __shfl_xor_sync(0xFFFFFFFF, x, ioffset);
  }

  return x;
}

}  // namespace hy4_ihc
}  // namespace rtp_llm
