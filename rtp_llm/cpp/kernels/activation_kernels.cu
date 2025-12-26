/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

/* Gelu Activation */

__forceinline__ __device__ float copysignf_pos(float a, float b) {
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template<typename T>
struct GeluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val) {
        const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (val + 0.044715f * val * val * val))));
        return val * cdf;
    }
};

template<typename T>
struct GeluActivationNoneApproximate {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val) {
        return 0.5f * val * (1.0f + std::erf(val * M_SQRT1_2));
    }
};

template<>
struct GeluActivation<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val) {
        half2  val_pow3 = __hmul2(val, __hmul2(val, val));
        float2 tmp_pow  = __half22float2(val_pow3);
        float2 tmp      = __half22float2(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));

        return __hmul2(val, __float22half2_rn(tmp));
    }
};

template<>
struct GeluActivationNoneApproximate<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val) {
        half2  val_pow3 = __hmul2(val, __hmul2(val, val));
        float2 tmp_pow  = __half22float2(val_pow3);
        float2 tmp      = __half22float2(val);

        tmp.x = 0.5f * (1.0f + std::erf(tmp.x * M_SQRT1_2));
        tmp.y = 0.5f * (1.0f + std::erf(tmp.y * M_SQRT1_2));

        return __hmul2(val, __float22half2_rn(tmp));
    }
};

#ifdef ENABLE_BF16
template<>
struct GeluActivation<__nv_bfloat162> {
    using return_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val) {
        __nv_bfloat162 val_pow3 = bf16hmul2(val, bf16hmul2(val, val));
        float2         tmp_pow  = bf1622float2(val_pow3);
        float2         tmp      = bf1622float2(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
        return bf16hmul2(val, __floats2bfloat162_rn(tmp.x, tmp.y));
    }
};

template<>
struct GeluActivationNoneApproximate<__nv_bfloat162> {
    using return_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val) {
        __nv_bfloat162 val_pow3 = bf16hmul2(val, bf16hmul2(val, val));
        float2         tmp_pow  = bf1622float2(val_pow3);
        float2         tmp      = bf1622float2(val);

        tmp.x = 0.5f * (1.0f + std::erf(tmp.x * M_SQRT1_2));
        ;
        tmp.y = 0.5f * (1.0f + std::erf(tmp.y * M_SQRT1_2));

        return bf16hmul2(val, __floats2bfloat162_rn(tmp.x, tmp.y));
    }
};

#endif

/* Relu Activation */

template<typename T>
struct ReluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val) {
        return val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);
    }
};

template<>
struct ReluActivation<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val) {
        const half zero_half = static_cast<half>(0.0f);
        return make_half2(val.x > zero_half ? val.x : zero_half, val.y > zero_half ? val.y : zero_half);
    }
};

#ifdef ENABLE_BF16
template<>
struct ReluActivation<__nv_bfloat162> {
    using return_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val) {
        const __nv_bfloat16 zero_bf16 = static_cast<__nv_bfloat16>(0.0f);
        return make_bfloat162(val.x > zero_bf16 ? val.x : zero_bf16, val.y > zero_bf16 ? val.y : zero_bf16);
    }
};
#endif

/* Silu Activation */

template<typename T>
struct SiluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val) {
        return (T)((float)val / (1.0f + __expf((float)-val)));
    }
};

template<>
struct SiluActivation<half2> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const half2& val) {
        return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
    }
};

#ifdef ENABLE_BF16
template<>
struct SiluActivation<__nv_bfloat162> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const __nv_bfloat162& val) {
        return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
    }
};
#endif  // ENABLE_BF16

/* Identity Activation (= no activation) */

template<typename T>
struct IdentityActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val) {
        return val;
    }
};

// clang-format off
template<template<typename T> class Activation, typename T, typename BT>
__global__ void generic_activation(T*                      up_out,
                                   const BT*  __restrict   bias,
                                   const T*   __restrict   gate,
                                   const BT*  __restrict   gate_bias,
                                   const int* __restrict   ia3_tasks,
                                   const T*   __restrict   ia3_weights,
                                   const int               int8_mode,
                                   const float* __restrict activation_in,
                                   const float* __restrict activation_out,
                                   const BT* __restrict    activation_scale,
                                   const int* __restrict padding_offset,
                                   const int seq_len,
                                   int m,
                                   int n,
                                   int total)
{
    constexpr size_t packed_elems = num_elems<T>::value;

    const bool with_bias = bias != nullptr;
    const bool with_gate = gate != nullptr;
    const bool with_ia3  = ia3_tasks != nullptr;
    const bool with_act_scale = activation_scale != nullptr;

    using Act_T         = typename Activation<T>::return_type;
    using Float_T       = typename packed_as<float, packed_elems>::type;
    using Packed_Int8_t = typename packed_as<int8_t, packed_elems>::type;

    for (int64_t id = blockIdx.x * blockDim.x + threadIdx.x; id < total; id += blockDim.x * gridDim.x) {
        T val;
        if (int8_mode == 2) {
            val = cuda_cast<T>(cuda_cast<Float_T>(reinterpret_cast<Packed_Int8_t*>(up_out)[id]) * activation_in[0]);
        }
        else {
            val = up_out[id];
        }

        T gate_val;
        if (with_gate) {
            gate_val = gate[id];
        }

        if (with_bias) {
            const T reg_bias = static_cast<T>(bias[id % n]);
            val              = val + reg_bias;

            if (with_gate) {
                const T reg_gated_bias = static_cast<T>(gate_bias[id % n]);
                gate_val              = gate_val + reg_gated_bias;
            }
        }

        if (with_gate) {
            val = cuda_cast<T>(Activation<T>::apply(gate_val) * cuda_cast<Act_T>(val));
        }
        else {
            val = cuda_cast<T>(Activation<T>::apply(val));
        }

        if (with_ia3) {
            const int word_id = id / n;
            const int offset = padding_offset == nullptr ? 0 : padding_offset[word_id];
            const int batch_id = (word_id + offset) / seq_len;
            const int task = ia3_tasks[batch_id];
            val            = val * ia3_weights[task * n + (id % n)];
        }

        if (with_act_scale) {
            const T reg_activation = static_cast<T>(activation_scale[id % n]);
            val = val / reg_activation;
        }

        if (int8_mode != 2 ) {
            up_out[id] = val;
        }
        else {
            reinterpret_cast<Packed_Int8_t*>(up_out)[id] =
                cuda_cast<Packed_Int8_t>(cuda_cast<Float_T>(val) * activation_out[0]);
        }
    }
}
// clang-format on

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(T*           up_out,
                             const BT*    bias,
                             const T*     gate,
                             const BT*    gate_bias,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             const int    m,
                             const int    n,
                             const int    int8_mode,
                             const float* activation_in,
                             const float* activation_out,
                             const BT*    activation_scale,
                             const int*   padding_offset,
                             const int    seq_len,
                             cudaStream_t stream) {
    using PT                   = typename packed_type_2<T>::type;
    constexpr int packed_elems = num_elems<PT>::value;
    using PBT                  = typename packed_as<BT, packed_elems>::type;

    // should be even
    int temp_n = n + n % 2;

    dim3          block, grid;
    constexpr int max_threads_per_block = 1024;
    constexpr int elems_per_thread      = 4 * packed_elems;

    if (temp_n / elems_per_thread <= max_threads_per_block) {
        block.x = temp_n / elems_per_thread;
        grid.x  = m;
    } else {
        block.x                       = max_threads_per_block;
        constexpr int elems_per_block = max_threads_per_block * elems_per_thread;
        // grid.x  = (m * temp_n + elems_per_block - 1) / elems_per_block;
        int64_t total_elems = static_cast<int64_t>(m) * static_cast<int64_t>(temp_n);
        grid.x              = static_cast<int>((total_elems + elems_per_block - 1) / elems_per_block);
    }
    generic_activation<Activation><<<grid, block, 0, stream>>>(reinterpret_cast<PT*>(up_out),
                                                               reinterpret_cast<const PBT*>(bias),
                                                               reinterpret_cast<const PT*>(gate),
                                                               reinterpret_cast<const PBT*>(gate_bias),
                                                               ia3_tasks,
                                                               reinterpret_cast<const PT*>(ia3_weights),
                                                               int8_mode,
                                                               activation_in,
                                                               activation_out,
                                                               reinterpret_cast<const PBT*>(activation_scale),
                                                               padding_offset,
                                                               seq_len,
                                                               m,
                                                               temp_n / packed_elems,
                                                               m * temp_n / packed_elems);
}

#define INSTANTIATE_GENERIC_ACTIVATION(Activation, T, BT)                                                              \
    template void invokeGenericActivation<Activation, T, BT>(T * up_out,                                               \
                                                             const BT*    bias,                                        \
                                                             const T*     gate,                                        \
                                                             const BT*    gate_bias,                                   \
                                                             const int*   ia3_tasks,                                   \
                                                             const T*     ia3_weights,                                 \
                                                             const int    m,                                           \
                                                             const int    n,                                           \
                                                             const int    int8_mode,                                   \
                                                             const float* activation_in,                               \
                                                             const float* activation_out,                              \
                                                             const BT*    activation_scale,                            \
                                                             const int*   padding_offset,                              \
                                                             const int    seq_len,                                     \
                                                             cudaStream_t stream);

INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(GeluActivationNoneApproximate, float, float);
INSTANTIATE_GENERIC_ACTIVATION(GeluActivationNoneApproximate, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(GeluActivationNoneApproximate, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, half, half);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, __nv_bfloat16, __nv_bfloat16);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, __nv_bfloat16);
#endif
#undef INSTANCIATE_GENERIC_ACTIVATION

template<typename T>
__global__ void add_bias_tanh(T* out, const T* __restrict bias, int m, int n) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            val = val + ldg(&bias[id % n]);
        }
        out[id] = tanhf(val);
    }
}

template<>
__global__ void add_bias_tanh(half* out, const half* __restrict bias, int m, int n) {
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            val = val + __ldg(&bias_ptr[id % n]);
        }
        val.x       = tanhf(val.x);
        val.y       = tanhf(val.y);
        out_ptr[id] = val;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void add_bias_tanh(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n) {
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            val = bf16hadd2(val, ldg(&bias_ptr[id % n]));
        }
        val.x       = tanhf(val.x);
        val.y       = tanhf(val.y);
        out_ptr[id] = val;
    }
}
#endif

template<typename T>
void invokeAddBiasTanh(T* out, const T* bias, const int m, const int n, cudaStream_t stream) {
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    } else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    add_bias_tanh<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template void invokeAddBiasTanh(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasTanh(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasTanh(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

template<typename T2, int N>
__global__ void addBiasGeluV2(T2* out,
                              const T2* __restrict bias,
                              const int* ia3_tasks,
                              const T2*  ia3_weights,
                              const int  size,
                              const int* padding_offset,
                              const int  seq_len) {
    const bool with_ia3 = ia3_tasks != nullptr;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
        T2 val = out[id];
        if (bias != nullptr) {
            T2 reg_bias = ldg(&bias[id % N]);
            val         = hadd2(val, reg_bias);
        }
        val = GeluActivation<T2>::apply(val);
        if (with_ia3) {
            const int word_id  = id / N;
            const int offset   = padding_offset == nullptr ? 0 : padding_offset[word_id];
            const int batch_id = (word_id + offset) / seq_len;
            const int task     = ia3_tasks[batch_id];
            val                = val * ia3_weights[task * N + (id % N)];
        }
        out[id] = val;
    }
}

template<typename T2, int N, int ELEMENT_PER_ROUND>
__global__ void addBiasGeluV3(T2* out,
                              const T2* __restrict bias,
                              const int* ia3_tasks,
                              const T2*  ia3_weights,
                              const int  size,
                              const int* padding_offset,
                              const int  seq_len) {
    const bool with_ia3 = ia3_tasks != nullptr;
    T2         buffer[ELEMENT_PER_ROUND];
    T2         tmp_bias[ELEMENT_PER_ROUND];
    for (int id = blockIdx.x * blockDim.x * ELEMENT_PER_ROUND + threadIdx.x * ELEMENT_PER_ROUND; id < size;
         id += blockDim.x * gridDim.x * ELEMENT_PER_ROUND) {
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            buffer[i] = out[id + i];
            if (bias != nullptr) {
                tmp_bias[i] = ldg(&bias[(id + i) % N]);
            }
        }
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            if (bias != nullptr) {
                buffer[i] = hadd2(buffer[i], tmp_bias[i]);
            }
            buffer[i] = GeluActivation<T2>::apply(buffer[i]);
            if (with_ia3) {
                const int word_id  = (id + i) / N;
                const int offset   = padding_offset == nullptr ? 0 : padding_offset[word_id];
                const int batch_id = (word_id + offset) / seq_len;
                const int task     = ia3_tasks[batch_id];
                buffer[i]          = buffer[i] * ia3_weights[task * N + ((id + i) % N)];
            }
            out[id + i] = buffer[i];
        }
    }
}

#define ADD_BIAS_GELU(HALF_N, ELEMENT_PER_ROUND)                                                                       \
    case HALF_N:                                                                                                       \
        if (ELEMENT_PER_ROUND > 1) {                                                                                   \
            grid.x = grid.x / ELEMENT_PER_ROUND;                                                                       \
            addBiasGeluV3<T2, HALF_N, ELEMENT_PER_ROUND><<<grid, block, 0, stream>>>(                                  \
                (T2*)out, (const T2*)bias, ia3_tasks, (T2*)ia3_weights, m * half_n, padding_offset, seq_len);          \
        } else {                                                                                                       \
            addBiasGeluV2<T2, HALF_N><<<grid, block, 0, stream>>>(                                                     \
                (T2*)out, (const T2*)bias, ia3_tasks, (T2*)ia3_weights, m * half_n, padding_offset, seq_len);          \
        }                                                                                                              \
        break;

template<typename T>
void invokeAddBiasGeluV2(T*           out,
                         const T*     bias,
                         const int*   ia3_tasks,
                         const T*     ia3_weights,
                         const int*   padding_offset,
                         const int    seq_len,
                         const int    m,
                         const int    n,
                         cudaStream_t stream) {
    if (n % 2 == 0 && sizeof(T) == 2) {
        const int half_n = n / 2;
        dim3      block, grid;
        block.x  = std::min(half_n, 512);
        grid.x   = (m * half_n + (block.x - 1)) / block.x;
        using T2 = typename TypeConverter<T>::Type;

        if (grid.x >= 512) {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 2)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 4)
                default:
                    invokeGenericActivation<GeluActivation>(out,
                                                            bias,
                                                            (T*)nullptr,
                                                            (T*)nullptr,
                                                            ia3_tasks,
                                                            ia3_weights,
                                                            m,
                                                            n,
                                                            0,
                                                            (float*)nullptr,
                                                            (float*)nullptr,
                                                            (T*)nullptr,
                                                            padding_offset,
                                                            seq_len,
                                                            stream);
                    break;
            }
        } else {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 1)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 2)
                default:
                    invokeGenericActivation<GeluActivation>(out,
                                                            bias,
                                                            (T*)nullptr,
                                                            (T*)nullptr,
                                                            ia3_tasks,
                                                            ia3_weights,
                                                            m,
                                                            n,
                                                            0,
                                                            (float*)nullptr,
                                                            (float*)nullptr,
                                                            (T*)nullptr,
                                                            padding_offset,
                                                            seq_len,
                                                            stream);
                    break;
            }
        }
    } else {
        invokeGenericActivation<GeluActivation>(out,
                                                bias,
                                                (T*)nullptr,
                                                (T*)nullptr,
                                                ia3_tasks,
                                                ia3_weights,
                                                m,
                                                n,
                                                0,
                                                (float*)nullptr,
                                                (float*)nullptr,
                                                (T*)nullptr,
                                                padding_offset,
                                                seq_len,
                                                stream);
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#undef ADD_BIAS_GELU

template void invokeAddBiasGeluV2(float*       out,
                                  const float* bias,
                                  const int*   ia3_tasks,
                                  const float* ia3_weights,
                                  const int*   padding_offset,
                                  const int    seq_len,
                                  const int    m,
                                  const int    n,
                                  cudaStream_t stream);
template void invokeAddBiasGeluV2(half*        out,
                                  const half*  bias,
                                  const int*   ia3_tasks,
                                  const half*  ia3_weights,
                                  const int*   padding_offset,
                                  const int    seq_len,
                                  const int    m,
                                  const int    n,
                                  cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasGeluV2(__nv_bfloat16*       out,
                                  const __nv_bfloat16* bias,
                                  const int*           ia3_tasks,
                                  const __nv_bfloat16* ia3_weights,
                                  const int*           padding_offset,
                                  const int            seq_len,
                                  const int            m,
                                  const int            n,
                                  cudaStream_t         stream);
#endif  // ENABLE_BF16

template<typename T>
__global__ void sigmoid_kernel(T* data, const int size, const float scale) {
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        float val   = cuda_cast<float>(data[index]);
        val         = 1.0f / (1.0f + exp(-val)) * scale;
        data[index] = T(val);
    }
}

template<>
__global__ void sigmoid_kernel(half2* data, const int size, const float scale) {
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size / 2) {
        half2  val        = data[index];
        float2 val_float2 = cuda_cast<float2>(val);
        val_float2.x      = 1.0f / (1.0f + exp(-val_float2.x)) * scale;
        val_float2.y      = 1.0f / (1.0f + exp(-val_float2.y)) * scale;
        data[index]       = cuda_cast<half2>(val_float2);
    }
}
#ifdef ENABLE_BF16
template<>
__global__ void sigmoid_kernel(__nv_bfloat162* data, const int size, const float scale) {
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size / 2) {
        __nv_bfloat162 val        = data[index];
        float2         val_float2 = cuda_cast<float2>(val);
        val_float2.x              = 1.0f / (1.0f + exp(-val_float2.x)) * scale;
        val_float2.y              = 1.0f / (1.0f + exp(-val_float2.y)) * scale;
        data[index]               = cuda_cast<__nv_bfloat162>(val_float2);
    }
}
#endif

template<typename T>
void invokeSigmoid(T* data, const int size, const float scale, cudaStream_t stream) {
    if (std::is_same<T, half>::value && (size % 2 == 0)) {
        dim3 block(128);
        dim3 grid((size + 255) / 256);
        sigmoid_kernel<<<grid, block, 0, stream>>>((half2*)data, size, scale);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value && (size % 2 == 0)) {
        dim3 block(128);
        dim3 grid((size + 255) / 256);
        sigmoid_kernel<<<grid, block, 0, stream>>>((__nv_bfloat162*)data, size, scale);
    }
#endif
    else {
        dim3 block(128);
        dim3 grid((size + 127) / 128);
        sigmoid_kernel<<<grid, block, 0, stream>>>(data, size, scale);
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template void invokeSigmoid(float* data, const int size, const float scale, cudaStream_t stream);
template void invokeSigmoid(half* data, const int size, const float scale, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeSigmoid(__nv_bfloat16* data, const int size, const float scale, cudaStream_t stream);
#endif

template<typename T>
__global__ void scaledot_kernel(T* out, const T* in, const T* scale, const int m, const int n) {
    const int size        = m * n;
    const int index       = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const int scale_index = index / n;
    if (index < size) {
        float val       = cuda_cast<float>(in[index]);
        float scale_val = cuda_cast<float>(scale[scale_index]);
        out[index]      = T(val * scale_val);
    }
}

template<typename T>
void invokeScaledDot(T* out, const T* input, const T* scale, const int m, const int n, cudaStream_t stream) {
    int temp_n = n + n % 2;

    dim3 block, grid;
    if (temp_n <= 1024) {
        block.x = temp_n;
        grid.x  = m;
    } else {
        block.x = 1024;
        grid.x  = ceil(m * temp_n / 1024.);
    }
    scaledot_kernel<<<grid, block, 0, stream>>>(out, input, scale, m, n);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template void
invokeScaledDot(float* out, const float* input, const float* scale, const int m, const int n, cudaStream_t stream);
template void
invokeScaledDot(half* out, const half* input, const half* scale, const int m, const int n, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeScaledDot(__nv_bfloat16*       out,
                              const __nv_bfloat16* input,
                              const __nv_bfloat16* scale,
                              const int            m,
                              const int            n,
                              cudaStream_t         stream);
#endif

template<typename T>
__global__ void
addBiasSoftMax(T* logits, const T* bias, const int* end_ids, const bool* finished, const int n_padded, const int n) {
    int  bid    = blockIdx.x;
    bool finish = (finished != nullptr) ? finished[bid] : false;
    int  offset = bid * n_padded;

    float            max_val   = -1 * FLT_MAX;
    const bool       IS_FP16   = std::is_same<T, half>::value;
    const T          MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    __shared__ float s_max_val;
    __shared__ float s_sum_val;

    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x) {
        if (tid < n) {
            if (finish) {
                logits[offset + tid] = (tid == end_ids[bid]) ? static_cast<T>(MAX_T_VAL) : static_cast<T>(-MAX_T_VAL);
            } else {
                T bias_val = (bias != nullptr) ? bias[tid] : static_cast<T>(0.0f);
                logits[offset + tid] += bias_val;
            }
        } else {
            logits[offset + tid] = static_cast<T>(-MAX_T_VAL);
        }
        max_val = max(max_val, (float)logits[offset + tid]);
    }

    max_val = blockReduceMax<float>((float)max_val);
    if (threadIdx.x == 0) {
        s_max_val = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;
    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x) {
        logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
        sum_val += (float)logits[offset + tid];
    }

    sum_val = blockReduceSum<float>(sum_val);
    if (threadIdx.x == 0) {
        s_sum_val = sum_val;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x) {
        logits[offset + tid] = ((float)logits[offset + tid] / (s_sum_val + 1e-6f));
    }
}

template<typename T>
void invokeAddBiasSoftMax(T*           logits,
                          const T*     bias,
                          const int*   end_ids,
                          const bool*  finished,
                          const int    m,
                          const int    n_padded,
                          const int    n,
                          cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(min(n, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    addBiasSoftMax<<<grid, block, 0, stream>>>(logits, bias, end_ids, finished, n_padded, n);
}

template void invokeAddBiasSoftMax(float*       logits,
                                   const float* bias,
                                   const int*   end_ids,
                                   const bool*  finished,
                                   const int    m,
                                   const int    n_padded,
                                   const int    n,
                                   cudaStream_t stream);

template void invokeAddBiasSoftMax(half*        logits,
                                   const half*  bias,
                                   const int*   end_ids,
                                   const bool*  finished,
                                   const int    m,
                                   const int    n_padded,
                                   const int    n,
                                   cudaStream_t stream);

template void invokeAddBiasSoftMax(__nv_bfloat16*       logits,
                                   const __nv_bfloat16* bias,
                                   const int*           end_ids,
                                   const bool*          finished,
                                   const int            m,
                                   const int            n_padded,
                                   const int            n,
                                   cudaStream_t         stream);

}  // namespace rtp_llm
