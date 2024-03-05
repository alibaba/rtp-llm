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

#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

template<typename T>
__device__ __inline__ T __ldg_func(const T* ptr) {
    return __ldg(ptr);
}

#ifdef ENABLE_BF16
template<>
__device__ __inline__ __nv_bfloat16 __ldg_func<__nv_bfloat16>(const __nv_bfloat16* ptr) {
#if __CUDA_ARCH__ >= 800
    return __ldg(ptr);
#else
    return *ptr;
#endif
}
#endif

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(T* normed_output,
                                                   T* output,
                                                   const T* __restrict input,
                                                   const T* __restrict bias,
                                                   const T* __restrict residual1,
                                                   const T* __restrict gamma,
                                                   const T* __restrict beta,
                                                   const float  layernorm_eps,
                                                   int          m,
                                                   int          n,
                                                   const float* scale_inter,
                                                   const float* scale_out,
                                                   const float* scale,
                                                   float*       dynamic_scale,
                                                   const int    int8_mode) {
    extern __shared__ __align__(sizeof(float)) char _shmem[];  // Align on largest type
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const bool scale_input     = int8_mode == 2 && scale_inter != nullptr;
    const bool dynamic_scaling = dynamic_scale != nullptr;

    T local_sum = cuda_cast<T>(0.0f);

    const Float_Packed_T scale_from_int = cuda_cast<Float_Packed_T>(scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
    const Float_Packed_T scale_to_int   = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = cuda_cast<T>(0.0f);

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        if (RESIDUAL) {
            val = hadd2(val, ldg(&residual1[index]));
        }

        if (IS_OUTPUT) {
            T in_val;
            if (scale_input) {
                in_val = cuda_cast<T>(cuda_cast<Float_Packed_T>(reinterpret_cast<const Int32_Packed_T*>(input)[index])
                                      * scale_from_int);
            } else {
                in_val = input[index];
            }
            val = hadd2(val, in_val);
        }
        shmem[i]      = val;
        output[index] = val;
        local_sum     = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / n / 2;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T     val    = input[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n / 2 + layernorm_eps);
    }
    __syncthreads();

    T mean_2 = cuda_cast<T>(s_mean);
    T var_2  = cuda_cast<T>(s_variance);

    Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }

        if (dynamic_scaling) {
            abs_max  = cuda_max(cuda_max<Scalar_T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        } else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_to_int);
        } else {
            normed_output[index] = val;
        }
    }

    if (dynamic_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const float dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int index                                        = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,
                                                    T* output,
                                                    const T* __restrict input,
                                                    const T* __restrict bias,
                                                    const T* __restrict residual1,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n,
                                                    const float* scale_inter,
                                                    const float* scale_out,
                                                    const float* scale,
                                                    float*       dynamic_scale,
                                                    const int    int8_mode) {
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            x_sum    = 0.0f;
    float            x2_sum   = 0.0f;
    const int        b_offset = blockIdx.x * n;

    using T1             = typename TypeConverter<T>::Type;
    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const bool           scale_input  = int8_mode == 2 && scale_inter != nullptr;
    const Float_Packed_T scale_vec_in = cuda_cast<Float_Packed_T>(scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
    const Float_Packed_T scale_vec    = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);
    const bool           dynamic_scaling = dynamic_scale != nullptr;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float     val_1 = 0.0f;
        float     val_2 = 0.0f;
        T         tmp;

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        if (RESIDUAL) {
            tmp = ldg(&residual1[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }

        if (IS_OUTPUT) {
            if (scale_input) {
                tmp = cuda_cast<T>(cuda_cast<Float_Packed_T>(reinterpret_cast<const Int32_Packed_T*>(input)[index])
                                   * scale_vec_in);
            } else {
                tmp = ldg(&input[index]);
            }
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        tmp.x         = cuda_cast<T1>(val_1);
        tmp.y         = cuda_cast<T1>(val_2);
        shmem[i]      = tmp;
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean     = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + layernorm_eps);
    }
    __syncthreads();

    T mean_2 = cuda_cast<T>(s_mean);
    T var_2  = cuda_cast<T>(s_variance);

    Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T         val   = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }

        if (dynamic_scaling) {
            abs_max  = cuda_max(cuda_max<Scalar_T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        } else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_vec);
        } else {
            normed_output[index] = val;
        }
    }

    if (dynamic_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const float dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int index                                        = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

// TODO(bhsueh) add half2 implementation
template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = (float)(add(out[blockIdx.x * n + idx], input[blockIdx.x * n + idx], ldg(&bias[idx])));
        mean += local_out;
        // save local_out to local_out_cache to save some recompute
        local_out_cache[i] = local_out;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + layernorm_eps;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        out[blockIdx.x * n + idx] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(ldg(&gamma[idx])) + (float)(ldg(&beta[idx])));
        idx += blockDim.x;
    }
}

template<int N>
__global__ void addBiasResidualPostLayerNormHalf(half*       out,
                                                 const half* input,
                                                 const half* bias,
                                                 const half* gamma,
                                                 const half* beta,
                                                 const float layernorm_eps,
                                                 int         m,
                                                 int         n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    half2*       out_ptr   = (half2*)out;
    const half2* input_ptr = (const half2*)input;
    const half2* bias_ptr  = (const half2*)bias;
    const half2* gamma_ptr = (const half2*)gamma;
    const half2* beta_ptr  = (const half2*)beta;

    float2 out_fp2_cache[N];

    float local_out = 0.0f;
#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n / 2 && i < N; ++i) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(__hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg_func(&bias_ptr[idx])));
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save local_out_fp2 to out_fp2_cache to save some recomputation
        out_fp2_cache[i] = local_out_fp2;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; i < N && idx < n / 2; ++i) {
        float2 local_out_fp2 = out_fp2_cache[i];
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
        idx += blockDim.x;
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; i < N && idx < n / 2; ++i) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = out_fp2_cache[i];
        float2 gamma_val     = __half22float2(__ldg_func(&gamma_ptr[idx]));
        float2 beta_val      = __half22float2(__ldg_func(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = __float22half2_rn(local_out_fp2);
        idx += blockDim.x;
    }
}

// Optimization for fp16 and fp16 (bf162 and half2)
template<typename T>
__global__ void generalAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n) {
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* bias_ptr  = (const T2*)bias;
    const T2* gamma_ptr = (const T2*)gamma;
    const T2* beta_ptr  = (const T2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        T2     tmp           = hadd2(hadd2(out_ptr[id], input_ptr[id]), ldg(&bias_ptr[idx]));
        float2 local_out_fp2 = cuda_cast<float2>(tmp);
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save tmp to out_ptr to save some recomputation
        out_ptr[id] = tmp;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = cuda_cast<float2>(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = cuda_cast<float2>(out_ptr[id]);
        float2 gamma_val     = cuda_cast<float2>(ldg(&gamma_ptr[idx]));
        float2 beta_val      = cuda_cast<float2>(ldg(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = cuda_cast<T2>(local_out_fp2);
    }
}

template<>
__global__ void generalAddBiasResidualPostLayerNorm(float*       out,
                                                    const float* input,
                                                    const float* bias,
                                                    const float* gamma,
                                                    const float* beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = (float)(out[blockIdx.x * n + idx] + input[blockIdx.x * n + idx] + __ldg_func(&bias[idx]));
        mean += local_out;
        // save local_out to out to save some recompute
        out[blockIdx.x * n + idx] = local_out;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out           = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] = (float)(((local_out - s_mean) * s_variance) * (float)(__ldg_func(&gamma[idx]))
                                            + (float)(__ldg_func(&beta[idx])));
    }
}

// applied to half and b16
template<typename T>
__global__ void addBiasResidualPostLayerNormV2(T* out,
                                               const T* __restrict input,
                                               const T* __restrict bias,
                                               const T* __restrict gamma,
                                               const T* __restrict beta,
                                               const float layernorm_eps,
                                               int         n) {
    using T2             = typename TypeConverter<T>::Type;
    const int        ite = 4;
    const int        tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    T2               local_out_half2[ite];

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* bias_ptr  = (const T2*)bias;
    const T2* gamma_ptr = (const T2*)gamma;
    const T2* beta_ptr  = (const T2*)beta;

    // float sum = 0.0f;
    T2 sum = cuda_cast<T2>(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id         = i * blockDim.x + tid;
        int id             = bid * n / 2 + col_id;
        local_out_half2[i] = add(out_ptr[id], ldg(&input_ptr[id]), ldg(&bias_ptr[col_id]));
        sum                = add(sum, local_out_half2[i]);
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var      = 0.0f;
    T2    s_mean_2 = cuda_cast<T2>(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = hsub2(local_out_half2[i], s_mean_2);
        float v1           = (float)local_out_half2[i].x;
        float v2           = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    T2 s_var_2 = cuda_cast<T2>(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id  = i * blockDim.x + tid;
        int id      = bid * n / 2 + col_id;
        out_ptr[id] = fma(local_out_half2[i], s_var_2, ldg(&gamma_ptr[col_id]), ldg(&beta_ptr[col_id]));
    }
}

template<>
__global__ void addBiasResidualPostLayerNormV2(float* out,
                                               const float* __restrict input,
                                               const float* __restrict bias,
                                               const float* __restrict gamma,
                                               const float* __restrict beta,
                                               const float layernorm_eps,
                                               int         n) {
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id   = i * blockDim.x + tid;
        int id       = bid * n + col_id;
        local_out[i] = (float)(out[id] + __ldg_func(&input[id]) + __ldg_func(&bias[col_id]));
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id     = bid * n + col_id;
        out[id]    = (float)((local_out[i] - s_mean) * s_variance * (float)__ldg_func(&gamma[col_id])
                          + (float)__ldg_func(&beta[col_id]));
    }
}

// bf16 and half data type
template<typename T>
void invokeAddBiasResidualLayerNorm(T*           out,
                                    const T*     input,
                                    const T*     bias,
                                    const T*     gamma,
                                    const T*     beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (m >= 512 && (n == 768 || n == 1024)) {
        addBiasResidualPostLayerNormV2<T><<<grid, n / 8, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<T, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<T, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else {
            generalAddBiasResidualPostLayerNorm<T>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

template<>
void invokeAddBiasResidualLayerNorm(float*       out,
                                    const float* input,
                                    const float* bias,
                                    const float* gamma,
                                    const float* beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2<float>
            <<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<float, 1>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<float, 2>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        } else {
            generalAddBiasResidualPostLayerNorm<float>
                <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

#define INVOKE_ADD_BIAS_RES_LN(T)                                                                                      \
    template void invokeAddBiasResidualLayerNorm(T*           out,                                                     \
                                                 const T*     input,                                                   \
                                                 const T*     bias,                                                    \
                                                 const T*     gamma,                                                   \
                                                 const T*     beta,                                                    \
                                                 const float  layernorm_eps,                                           \
                                                 int          m,                                                       \
                                                 int          n,                                                       \
                                                 cudaStream_t stream);
INVOKE_ADD_BIAS_RES_LN(float)
INVOKE_ADD_BIAS_RES_LN(half)
#ifdef ENABLE_BF16
INVOKE_ADD_BIAS_RES_LN(__nv_bfloat16)
#endif

template<typename T, bool RESIDUAL>
__global__ void generalAddBiasResidualLayerNorm(const T* __restrict input,
                                                const T* __restrict residual1,
                                                const T* __restrict gamma,
                                                const T* __restrict beta,
                                                const T* __restrict bias,
                                                T*           output,
                                                T*           norm_output,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                const float* scale_inter,
                                                const float* scale_out,
                                                const float* scale,
                                                float*       dynamic_scale,
                                                const int    int8_mode) {
    int tid = threadIdx.x;

    // NOTE: float shmem may exceed the shared memory limit
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const bool dynamic_scaling = dynamic_scale != nullptr;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    const bool  is_input_i32  = int8_mode == 2 && scale_inter != nullptr && scale_out != nullptr;
    const float scale_out_val = is_input_i32 ? (*scale_inter) * (*scale_out) : 0.0f;
    float       local_sum     = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float local_out = 0.0f;
        if (RESIDUAL) {
            local_out = (float)(ldg(&residual1[blockIdx.x * n + i]));
        }
        if (is_input_i32) {
            local_out += cuda_cast<float>(reinterpret_cast<const int32_t*>(input)[blockIdx.x * n + i]) * scale_out_val;
        } else {
            local_out += (float)(input[blockIdx.x * n + i]);
        }

        if (bias != nullptr) {
            local_out += (float)(ldg(&bias[i]));
        }
        shmem[i]                   = (T)local_out;
        output[blockIdx.x * n + i] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    Scalar_T abs_max = 1e-6f;

    const float scale_val = int8_mode == 2 ? *scale : 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float       beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        const float val      = ((((float)shmem[i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

        if (dynamic_scaling) {
            abs_max  = cuda_max(cuda_max<Scalar_T, float>(cuda_abs(val)), abs_max);
            shmem[i] = (T)val;
        } else if (int8_mode == 2) {
            reinterpret_cast<int8_t*>(norm_output)[blockIdx.x * n + i] = cuda_cast<int8_t>(val * scale_val);
        } else {
            norm_output[blockIdx.x * n + i] = (T)val;
        }
    }

    if (dynamic_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const float dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = tid; i < n; i += blockDim.x) {
            const int index                                      = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(norm_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

template<typename T, bool IS_OUTPUT, bool IS_BIAS, int UNROLL_FACTOR, bool RESIDUAL>
void dispatch_generalAddBiasResidualLayerNormOpt_opt_version(T*           norm_output,
                                                             T*           output,
                                                             const T*     input,
                                                             const T*     bias,
                                                             const T*     residual1,
                                                             const T*     gamma,
                                                             const T*     beta,
                                                             float        layernorm_eps,
                                                             int          m,
                                                             int          half_n,
                                                             const float* scale_inter,
                                                             const float* scale_out,
                                                             const float* scale,
                                                             float*       dynamic_scale,
                                                             int          int8_mode,
                                                             dim3         grid,
                                                             dim3         block,
                                                             cudaStream_t stream,
                                                             int          opt_version) {
    size_t maxbytes = half_n * sizeof(T);
    if (opt_version == 1) {
        if (maxbytes >= (48 << 10)) {
            check_cuda_error(cudaFuncSetAttribute(
                generalAddBiasResidualLayerNormOpt<T, IS_OUTPUT, IS_BIAS, RESIDUAL, true, UNROLL_FACTOR>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxbytes));
        }
        generalAddBiasResidualLayerNormOpt<T, IS_OUTPUT, IS_BIAS, RESIDUAL, true, UNROLL_FACTOR>
            <<<grid, block, maxbytes, stream>>>(norm_output,
                                                output,
                                                input,
                                                bias,
                                                residual1,
                                                gamma,
                                                beta,
                                                layernorm_eps,
                                                m,
                                                half_n,
                                                scale_inter,
                                                scale_out,
                                                scale,
                                                dynamic_scale,
                                                int8_mode);
    } else if (opt_version == 2) {
        if (maxbytes >= (48 << 10)) {
            check_cuda_error(cudaFuncSetAttribute(
                generalAddBiasResidualLayerNormOpt2<T, IS_OUTPUT, IS_BIAS, RESIDUAL, true, UNROLL_FACTOR>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxbytes));
        }
        generalAddBiasResidualLayerNormOpt2<T, IS_OUTPUT, IS_BIAS, RESIDUAL, true, UNROLL_FACTOR>
            <<<grid, block, maxbytes, stream>>>(norm_output,
                                                output,
                                                input,
                                                bias,
                                                residual1,
                                                gamma,
                                                beta,
                                                layernorm_eps,
                                                m,
                                                half_n,
                                                scale_inter,
                                                scale_out,
                                                scale,
                                                dynamic_scale,
                                                int8_mode);
    } else {
        FT_CHECK_WITH_INFO(false, "opt_num must be 1 or 2");
    }
}

template<typename T, bool IS_BIAS, int UNROLL_FACTOR, bool RESIDUAL>
void dispatch_generalAddBiasResidualLayerNormOpt_is_output(T*           norm_output,
                                                           T*           output,
                                                           const T*     input,
                                                           const T*     bias,
                                                           const T*     residual1,
                                                           const T*     gamma,
                                                           const T*     beta,
                                                           float        layernorm_eps,
                                                           int          m,
                                                           int          half_n,
                                                           const float* scale_inter,
                                                           const float* scale_out,
                                                           const float* scale,
                                                           float*       dynamic_scale,
                                                           int          int8_mode,
                                                           dim3         grid,
                                                           dim3         block,
                                                           cudaStream_t stream,
                                                           int          opt_version,
                                                           bool         is_output) {
    if (is_output) {
        dispatch_generalAddBiasResidualLayerNormOpt_opt_version<T, true, IS_BIAS, UNROLL_FACTOR, RESIDUAL>(
            norm_output,
            output,
            input,
            bias,
            residual1,
            gamma,
            beta,
            layernorm_eps,
            m,
            half_n,
            scale_inter,
            scale_out,
            scale,
            dynamic_scale,
            int8_mode,
            grid,
            block,
            stream,
            opt_version);
    } else {
        dispatch_generalAddBiasResidualLayerNormOpt_opt_version<T, false, IS_BIAS, UNROLL_FACTOR, RESIDUAL>(
            norm_output,
            output,
            input,
            bias,
            residual1,
            gamma,
            beta,
            layernorm_eps,
            m,
            half_n,
            scale_inter,
            scale_out,
            scale,
            dynamic_scale,
            int8_mode,
            grid,
            block,
            stream,
            opt_version);
    }
}

template<typename T, int UNROLL_FACTOR>
void dispatch_generalAddBiasResidualLayerNormOpt_bias(T*           norm_output,
                                                      T*           output,
                                                      const T*     input,
                                                      const T*     bias,
                                                      const T*     residual1,
                                                      const T*     gamma,
                                                      const T*     beta,
                                                      float        layernorm_eps,
                                                      int          m,
                                                      int          half_n,
                                                      const float* scale_inter,
                                                      const float* scale_out,
                                                      const float* scale,
                                                      float*       dynamic_scale,
                                                      int          int8_mode,
                                                      dim3         grid,
                                                      dim3         block,
                                                      cudaStream_t stream,
                                                      int          opt_version,
                                                      bool         is_output) {
    if (bias != nullptr) {
        if (residual1 != nullptr) {
            dispatch_generalAddBiasResidualLayerNormOpt_is_output<T, true, UNROLL_FACTOR, true>(norm_output,
                                                                                                output,
                                                                                                input,
                                                                                                bias,
                                                                                                residual1,
                                                                                                gamma,
                                                                                                beta,
                                                                                                layernorm_eps,
                                                                                                m,
                                                                                                half_n,
                                                                                                scale_inter,
                                                                                                scale_out,
                                                                                                scale,
                                                                                                dynamic_scale,
                                                                                                int8_mode,
                                                                                                grid,
                                                                                                block,
                                                                                                stream,
                                                                                                opt_version,
                                                                                                is_output);
        } else {
            dispatch_generalAddBiasResidualLayerNormOpt_is_output<T, true, UNROLL_FACTOR, false>(norm_output,
                                                                                                 output,
                                                                                                 input,
                                                                                                 bias,
                                                                                                 residual1,
                                                                                                 gamma,
                                                                                                 beta,
                                                                                                 layernorm_eps,
                                                                                                 m,
                                                                                                 half_n,
                                                                                                 scale_inter,
                                                                                                 scale_out,
                                                                                                 scale,
                                                                                                 dynamic_scale,
                                                                                                 int8_mode,
                                                                                                 grid,
                                                                                                 block,
                                                                                                 stream,
                                                                                                 opt_version,
                                                                                                 is_output);
        }
    } else {
        if (residual1 != nullptr) {
            dispatch_generalAddBiasResidualLayerNormOpt_is_output<T, false, UNROLL_FACTOR, true>(norm_output,
                                                                                                 output,
                                                                                                 input,
                                                                                                 bias,
                                                                                                 residual1,
                                                                                                 gamma,
                                                                                                 beta,
                                                                                                 layernorm_eps,
                                                                                                 m,
                                                                                                 half_n,
                                                                                                 scale_inter,
                                                                                                 scale_out,
                                                                                                 scale,
                                                                                                 dynamic_scale,
                                                                                                 int8_mode,
                                                                                                 grid,
                                                                                                 block,
                                                                                                 stream,
                                                                                                 opt_version,
                                                                                                 is_output);
        } else {
            dispatch_generalAddBiasResidualLayerNormOpt_is_output<T, false, UNROLL_FACTOR, false>(norm_output,
                                                                                                  output,
                                                                                                  input,
                                                                                                  bias,
                                                                                                  residual1,
                                                                                                  gamma,
                                                                                                  beta,
                                                                                                  layernorm_eps,
                                                                                                  m,
                                                                                                  half_n,
                                                                                                  scale_inter,
                                                                                                  scale_out,
                                                                                                  scale,
                                                                                                  dynamic_scale,
                                                                                                  int8_mode,
                                                                                                  grid,
                                                                                                  block,
                                                                                                  stream,
                                                                                                  opt_version,
                                                                                                  is_output);
        }
    }
}

template<typename T>
void dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor(T*           norm_output,
                                                               T*           output,
                                                               const T*     input,
                                                               const T*     bias,
                                                               const T*     residual1,
                                                               const T*     gamma,
                                                               const T*     beta,
                                                               float        layernorm_eps,
                                                               int          m,
                                                               int          half_n,
                                                               const float* scale_inter,
                                                               const float* scale_out,
                                                               const float* scale,
                                                               float*       dynamic_scale,
                                                               int          int8_mode,
                                                               dim3         grid,
                                                               dim3         block,
                                                               cudaStream_t stream,
                                                               int          opt_version,
                                                               bool         is_output,
                                                               int          unroll_factor) {
    switch (unroll_factor) {
        case 1:
            dispatch_generalAddBiasResidualLayerNormOpt_bias<T, 1>(norm_output,
                                                                   output,
                                                                   input,
                                                                   bias,
                                                                   residual1,
                                                                   gamma,
                                                                   beta,
                                                                   layernorm_eps,
                                                                   m,
                                                                   half_n,
                                                                   scale_inter,
                                                                   scale_out,
                                                                   scale,
                                                                   dynamic_scale,
                                                                   int8_mode,
                                                                   grid,
                                                                   block,
                                                                   stream,
                                                                   opt_version,
                                                                   is_output);
            break;
        case 2:
            dispatch_generalAddBiasResidualLayerNormOpt_bias<T, 2>(norm_output,
                                                                   output,
                                                                   input,
                                                                   bias,
                                                                   residual1,
                                                                   gamma,
                                                                   beta,
                                                                   layernorm_eps,
                                                                   m,
                                                                   half_n,
                                                                   scale_inter,
                                                                   scale_out,
                                                                   scale,
                                                                   dynamic_scale,
                                                                   int8_mode,
                                                                   grid,
                                                                   block,
                                                                   stream,
                                                                   opt_version,
                                                                   is_output);
            break;
        case 4:
            dispatch_generalAddBiasResidualLayerNormOpt_bias<T, 4>(norm_output,
                                                                   output,
                                                                   input,
                                                                   bias,
                                                                   residual1,
                                                                   gamma,
                                                                   beta,
                                                                   layernorm_eps,
                                                                   m,
                                                                   half_n,
                                                                   scale_inter,
                                                                   scale_out,
                                                                   scale,
                                                                   dynamic_scale,
                                                                   int8_mode,
                                                                   grid,
                                                                   block,
                                                                   stream,
                                                                   opt_version,
                                                                   is_output);
            break;
        case 8:
            dispatch_generalAddBiasResidualLayerNormOpt_bias<T, 8>(norm_output,
                                                                   output,
                                                                   input,
                                                                   bias,
                                                                   residual1,
                                                                   gamma,
                                                                   beta,
                                                                   layernorm_eps,
                                                                   m,
                                                                   half_n,
                                                                   scale_inter,
                                                                   scale_out,
                                                                   scale,
                                                                   dynamic_scale,
                                                                   int8_mode,
                                                                   grid,
                                                                   block,
                                                                   stream,
                                                                   opt_version,
                                                                   is_output);
            break;
        default:
            FT_CHECK_WITH_INFO(false, "unroll_factor must be 1, 2, 4 or 8");
    }
}

/* output      <- output + bias + residual_1 + residual_2
 * output_norm <- LN(output) */
template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T*           output,
                                              T*           norm_output,
                                              const T*     input,
                                              const T*     residual1,
                                              const T*     gamma,
                                              const T*     beta,
                                              const T*     bias,
                                              const float  layernorm_eps,
                                              int          m,
                                              int          n,
                                              const float* scale_inter,
                                              const float* scale_out,
                                              float*       scale,
                                              float*       dynamic_scale,
                                              const int    int8_mode,
                                              cudaStream_t stream,
                                              int          opt_version) {
    if (opt_version > 0 && sizeof(T) == 2 && n % 2 == 0) {
        dim3 grid(m);
        int  half_n    = n / 2;
        int  half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int  rolls_per_thread = half_n / block.x;
        int  unroll_factor    = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }

        using T2 = typename TypeConverter<T>::Type;

        /* we launch (and instantiate) the kernel by specializing for unroll_factor -> residual_num -> is_bias ->
         * opt_version */
        dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor((T2*)norm_output,
                                                                  (T2*)output,
                                                                  (const T2*)input,
                                                                  (const T2*)bias,
                                                                  (const T2*)residual1,
                                                                  (const T2*)gamma,
                                                                  (const T2*)beta,
                                                                  layernorm_eps,
                                                                  m,
                                                                  half_n,
                                                                  scale_inter,
                                                                  scale_out,
                                                                  scale,
                                                                  dynamic_scale,
                                                                  int8_mode,
                                                                  grid,
                                                                  block,
                                                                  stream,
                                                                  opt_version,
                                                                  true,  // is_output
                                                                  unroll_factor);
    } else {

        dim3 grid(m);
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */
        block.x = (block.x + 31) / 32 * 32;

        size_t maxbytes = n * sizeof(T);
        if (residual1 == nullptr) {
            if (maxbytes >= (48 << 10)) {
                check_cuda_error(cudaFuncSetAttribute(
                    generalAddBiasResidualLayerNorm<T, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            }
            generalAddBiasResidualLayerNorm<T, false><<<grid, block, maxbytes, stream>>>(input,
                                                                                         residual1,
                                                                                         gamma,
                                                                                         beta,
                                                                                         bias,
                                                                                         output,
                                                                                         norm_output,
                                                                                         layernorm_eps,
                                                                                         m,
                                                                                         n,
                                                                                         scale_inter,
                                                                                         scale_out,
                                                                                         scale,
                                                                                         dynamic_scale,
                                                                                         int8_mode);
        } else {
            if (maxbytes >= (48 << 10)) {
                check_cuda_error(cudaFuncSetAttribute(
                    generalAddBiasResidualLayerNorm<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            }
            generalAddBiasResidualLayerNorm<T, true><<<grid, block, maxbytes, stream>>>(input,
                                                                                        residual1,
                                                                                        gamma,
                                                                                        beta,
                                                                                        bias,
                                                                                        output,
                                                                                        norm_output,
                                                                                        layernorm_eps,
                                                                                        m,
                                                                                        n,
                                                                                        scale_inter,
                                                                                        scale_out,
                                                                                        scale,
                                                                                        dynamic_scale,
                                                                                        int8_mode);
        }
    }
}

#define INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(T)                                                 \
    template void invokeGeneralAddBiasResidualPreLayerNorm(T*           output,                                        \
                                                           T*           norm_output,                                   \
                                                           const T*     input,                                         \
                                                           const T*     residual1,                                     \
                                                           const T*     gamma,                                         \
                                                           const T*     beta,                                          \
                                                           const T*     bias,                                          \
                                                           const float  layernorm_eps,                                 \
                                                           int          m,                                             \
                                                           int          n,                                             \
                                                           const float* scale_inter,                                   \
                                                           const float* scale_out,                                     \
                                                           float*       scale,                                         \
                                                           float*       dynamic_scale,                                 \
                                                           const int    int8_mode,                                     \
                                                           cudaStream_t stream,                                        \
                                                           int          opt_version)
INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(float);
INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_GENERAL_ADD_BIAS_RESIDUAL_PRE_LAYER_NORM

template<typename T>
__global__ void generalAddResidualT5LayerNorm(const T* __restrict input,
                                              const T* __restrict gamma,
                                              T*          output,
                                              T*          norm_output,
                                              const float layernorm_eps,
                                              int         m,
                                              int         n) {
    // layernorm module in the T5 style No bias and no subtraction of mean.
    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((float)ldg(&input[blockIdx.x * n + i]) + (float)output[blockIdx.x * n + i]);

        float diff = (float)(output[blockIdx.x * n + i]);
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        norm_output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)output[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
    }
}

template<typename T>
__global__ void generalAddBiasResidualT5LayerNorm(const T* __restrict input,
                                                  const T* __restrict gamma,
                                                  const T* __restrict bias,
                                                  T*          output,
                                                  T*          norm_output,
                                                  const float layernorm_eps,
                                                  int         m,
                                                  int         n) {
    // layernorm module in the T5 style No bias and no subtraction of mean.
    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] = clamp_inf_for_half<T>((float)ldg(&input[blockIdx.x * n + i])
                                                           + (float)output[blockIdx.x * n + i] + (float)bias[i]);

        float diff = (float)(output[blockIdx.x * n + i]);
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        norm_output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)output[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
    }
}

template<typename T>
void invokeGeneralAddResidualT5PreLayerNorm(T*           output,
                                            T*           norm_output,
                                            const T*     input,
                                            const T*     gamma,
                                            const float  layernorm_eps,
                                            int          m,
                                            int          n,
                                            cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
    Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */

    block.x = 32 * ((block.x + 31) / 32);

    // TODO(bhsueh) add 16bitx2 implementation
    /* should pay attention to the rsqrt precision*/
    generalAddResidualT5LayerNorm<T>
        <<<grid, block, 0, stream>>>(input, gamma, output, norm_output, layernorm_eps, m, n);
}

#define INVOKE_GENERAL_ADD_RES_T5_LN(T)                                                                                \
    template void invokeGeneralAddResidualT5PreLayerNorm(T*           output,                                          \
                                                         T*           norm_output,                                     \
                                                         const T*     input,                                           \
                                                         const T*     gamma,                                           \
                                                         const float  layernorm_eps,                                   \
                                                         int          m,                                               \
                                                         int          n,                                               \
                                                         cudaStream_t stream);
INVOKE_GENERAL_ADD_RES_T5_LN(float)
INVOKE_GENERAL_ADD_RES_T5_LN(half)
#ifdef ENABLE_BF16
INVOKE_GENERAL_ADD_RES_T5_LN(__nv_bfloat16)
#endif

template<typename T>
void invokeGeneralAddBiasResidualT5PreLayerNorm(T*           output,
                                                T*           norm_output,
                                                const T*     input,
                                                const T*     gamma,
                                                const T*     beta,
                                                const T*     bias,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                cudaStream_t stream) {
    if (beta != nullptr) {
        FT_CHECK_WITH_INFO(false, "rmsnorm beta should always be nullptr");
    } else if (bias != nullptr) {
        dim3 grid(m);
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */

        block.x = 32 * ((block.x + 31) / 32);

        // TODO(bhsueh) add 16bitx2 implementation
        /* should pay attention to the rsqrt precision*/
        generalAddBiasResidualT5LayerNorm<T>
            <<<grid, block, 0, stream>>>(input, gamma, bias, output, norm_output, layernorm_eps, m, n);
    } else {
        FT_CHECK_WITH_INFO(bias == nullptr, "bias should be nullptr when beta is nullptr");
        invokeGeneralAddResidualT5PreLayerNorm(output, norm_output, input, gamma, layernorm_eps, m, n, stream);
    }
    return;
}

#define INVOKE_GENERAL_ADD_BIAS_RES_T5_PRE_LN(T)                                                                       \
    template void invokeGeneralAddBiasResidualT5PreLayerNorm(T*           output,                                      \
                                                             T*           norm_output,                                 \
                                                             const T*     input,                                       \
                                                             const T*     gamma,                                       \
                                                             const T*     beta,                                        \
                                                             const T*     bias,                                        \
                                                             const float  layernorm_eps,                               \
                                                             int          m,                                           \
                                                             int          n,                                           \
                                                             cudaStream_t stream);
INVOKE_GENERAL_ADD_BIAS_RES_T5_PRE_LN(float)
INVOKE_GENERAL_ADD_BIAS_RES_T5_PRE_LN(half)
#ifdef ENABLE_BF16
INVOKE_GENERAL_ADD_BIAS_RES_T5_PRE_LN(__nv_bfloat16)
#endif

template<typename T, bool DYNAMIC_SCALING = false>
__global__ void generalLayerNorm(const T* __restrict input,
                                 const T* __restrict gamma,
                                 const T* __restrict beta,
                                 T*          normed_output,
                                 const float layernorm_eps,
                                 int         m,
                                 int         n,
                                 float*      scale,
                                 float*      dynamic_scale,
                                 const int   int8_mode) {
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const Float_Packed_T scale_to_int = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    Scalar_T abs_max = 1e-6f;

    for (int i = tid; i < n; i += blockDim.x) {
        const int index    = blockIdx.x * n + i;
        float     beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        T         val      = (T)((((float)input[index] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

        if (DYNAMIC_SCALING) {
            abs_max  = cuda_max(cuda_max<Scalar_T, T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        } else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_to_int);
        } else {
            normed_output[index] = val;
        }
    }

    if (DYNAMIC_SCALING) {
        float          abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const Scalar_T dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = tid; i < n; i += blockDim.x) {
            const int index                                        = blockIdx.x * n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

template<typename T>
void invokeGeneralLayerNorm(T*           out,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  layernorm_eps,
                            const int    m,
                            const int    n,
                            float*       scale,
                            float*       dynamic_scale,
                            const int    int8_mode,
                            cudaStream_t stream,
                            int          opt_version) {
    dim3       grid(m);
    const bool dynamic_quant = dynamic_scale != nullptr;
#ifdef ENABLE_BF16
    if (n % 2 == 0 && (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
#else
    if (n % 2 == 0 && (std::is_same<T, half>::value)
#endif
        && opt_version > 0) {
        int  half_n    = n / 2;
        int  half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int  rolls_per_thread = half_n / block.x;
        int  unroll_factor    = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;

        /* we launch (and instantiate) the kernel by specializing for unroll_factor -> residual_num -> is_bias ->
         * opt_version */
        dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor((T2*)out,
                                                                  (T2*)out,
                                                                  (const T2*)out,
                                                                  (const T2*)nullptr,
                                                                  (const T2*)input,
                                                                  (const T2*)gamma,
                                                                  (const T2*)beta,
                                                                  layernorm_eps,
                                                                  m,
                                                                  half_n,
                                                                  nullptr,
                                                                  nullptr,
                                                                  scale,
                                                                  dynamic_scale,
                                                                  int8_mode,
                                                                  grid,
                                                                  block,
                                                                  stream,
                                                                  opt_version,
                                                                  false,  // is_output
                                                                  unroll_factor);
    } else {
        dim3 block(min(n, 1024));

        block.x = 32 * ((block.x + 31) / 32);

        /* should pay attention to the rsqrt precision*/
        if (dynamic_quant) {
            size_t maxbytes = n * sizeof(T);
            if (maxbytes >= (48 << 10)) {
                check_cuda_error(cudaFuncSetAttribute(
                    generalLayerNorm<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            }
            generalLayerNorm<T, true><<<grid, block, maxbytes, stream>>>(
                input, gamma, beta, out, layernorm_eps, m, n, scale, dynamic_scale, int8_mode);  // For gpt-3
        } else {
            generalLayerNorm<T, false><<<grid, block, 0, stream>>>(
                input, gamma, beta, out, layernorm_eps, m, n, scale, dynamic_scale, int8_mode);  // For gpt-3
        }
    }
}

#define INVOKE_GENERAL_LN(T)                                                                                           \
    template void invokeGeneralLayerNorm(T*           out,                                                             \
                                         const T*     input,                                                           \
                                         const T*     gamma,                                                           \
                                         const T*     beta,                                                            \
                                         const float  layernorm_eps,                                                   \
                                         const int    m,                                                               \
                                         const int    n,                                                               \
                                         float*       scale,                                                           \
                                         float*       dynamic_scale,                                                   \
                                         const int    int8_mode,                                                       \
                                         cudaStream_t stream,                                                          \
                                         int          opt_version);

INVOKE_GENERAL_LN(float)
INVOKE_GENERAL_LN(half)
#ifdef ENABLE_BF16
INVOKE_GENERAL_LN(__nv_bfloat16)
#endif

template<typename T, bool DYNAMIC_SCALING = false>
__global__ void generalLayerNormWithPadding(const T* __restrict input,
                                            const T* __restrict gamma,
                                            const T* __restrict beta,
                                            T*          normed_output,
                                            const float layernorm_eps,
                                            int         m,
                                            int         real_n,
                                            int         padding_n,
                                            float*      scale,
                                            float*      dynamic_scale,
                                            const int   int8_mode) {
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    using Scalar_T       = typename packed_as<T, 1>::type;

    const Float_Packed_T scale_to_int = cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

    float local_sum = 0.0f;
    for (int i = tid; i < real_n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * padding_n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / real_n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < real_n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * padding_n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / real_n + layernorm_eps);
    }
    __syncthreads();

    Scalar_T abs_max = 1e-6f;

    for (int i = tid; i < real_n; i += blockDim.x) {
        const int index    = blockIdx.x * padding_n + i;
        float     beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        T         val      = (T)((((float)input[index] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

        if (DYNAMIC_SCALING) {
            abs_max  = cuda_max(cuda_max<Scalar_T, T>(cuda_abs(val)), abs_max);
            shmem[i] = val;
        } else if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_to_int);
        } else {
            normed_output[index] = val;
        }
    }

    if (DYNAMIC_SCALING) {
        float          abs_max_f               = blockAllReduceMax(cuda_cast<float>(abs_max));
        const Scalar_T dynamic_per_token_scale = 127. / abs_max_f;
        for (int i = tid; i < real_n; i += blockDim.x) {
            const int index                                        = blockIdx.x * padding_n + i;
            reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = cuda_cast<Int8_Packed_T>(
                cuda_cast<Float_Packed_T>(shmem[i]) * cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
        }
        if (threadIdx.x == 0) {
            dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
        }
    }
}

template<typename T>
void invokeGeneralLayerNormWithPadding(T*           out,
                                       const T*     input,
                                       const T*     gamma,
                                       const T*     beta,
                                       const float  layernorm_eps,
                                       const int    m,
                                       const int    real_n,
                                       const int    padding_n,
                                       float*       scale,
                                       float*       dynamic_scale,
                                       const int    int8_mode,
                                       cudaStream_t stream,
                                       int          opt_version) {
    dim3       grid(m);
    const bool dynamic_quant = dynamic_scale != nullptr;

    // FT_CHECK_WITH_INFO(real_n % 2 == 1, "just support odd number of real_n");

    dim3 block(min(real_n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    block.x = 32 * ((block.x + 31) / 32);

    /* should pay attention to the rsqrt precision*/
    if (dynamic_quant) {
        size_t maxbytes = real_n * sizeof(T);
        if (maxbytes >= (48 << 10)) {
            check_cuda_error(cudaFuncSetAttribute(
                generalLayerNormWithPadding<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
        }
        generalLayerNormWithPadding<T, true><<<grid, block, maxbytes, stream>>>(
            input, gamma, beta, out, layernorm_eps, m, real_n, padding_n, scale, dynamic_scale, int8_mode);  // For
                                                                                                             // gpt-3
    } else {
        generalLayerNormWithPadding<T, false><<<grid, block, 0, stream>>>(
            input, gamma, beta, out, layernorm_eps, m, real_n, padding_n, scale, dynamic_scale, int8_mode);  // For
                                                                                                             // gpt-3
    }
}

#define INVOKE_GENERAL_LN_WITH_PADDING(T)                                                                              \
    template void invokeGeneralLayerNormWithPadding(T*           out,                                                  \
                                                    const T*     input,                                                \
                                                    const T*     gamma,                                                \
                                                    const T*     beta,                                                 \
                                                    const float  layernorm_eps,                                        \
                                                    const int    m,                                                    \
                                                    const int    real_n,                                               \
                                                    const int    padding_n,                                            \
                                                    float*       scale,                                                \
                                                    float*       dynamic_scale,                                        \
                                                    const int    int8_mode,                                            \
                                                    cudaStream_t stream,                                               \
                                                    int          opt_version);
INVOKE_GENERAL_LN_WITH_PADDING(float)
INVOKE_GENERAL_LN_WITH_PADDING(half)
#ifdef ENABLE_BF16
INVOKE_GENERAL_LN_WITH_PADDING(__nv_bfloat16)
#endif

template<typename T>
__global__ void generalT5LayerNorm(
    const T* __restrict input, const T* __restrict gamma, T* output, const float layernorm_eps, int m, int n) {
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((((float)input[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
    }
}

template<typename T, int N>
__global__ void alphaAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* residual1, const T* bias, const T* gamma, const T* beta, T alpha, int m, int n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out =
            (float)(input[blockIdx.x * n + idx] + residual1[blockIdx.x * n + idx] * alpha + __ldg_func(&bias[idx]));
        mean += local_out;
        // save local_out to local_out_cache to save some recompute
        local_out_cache[i] = local_out;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out           = local_out_cache[i];
        out[blockIdx.x * n + idx] = (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg_func(&gamma[idx]))
                                        + (float)(__ldg_func(&beta[idx])));
        idx += blockDim.x;
    }
}

template<typename T>
__global__ void generalAlphaAddBiasResidualPostLayerNorm(
    T* out, const T* input, const T* residual1, const T* bias, const T* gamma, const T* beta, T alpha, int m, int n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out =
            (float)(input[blockIdx.x * n + idx] + residual1[blockIdx.x * n + idx] * alpha + __ldg_func(&bias[idx]));
        mean += local_out;
        // save local_out to out to save some recompute
        out[blockIdx.x * n + idx] = local_out;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + 1e-6f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out           = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] = (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg_func(&gamma[idx]))
                                        + (float)(__ldg_func(&beta[idx])));
    }
}

template<>
__global__ void generalAlphaAddBiasResidualPostLayerNorm(half*       out,
                                                         const half* input,
                                                         const half* residual1,
                                                         const half* bias,
                                                         const half* gamma,
                                                         const half* beta,
                                                         half        alpha,
                                                         int         m,
                                                         int         n) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    half2        alpha2       = make_half2(alpha, alpha);
    half2*       out_ptr      = (half2*)out;
    const half2* input_ptr    = (const half2*)input;
    const half2* residual_ptr = (const half2*)residual1;
    const half2* bias_ptr     = (const half2*)bias;
    const half2* gamma_ptr    = (const half2*)gamma;
    const half2* beta_ptr     = (const half2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id  = blockIdx.x * n / 2 + idx;
        half2  tmp = __hadd2(__hadd2(input_ptr[id], __hmul2(residual_ptr[id], alpha2)), __ldg_func(&bias_ptr[idx]));
        float2 local_out_fp2 = __half22float2(tmp);
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save tmp to out_ptr to save some recomputation
        out_ptr[id] = tmp;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        float2 gamma_val     = __half22float2(__ldg_func(&gamma_ptr[idx]));
        float2 beta_val      = __half22float2(__ldg_func(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = __float22half2_rn(local_out_fp2);
    }
}

template<typename T>
__global__ void alphaAddBiasResidualPostLayerNormV2(T* out,
                                                    const T* __restrict input,
                                                    const T* __restrict residual1,
                                                    const T* __restrict bias,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    T   alpha,
                                                    int n) {
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id   = i * blockDim.x + tid;
        int id       = bid * n + col_id;
        local_out[i] = (float)(input[id] + __ldg_func(&residual1[id]) * alpha + __ldg_func(&bias[col_id]));
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id     = bid * n + col_id;
        out[id]    = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg_func(&gamma[col_id])
                      + (float)__ldg_func(&beta[col_id]));
    }
}

template<>
__global__ void alphaAddBiasResidualPostLayerNormV2(half* out,
                                                    const half* __restrict input,
                                                    const half* __restrict residual1,
                                                    const half* __restrict bias,
                                                    const half* __restrict gamma,
                                                    const half* __restrict beta,
                                                    half alpha,
                                                    int  n) {
    const int        ite = 4;
    const int        tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    half2            local_out_half2[ite];

    half2        alpha2       = make_half2(alpha, alpha);
    half2*       out_ptr      = (half2*)out;
    const half2* input_ptr    = (const half2*)input;
    const half2* residual_ptr = (const half2*)residual1;
    const half2* bias_ptr     = (const half2*)bias;
    const half2* gamma_ptr    = (const half2*)gamma;
    const half2* beta_ptr     = (const half2*)beta;

    // float sum = 0.0f;
    half2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id         = i * blockDim.x + tid;
        int id             = bid * n / 2 + col_id;
        local_out_half2[i] = input_ptr[id] + __ldg_func(&residual_ptr[id]) * alpha2 + __ldg_func(&bias_ptr[col_id]);
        sum += local_out_half2[i];
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var      = 0.0f;
    half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = local_out_half2[i] - s_mean_2;
        float v1           = (float)local_out_half2[i].x;
        float v2           = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id  = i * blockDim.x + tid;
        int id      = bid * n / 2 + col_id;
        out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg_func(&gamma_ptr[col_id]) + __ldg_func(&beta_ptr[col_id]);
    }
}

template<typename T>
void invokeAlphaAddBiasResidualLayerNorm(T*           out,
                                         const T*     input,
                                         const T*     residual1,
                                         const T*     bias,
                                         const T*     gamma,
                                         const T*     beta,
                                         T            alpha,
                                         int          m,
                                         int          n,
                                         cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        alphaAddBiasResidualPostLayerNormV2<T>
            <<<grid, n / 4, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            alphaAddBiasResidualPostLayerNorm<T, 1>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else if (num_trips == 2) {
            alphaAddBiasResidualPostLayerNorm<T, 2>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else {
            generalAlphaAddBiasResidualPostLayerNorm<T>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        }
    }
}

template<>
void invokeAlphaAddBiasResidualLayerNorm(half*        out,
                                         const half*  input,
                                         const half*  residual1,
                                         const half*  bias,
                                         const half*  gamma,
                                         const half*  beta,
                                         half         alpha,
                                         int          m,
                                         int          n,
                                         cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (m >= 512 && (n == 768 || n == 1024)) {
        alphaAddBiasResidualPostLayerNormV2<half>
            <<<grid, n / 8, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, n);
    } else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            alphaAddBiasResidualPostLayerNorm<half, 1>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else if (num_trips == 2) {
            alphaAddBiasResidualPostLayerNorm<half, 2>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        } else {
            generalAlphaAddBiasResidualPostLayerNorm<half>
                <<<grid, block, 0, stream>>>(out, input, residual1, bias, gamma, beta, alpha, m, n);
        }
    }
}

#define INVOKE_ALPHA_ADD_BIAS_RES_LN(T)                                                                                \
    template void invokeAlphaAddBiasResidualLayerNorm(T*           out,                                                \
                                                      const T*     input,                                              \
                                                      const T*     residual1,                                          \
                                                      const T*     bias,                                               \
                                                      const T*     gamma,                                              \
                                                      const T*     beta,                                               \
                                                      T            alpha,                                              \
                                                      int          m,                                                  \
                                                      int          n,                                                  \
                                                      cudaStream_t stream);
INVOKE_ALPHA_ADD_BIAS_RES_LN(float)
INVOKE_ALPHA_ADD_BIAS_RES_LN(half)
#ifdef ENABLE_BF16
INVOKE_ALPHA_ADD_BIAS_RES_LN(__nv_bfloat16)
#endif

}  // namespace fastertransformer
