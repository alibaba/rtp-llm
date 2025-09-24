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

#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/kernels/rocm/layernorm_kernels.h"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"

#if ENABLE_TRITON
#include "rtp_llm/cpp/kernels/triton/layernorm_kernels.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

// wont't support new features
namespace rtp_llm {

__device__ __forceinline__ int64_t loadOffset(int head_num, int size_per_head) {
    // [[q_head_1],[q_head_2]...[k_head_1],[k_head_2]...[v_head_1],[v_head_2]...]
    int head_id  = blockIdx.y;
    int batch_id = blockIdx.x;
    int offset   = batch_id * head_num * size_per_head + size_per_head * head_id;
    return offset;
}

__device__ __forceinline__ int64_t loadOffsetStrided(const int stride, const int n_elems) {
    return blockIdx.x * stride / n_elems;
}

template<typename T>
__global__ void
qkLayerNorm(T* __restrict qkv, const T* __restrict gamma, const float layernorm_eps, int head_num, int size_per_head) {
    constexpr auto   num_elems_T       = num_elems<T>::value;
    constexpr size_t warp_size         = 32;
    const int        vec_size_per_head = size_per_head / num_elems_T;
    const int        n_elems           = vec_size_per_head / warp_size;
    using float_packed_t               = typename packed_as<float, num_elems_T>::type;

    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = 0; i < n_elems; i++) {
        auto index = loadOffset(head_num, vec_size_per_head) + tid * n_elems + i;
        auto val_f = cuda_cast<float_packed_t>(ldg(&qkv[index]));
        local_sum += cuda_sum<float>(val_f);
    }

    mean = warpReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / size_per_head;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = 0; i < n_elems; i++) {
        auto index = loadOffset(head_num, vec_size_per_head) + tid * n_elems + i;
        auto val_f = cuda_cast<float_packed_t>(ldg(&qkv[index]));
        auto diff  = val_f - s_mean;
        local_var_sum += cuda_sum<float>(diff * diff);
    }
    variance = warpReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / size_per_head + layernorm_eps);
    }
    __syncthreads();

    for (int i = 0; i < n_elems; i++) {
        auto index       = loadOffset(head_num, vec_size_per_head) + tid * n_elems + i;
        auto gamma_index = blockIdx.y * vec_size_per_head + tid * n_elems + i;
        auto val_f       = cuda_cast<float_packed_t>(ldg(&qkv[index]));
        auto val_gamma   = cuda_cast<float_packed_t>(gamma[gamma_index]);
        qkv[index]       = cuda_cast<T>((val_f - s_mean) * s_variance * val_gamma);
    }
}

template<typename T, bool IS_BIAS>
__global__ void layerNormWithStride(T* __restrict output,
                                    const int out_stride,
                                    const T* __restrict input,
                                    const int in_stride,
                                    const T* __restrict gamma,
                                    const T* __restrict beta,
                                    const float layernorm_eps,
                                    const int   n,  // 总特征维度
                                    const int   norm_size) {
    constexpr auto   num_elems_T = num_elems<T>::value;  // 向量化元素数
    constexpr size_t warp_size   = 32;
    const int        n_elems     = norm_size / num_elems_T / warp_size;
    using float_packed_t         = typename packed_as<float, num_elems_T>::type;

    const int tid        = threadIdx.x;
    const int sample_idx = blockIdx.x / (n / norm_size);  // 样本索引
    const int head_idx   = blockIdx.x % (n / norm_size);  // 头/窗口索引

    __shared__ float s_mean;
    __shared__ float s_variance;

    // 计算当前窗口的起始位置
    const T* sample_start   = input + sample_idx * (in_stride / num_elems_T);
    T*       output_start   = output + sample_idx * (out_stride / num_elems_T);
    const T* head_start     = sample_start + head_idx * (norm_size / num_elems_T);
    T*       out_head_start = output_start + head_idx * (norm_size / num_elems_T);

    // Stage 1: 计算均值
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < n_elems; i++) {
        int  elem_idx = i * warp_size + tid;
        auto val_f    = cuda_cast<float_packed_t>(ldg(&head_start[elem_idx]));
        local_sum += cuda_sum<float>(val_f);
    }

    float mean = warpReduceSum(local_sum);
    if (tid == 0) {
        s_mean = mean / norm_size;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < n_elems; i++) {
        int  elem_idx = i * warp_size + tid;
        auto val_f    = cuda_cast<float_packed_t>(ldg(&head_start[elem_idx]));
        auto diff     = val_f - s_mean;
        local_var_sum += cuda_sum<float>(diff * diff);
    }

    float variance = warpReduceSum(local_var_sum);
    if (tid == 0) {
        s_variance = rsqrtf(variance / norm_size + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < n_elems; i++) {
        int  elem_idx = i * warp_size + tid;
        auto val_f    = cuda_cast<float_packed_t>(ldg(&head_start[elem_idx]));

        auto gamma_val = cuda_cast<float_packed_t>(gamma[elem_idx]);
        if (IS_BIAS) {
            auto beta_val = cuda_cast<float_packed_t>(beta[elem_idx]);

            val_f = (val_f - s_mean) * s_variance * gamma_val + beta_val;
        } else {
            val_f = (val_f - s_mean) * s_variance * gamma_val;
        }
        out_head_start[elem_idx] = cuda_cast<T>(val_f);
    }
}

template<typename T>
void invokeQkLayerNorm(T* __restrict qkv,
                       const T* __restrict gamma,
                       const float  layernorm_eps,
                       const int    tokens,
                       const int    head_num,
                       const int    head_num_kv,
                       const int    size_per_head,
                       cudaStream_t stream) {
    constexpr size_t vec_size  = 2;
    constexpr size_t warp_size = 32;

    if (size_per_head % warp_size != 0) {
        throw std::invalid_argument("not supported size_per_head: " + std::to_string(size_per_head));
    }
    dim3 grid(tokens, head_num + head_num_kv);
    dim3 block(warp_size);

    int total_head_num = head_num + 2 * head_num_kv;
    using Tp           = typename packed_as<T, vec_size>::type;
    qkLayerNorm<Tp><<<grid, block, 0, stream>>>(
        reinterpret_cast<Tp*>(qkv), reinterpret_cast<const Tp*>(gamma), layernorm_eps, total_head_num, size_per_head);
}

template<typename T>
void invokeLayerNormWithStride(T* __restrict output,
                               const int out_stride,
                               const T* __restrict input,
                               const int in_stride,
                               const T* __restrict gamma,
                               const T* __restrict beta,
                               const float  layernorm_eps,
                               const int    m,
                               const int    n,
                               const int    norm_size,
                               cudaStream_t stream) {
    constexpr size_t vec_size  = 2;
    constexpr size_t warp_size = 32;

    // 参数校验
    if (n % norm_size != 0) {
        throw std::invalid_argument("n:" + std::to_string(n)
                                    + " must be divisible by norm_size:" + std::to_string(norm_size));
    }
    if (norm_size % (warp_size * vec_size) != 0) {
        throw std::invalid_argument("norm_size must be multiple of " + std::to_string(warp_size * vec_size));
    }

    const int num_heads = n / norm_size;
    dim3      grid(m * num_heads);  // 每个block处理一个样本的一个头
    dim3      block(warp_size);

    using Tp     = typename packed_as<T, vec_size>::type;
    bool is_bias = beta != nullptr;
    if (is_bias) {
        layerNormWithStride<Tp, true><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(output),
                                                                  out_stride,
                                                                  reinterpret_cast<const Tp*>(input),
                                                                  in_stride,
                                                                  reinterpret_cast<const Tp*>(gamma),
                                                                  reinterpret_cast<const Tp*>(beta),
                                                                  layernorm_eps,
                                                                  n,
                                                                  norm_size);
    } else {
        layerNormWithStride<Tp, false><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(output),
                                                                   out_stride,
                                                                   reinterpret_cast<const Tp*>(input),
                                                                   in_stride,
                                                                   reinterpret_cast<const Tp*>(gamma),
                                                                   nullptr,
                                                                   layernorm_eps,
                                                                   n,
                                                                   norm_size);
    }
}

#define INSTANTIATE_QK_LAYERNORM(T)                                                                                    \
    template void invokeQkLayerNorm(T* __restrict qkv,                                                                 \
                                    const T* __restrict gamma,                                                         \
                                    const float  layernorm_eps,                                                        \
                                    const int    tokens,                                                               \
                                    const int    head_num,                                                             \
                                    const int    head_num_kv,                                                          \
                                    const int    size_per_head,                                                        \
                                    cudaStream_t stream)
INSTANTIATE_QK_LAYERNORM(float);
INSTANTIATE_QK_LAYERNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_QK_LAYERNORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_QK_LAYERNORM

#define INSTANTIATE_STRIDED_LAYERNORM(T)                                                                               \
    template void invokeLayerNormWithStride(T* __restrict output,                                                      \
                                            const int out_stride,                                                      \
                                            const T* __restrict input,                                                 \
                                            const int in_stride,                                                       \
                                            const T* __restrict gamma,                                                 \
                                            const T* __restrict beta,                                                  \
                                            const float  layernorm_eps,                                                \
                                            const int    m,                                                            \
                                            const int    n,                                                            \
                                            const int    norm_size,                                                    \
                                            cudaStream_t stream);
INSTANTIATE_STRIDED_LAYERNORM(float);
INSTANTIATE_STRIDED_LAYERNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_STRIDED_LAYERNORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_STRIDED_LAYERNORM

template<typename Tf, typename T, bool IS_BETA>
__inline__ __device__ Tf
compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i) {
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (IS_BETA) {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

/* Computes the layernorm https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 * normed_output <- ( (input - E[input]) / Sqrt(Var[input] + eps) ) * gamma + beta
 * input is [tokens, hidden_dim]. Mean and Variance are per-row (i.e. per-token)
 *
 * One CTA handles one row.
 *
 * with USE_DIFF_OF_SQUARES set to false:
 * First pass (loop) computes the mean.
 * Second computes the variance via Var[x] = E[(x - E[x])²].
 * Third pass computes and writes normed_output
 *
 * with USE_DIFF_OF_SQUARES set to true (may be faster but less accurate):
 * First pass (loop) computes the mean and variance via Var[x] = E[x²] - E[x]²
 * Second pass computes and writes normed_output
 *
 * use_shmem controls if we cache input values into shared memory
 *
 * Optional: with dynamic scaling, the last pass doesn't write immediately but finds the
 *           amax per row. A final pass scales to int8 accordingly, and writes output to
 *           normed_output_quant.
 */
template<typename T,
         typename QUANT_OUT_T,
         bool IS_OUTPUT,
         bool IS_BIAS,
         bool RESIDUAL,
         bool IS_BETA,
         bool RETURN_NORMED_OUTPUT,
         bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(T*           output,
                                 T*           normed_output,
                                 const T*     input,
                                 const T*     bias,
                                 const T*     residual,
                                 const T*     gamma,
                                 const T*     beta,
                                 const float  eps,
                                 int          tokens,
                                 int          hidden_dim,
                                 const float* scale_orig_quant_per_tensor,
                                 float*       scale_orig_quant_per_token,
                                 QUANT_OUT_T* normed_output_quant) {
    constexpr auto num_elems_T = num_elems<T>::value;
    using quant_packed_t       = typename packed_as<QUANT_OUT_T, num_elems_T>::type;
    using Int32_Packed_T       = typename packed_as<int32_t, num_elems_T>::type;
    using float_packed_t       = typename packed_as<float, num_elems_T>::type;
    using T_scalar             = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float                                s_mean;
    __shared__ float                                s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean          = 0.0f;
    float variance      = 0.0f;
    float local_sum     = 0.0f;
    float local_var_sum = 0.0f;

    const bool           with_per_token_scaling  = scale_orig_quant_per_token != nullptr;
    const bool           with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant =
        cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
    T_scalar  amax    = getAmax<QUANT_OUT_T>();
    const int n_elems = hidden_dim / num_elems_T;

    for (int i = tidx; i < n_elems; i += blockDim.x) {
        // const T val = input[bidx * n_elems + i];
        const int index = bidx * n_elems + i;
        T         val   = input[index];
        // const T val = input[index];
        if (IS_BIAS) {
            val = add(val, ldg(&bias[i]));
        }
        if (RESIDUAL) {
            val = add(val, ldg(&residual[index]));
        }
        if (IS_OUTPUT && !RETURN_NORMED_OUTPUT) {
            output[index] = val;
        }
        shmem[i] = val;

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES) {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES) {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean     = packed[0];
        variance = packed[1];
    } else {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0) {
        mean   = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES) {
            variance   = (variance / hidden_dim) - (mean * mean);  // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES) {
        for (int i = tidx; i < n_elems; i += blockDim.x) {
            const T        val  = shmem[i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0) {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    for (int i = tidx; i < n_elems; i += blockDim.x) {
        const int            index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
        const T              val =
            cuda_cast<T>(compute_layernorm<float_packed_t, T, IS_BETA>(val_f, s_mean, s_variance, gamma, beta, i));
        if (RETURN_NORMED_OUTPUT && IS_OUTPUT) {
            output[index] = val;
        }

        if (with_per_token_scaling) {
            amax     = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            shmem[i] = val;
        } else if (with_per_tensor_scaling) {
            reinterpret_cast<quant_packed_t*>(normed_output_quant)[index] =
                cuda_cast<quant_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        } else {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling) {
        float       abs_max_f               = blockAllReduceMax(cuda_cast<float>(amax));
        const float scale_factor            = getScaleFactor<QUANT_OUT_T>();
        const float dynamic_per_token_scale = scale_factor / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x) {
            const int      index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
            reinterpret_cast<quant_packed_t*>(normed_output_quant)[index] =
                cuda_cast<quant_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0) {
            scale_orig_quant_per_token[bidx] = abs_max_f / scale_factor;
        }
    }
}

template<typename T,
         typename QUANT_OUT_T,
         bool IS_OUTPUT,
         bool IS_BIAS,
         bool RESIDUAL,
         bool IS_BETA,
         bool RETURN_NORMED_OUTPUT,
         bool USE_DIFF_OF_SQUARES>
void dispatch_layernorm_type_square_method(T*           output,
                                           T*           normed_output,
                                           const T*     input,
                                           const T*     bias,
                                           const T*     residual,
                                           const T*     gamma,
                                           const T*     beta,
                                           const float  eps,
                                           int          tokens,
                                           int          hidden_dim,
                                           const float* scale_orig_quant_per_tensor,
                                           float*       scale_orig_quant_per_token,
                                           QUANT_OUT_T* normed_output_quant,
                                           const dim3   grid,
                                           const dim3   block,
                                           const size_t shmem_size,
                                           cudaStream_t stream) {
    if (shmem_size >= (48 << 10)) {
#if USING_CUDA
        cudaError_t ret = cudaFuncSetAttribute(generalLayerNorm<T,
                                                                QUANT_OUT_T,
                                                                IS_OUTPUT,
                                                                IS_BIAS,
                                                                RESIDUAL,
                                                                IS_BETA,
                                                                RETURN_NORMED_OUTPUT,
                                                                USE_DIFF_OF_SQUARES>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               shmem_size);
#endif
    }
    generalLayerNorm<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, RETURN_NORMED_OUTPUT, USE_DIFF_OF_SQUARES>
        <<<grid, block, shmem_size, stream>>>(output,
                                              normed_output,
                                              input,
                                              bias,
                                              residual,
                                              gamma,
                                              beta,
                                              eps,
                                              tokens,
                                              hidden_dim,
                                              scale_orig_quant_per_tensor,
                                              scale_orig_quant_per_token,
                                              normed_output_quant);
}

template<typename T,
         typename QUANT_OUT_T,
         bool IS_OUTPUT,
         bool IS_BIAS,
         bool RESIDUAL,
         bool IS_BETA,
         bool RETURN_NORMED_OUTPUT>
void dispatch_layernorm_return_normed(T*           output,
                                      T*           normed_output,
                                      const T*     input,
                                      const T*     bias,
                                      const T*     residual,
                                      const T*     gamma,
                                      const T*     beta,
                                      const float  eps,
                                      int          tokens,
                                      int          hidden_dim,
                                      const float* scale_orig_quant_per_tensor,
                                      float*       scale_orig_quant_per_token,
                                      QUANT_OUT_T* normed_output_quant,
                                      const dim3   grid,
                                      const dim3   block,
                                      const size_t shmem_size,
                                      cudaStream_t stream,
                                      bool         use_diff_of_squares) {
    if (use_diff_of_squares) {
        dispatch_layernorm_type_square_method<T,
                                              QUANT_OUT_T,
                                              IS_OUTPUT,
                                              IS_BIAS,
                                              RESIDUAL,
                                              IS_BETA,
                                              RETURN_NORMED_OUTPUT,
                                              true>(output,
                                                    normed_output,
                                                    input,
                                                    bias,
                                                    residual,
                                                    gamma,
                                                    beta,
                                                    eps,
                                                    tokens,
                                                    hidden_dim,
                                                    scale_orig_quant_per_tensor,
                                                    scale_orig_quant_per_token,
                                                    normed_output_quant,
                                                    grid,
                                                    block,
                                                    shmem_size,
                                                    stream);
    } else {
        dispatch_layernorm_type_square_method<T,
                                              QUANT_OUT_T,
                                              IS_OUTPUT,
                                              IS_BIAS,
                                              RESIDUAL,
                                              IS_BETA,
                                              RETURN_NORMED_OUTPUT,
                                              false>(output,
                                                     normed_output,
                                                     input,
                                                     bias,
                                                     residual,
                                                     gamma,
                                                     beta,
                                                     eps,
                                                     tokens,
                                                     hidden_dim,
                                                     scale_orig_quant_per_tensor,
                                                     scale_orig_quant_per_token,
                                                     normed_output_quant,
                                                     grid,
                                                     block,
                                                     shmem_size,
                                                     stream);
    }
}

template<typename T, typename QUANT_OUT_T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA>
void dispatch_layernorm_type(T*           output,
                             T*           normed_output,
                             const T*     input,
                             const T*     bias,
                             const T*     residual,
                             const T*     gamma,
                             const T*     beta,
                             const float  eps,
                             int          tokens,
                             int          hidden_dim,
                             const float* scale_orig_quant_per_tensor,
                             float*       scale_orig_quant_per_token,
                             QUANT_OUT_T* normed_output_quant,
                             const dim3   grid,
                             const dim3   block,
                             const size_t shmem_size,
                             cudaStream_t stream,
                             bool         use_diff_of_squares,
                             bool         return_normed_output) {
    if (return_normed_output) {
        dispatch_layernorm_return_normed<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, true>(
            output,
            normed_output,
            input,
            bias,
            residual,
            gamma,
            beta,
            eps,
            tokens,
            hidden_dim,
            scale_orig_quant_per_tensor,
            scale_orig_quant_per_token,
            normed_output_quant,
            grid,
            block,
            shmem_size,
            stream,
            use_diff_of_squares);
    } else {
        dispatch_layernorm_return_normed<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, false>(
            output,
            normed_output,
            input,
            bias,
            residual,
            gamma,
            beta,
            eps,
            tokens,
            hidden_dim,
            scale_orig_quant_per_tensor,
            scale_orig_quant_per_token,
            normed_output_quant,
            grid,
            block,
            shmem_size,
            stream,
            use_diff_of_squares);
    }
}

template<typename T, typename QUANT_OUT_T, bool IS_OUTPUT, bool IS_BIAS, bool RESIUDAL>
void dispatch_layernorm_beta(T*           output,
                             T*           normed_output,
                             const T*     input,
                             const T*     bias,
                             const T*     residual,
                             const T*     gamma,
                             const T*     beta,
                             const float  eps,
                             int          tokens,
                             int          hidden_dim,
                             const float* scale_orig_quant_per_tensor,
                             float*       scale_orig_quant_per_token,
                             QUANT_OUT_T* normed_output_quant,
                             const dim3   grid,
                             const dim3   block,
                             const size_t shmem_size,
                             cudaStream_t stream,
                             bool         use_diff_of_squares,
                             bool         return_normed_output) {
    if (beta != nullptr) {
        dispatch_layernorm_type<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, RESIUDAL, true>(output,
                                                                                    normed_output,
                                                                                    input,
                                                                                    bias,
                                                                                    residual,
                                                                                    gamma,
                                                                                    beta,
                                                                                    eps,
                                                                                    tokens,
                                                                                    hidden_dim,
                                                                                    scale_orig_quant_per_tensor,
                                                                                    scale_orig_quant_per_token,
                                                                                    normed_output_quant,
                                                                                    grid,
                                                                                    block,
                                                                                    shmem_size,
                                                                                    stream,
                                                                                    use_diff_of_squares,
                                                                                    return_normed_output);
    } else {
        dispatch_layernorm_type<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, RESIUDAL, false>(output,
                                                                                     normed_output,
                                                                                     input,
                                                                                     bias,
                                                                                     residual,
                                                                                     gamma,
                                                                                     beta,
                                                                                     eps,
                                                                                     tokens,
                                                                                     hidden_dim,
                                                                                     scale_orig_quant_per_tensor,
                                                                                     scale_orig_quant_per_token,
                                                                                     normed_output_quant,
                                                                                     grid,
                                                                                     block,
                                                                                     shmem_size,
                                                                                     stream,
                                                                                     use_diff_of_squares,
                                                                                     return_normed_output);
    }
}

template<typename T, typename QUANT_OUT_T, bool IS_OUTPUT, bool IS_BIAS>
void dispatch_layernorm_residual(T*           output,
                                 T*           normed_output,
                                 const T*     input,
                                 const T*     bias,
                                 const T*     residual,
                                 const T*     gamma,
                                 const T*     beta,
                                 const float  eps,
                                 int          tokens,
                                 int          hidden_dim,
                                 const float* scale_orig_quant_per_tensor,
                                 float*       scale_orig_quant_per_token,
                                 QUANT_OUT_T* normed_output_quant,
                                 const dim3   grid,
                                 const dim3   block,
                                 const size_t shmem_size,
                                 cudaStream_t stream,
                                 bool         use_diff_of_squares,
                                 bool         return_normed_output) {
    if (residual != nullptr) {
        dispatch_layernorm_beta<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, true>(output,
                                                                          normed_output,
                                                                          input,
                                                                          bias,
                                                                          residual,
                                                                          gamma,
                                                                          beta,
                                                                          eps,
                                                                          tokens,
                                                                          hidden_dim,
                                                                          scale_orig_quant_per_tensor,
                                                                          scale_orig_quant_per_token,
                                                                          normed_output_quant,
                                                                          grid,
                                                                          block,
                                                                          shmem_size,
                                                                          stream,
                                                                          use_diff_of_squares,
                                                                          return_normed_output);
    } else {
        dispatch_layernorm_beta<T, QUANT_OUT_T, IS_OUTPUT, IS_BIAS, false>(output,
                                                                           normed_output,
                                                                           input,
                                                                           bias,
                                                                           residual,
                                                                           gamma,
                                                                           beta,
                                                                           eps,
                                                                           tokens,
                                                                           hidden_dim,
                                                                           scale_orig_quant_per_tensor,
                                                                           scale_orig_quant_per_token,
                                                                           normed_output_quant,
                                                                           grid,
                                                                           block,
                                                                           shmem_size,
                                                                           stream,
                                                                           use_diff_of_squares,
                                                                           return_normed_output);
    }
}

template<typename T, typename QUANT_OUT_T, bool IS_OUTPUT>
void dispatch_layernorm_bias(T*           output,
                             T*           normed_output,
                             const T*     input,
                             const T*     bias,
                             const T*     residual,
                             const T*     gamma,
                             const T*     beta,
                             const float  eps,
                             int          tokens,
                             int          hidden_dim,
                             const float* scale_orig_quant_per_tensor,
                             float*       scale_orig_quant_per_token,
                             QUANT_OUT_T* normed_output_quant,
                             const dim3   grid,
                             const dim3   block,
                             const size_t shmem_size,
                             cudaStream_t stream,
                             bool         use_diff_of_squares,
                             bool         return_normed_output) {
    if (bias != nullptr) {
        dispatch_layernorm_residual<T, QUANT_OUT_T, IS_OUTPUT, true>(output,
                                                                     normed_output,
                                                                     input,
                                                                     bias,
                                                                     residual,
                                                                     gamma,
                                                                     beta,
                                                                     eps,
                                                                     tokens,
                                                                     hidden_dim,
                                                                     scale_orig_quant_per_tensor,
                                                                     scale_orig_quant_per_token,
                                                                     normed_output_quant,
                                                                     grid,
                                                                     block,
                                                                     shmem_size,
                                                                     stream,
                                                                     use_diff_of_squares,
                                                                     return_normed_output);
    } else {
        dispatch_layernorm_residual<T, QUANT_OUT_T, IS_OUTPUT, false>(output,
                                                                      normed_output,
                                                                      input,
                                                                      bias,
                                                                      residual,
                                                                      gamma,
                                                                      beta,
                                                                      eps,
                                                                      tokens,
                                                                      hidden_dim,
                                                                      scale_orig_quant_per_tensor,
                                                                      scale_orig_quant_per_token,
                                                                      normed_output_quant,
                                                                      grid,
                                                                      block,
                                                                      shmem_size,
                                                                      stream,
                                                                      use_diff_of_squares,
                                                                      return_normed_output);
    }
}

template<typename T, typename QUANT_OUT_T>
void dispatch_layernorm_output(T*           output,
                               T*           normed_output,
                               const T*     input,
                               const T*     bias,
                               const T*     residual,
                               const T*     gamma,
                               const T*     beta,
                               const float  eps,
                               int          tokens,
                               int          hidden_dim,
                               const float* scale_orig_quant_per_tensor,
                               float*       scale_orig_quant_per_token,
                               QUANT_OUT_T* normed_output_quant,
                               const dim3   grid,
                               const dim3   block,
                               const size_t shmem_size,
                               cudaStream_t stream,
                               bool         use_diff_of_squares,
                               bool         is_output,
                               bool         return_normed_output) {
    if (is_output) {
        dispatch_layernorm_bias<T, QUANT_OUT_T, true>(output,
                                                      normed_output,
                                                      input,
                                                      bias,
                                                      residual,
                                                      gamma,
                                                      beta,
                                                      eps,
                                                      tokens,
                                                      hidden_dim,
                                                      scale_orig_quant_per_tensor,
                                                      scale_orig_quant_per_token,
                                                      normed_output_quant,
                                                      grid,
                                                      block,
                                                      shmem_size,
                                                      stream,
                                                      use_diff_of_squares,
                                                      return_normed_output);
    } else {
        dispatch_layernorm_bias<T, QUANT_OUT_T, false>(output,
                                                       normed_output,
                                                       input,
                                                       bias,
                                                       residual,
                                                       gamma,
                                                       beta,
                                                       eps,
                                                       tokens,
                                                       hidden_dim,
                                                       scale_orig_quant_per_tensor,
                                                       scale_orig_quant_per_token,
                                                       normed_output_quant,
                                                       grid,
                                                       block,
                                                       shmem_size,
                                                       stream,
                                                       use_diff_of_squares,
                                                       return_normed_output);
    }
}

template<typename T, typename QUANT_OUT_T>
void invokeGeneralLayerNorm(T*           out,
                            T*           normed_output,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  eps,
                            const int    tokens,
                            const int    hidden_dim,
                            cudaStream_t stream,
                            bool         use_diff_of_squares,
                            const float* scale,
                            float*       dynamic_scale,
                            QUANT_OUT_T* out_quant,
                            bool         return_normed_output) {
#if ENABLE_TRITON && !defined(ENABLE_FP8)
    if (hidden_dim <= 4096 && dynamic_scale == nullptr && scale == nullptr && beta != nullptr
        && (out == nullptr || return_normed_output == true)) {
        invokeTritonLayerNorm<T, QUANT_OUT_T, false>(out,
                                                     normed_output,
                                                     input,
                                                     (const T*)nullptr,
                                                     (const T*)nullptr,
                                                     gamma,
                                                     beta,
                                                     eps,
                                                     tokens,
                                                     hidden_dim,
                                                     stream,
                                                     use_diff_of_squares,
                                                     scale,
                                                     dynamic_scale,
                                                     out_quant,
                                                     return_normed_output);
        return;
    }
#endif

    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size     = 2;
    const size_t     shmem_size   = hidden_dim * sizeof(T);
    const bool       use_vec_type = (hidden_dim % vec_size == 0)
                              && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
                                  || std::is_same<T, __nv_bfloat16>::value
#endif
                              );

    if (use_vec_type) {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_layernorm_output(reinterpret_cast<Tp*>(out),
                                  reinterpret_cast<Tp*>(normed_output),
                                  reinterpret_cast<const Tp*>(input),
                                  (const Tp*)nullptr,
                                  (const Tp*)nullptr,
                                  reinterpret_cast<const Tp*>(gamma),
                                  reinterpret_cast<const Tp*>(beta),
                                  eps,
                                  tokens,
                                  hidden_dim,
                                  scale,
                                  dynamic_scale,
                                  out_quant,
                                  grid,
                                  block,
                                  shmem_size,
                                  stream,
                                  use_diff_of_squares,
                                  out != nullptr,
                                  return_normed_output);
    } else {
        dispatch_layernorm_output(out,
                                  normed_output,
                                  (const T*)input,
                                  (const T*)nullptr,
                                  (const T*)nullptr,
                                  gamma,
                                  beta,
                                  eps,
                                  tokens,
                                  hidden_dim,
                                  scale,
                                  dynamic_scale,
                                  out_quant,
                                  grid,
                                  block,
                                  shmem_size,
                                  stream,
                                  use_diff_of_squares,
                                  out != nullptr,
                                  return_normed_output);
    }
}

template<typename T, typename QUANT_OUT_T>
void invokeGeneralAddBiasResidualLayerNorm(T*           out,
                                           T*           norm_output,
                                           const T*     input,
                                           const T*     bias,
                                           const T*     residual,
                                           const T*     gamma,
                                           const T*     beta,
                                           const float  eps,
                                           const int    tokens,
                                           const int    hidden_dim,
                                           cudaStream_t stream,
                                           bool         use_diff_of_squares,
                                           const float* scale,
                                           float*       dynamic_scale,
                                           QUANT_OUT_T* out_quant,
                                           bool         return_normed_output) {
#if ENABLE_TRITON && !defined(ENABLE_FP8)
    if (hidden_dim <= 4096 && dynamic_scale == nullptr && scale == nullptr && beta != nullptr
        && (out == nullptr || return_normed_output == true)) {
        invokeTritonLayerNorm<T, QUANT_OUT_T, true>(out,
                                                    norm_output,
                                                    input,
                                                    bias,
                                                    residual,
                                                    gamma,
                                                    beta,
                                                    eps,
                                                    tokens,
                                                    hidden_dim,
                                                    stream,
                                                    use_diff_of_squares,
                                                    scale,
                                                    dynamic_scale,
                                                    out_quant,
                                                    return_normed_output);
        return;
    }
#endif

    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size     = 2;
    const size_t     shmem_size   = hidden_dim * sizeof(T);
    const bool       use_vec_type = (hidden_dim % vec_size == 0)
                              && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
                                  || std::is_same<T, __nv_bfloat16>::value
#endif
                              );

    if (use_vec_type) {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_layernorm_output(reinterpret_cast<Tp*>(out),
                                  reinterpret_cast<Tp*>(norm_output),
                                  reinterpret_cast<const Tp*>(input),
                                  reinterpret_cast<const Tp*>(bias),
                                  reinterpret_cast<const Tp*>(residual),
                                  reinterpret_cast<const Tp*>(gamma),
                                  reinterpret_cast<const Tp*>(beta),
                                  eps,
                                  tokens,
                                  hidden_dim,
                                  scale,
                                  dynamic_scale,
                                  out_quant,
                                  grid,
                                  block,
                                  shmem_size,
                                  stream,
                                  use_diff_of_squares,
                                  true,
                                  return_normed_output);
    } else {
        dispatch_layernorm_output(out,
                                  norm_output,
                                  input,
                                  bias,
                                  residual,
                                  gamma,
                                  beta,
                                  eps,
                                  tokens,
                                  hidden_dim,
                                  scale,
                                  dynamic_scale,
                                  out_quant,
                                  grid,
                                  block,
                                  shmem_size,
                                  stream,
                                  use_diff_of_squares,
                                  true,
                                  return_normed_output);
    }
}

#define INSTANTIATE_GENERAL_LAYERNORM(T, QUANT_OUT_T)                                                                  \
    template void invokeGeneralLayerNorm(T*           out,                                                             \
                                         T*           normed_output,                                                   \
                                         const T*     input,                                                           \
                                         const T*     gamma,                                                           \
                                         const T*     beta,                                                            \
                                         const float  eps,                                                             \
                                         const int    tokens,                                                          \
                                         const int    hidden_dim,                                                      \
                                         cudaStream_t stream,                                                          \
                                         bool         use_diff_of_squares,                                             \
                                         const float* scale,                                                           \
                                         float*       dynamic_scale,                                                   \
                                         QUANT_OUT_T* out_quant,                                                       \
                                         bool         return_normed_output);

INSTANTIATE_GENERAL_LAYERNORM(float, int8_t);
INSTANTIATE_GENERAL_LAYERNORM(half, int8_t);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_LAYERNORM(__nv_bfloat16, int8_t);
#endif
#ifdef ENABLE_FP8
INSTANTIATE_GENERAL_LAYERNORM(float, __nv_fp8_e4m3);
INSTANTIATE_GENERAL_LAYERNORM(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_LAYERNORM(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

#define INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(T, QUANT_OUT_T)                                                \
    template void invokeGeneralAddBiasResidualLayerNorm(T*           out,                                              \
                                                        T*           norm_output,                                      \
                                                        const T*     input,                                            \
                                                        const T*     bias,                                             \
                                                        const T*     residual,                                         \
                                                        const T*     gamma,                                            \
                                                        const T*     beta,                                             \
                                                        const float  eps,                                              \
                                                        const int    tokens,                                           \
                                                        const int    hidden_dim,                                       \
                                                        cudaStream_t stream,                                           \
                                                        bool         use_diff_of_squares,                              \
                                                        const float* scale,                                            \
                                                        float*       dynamic_scale,                                    \
                                                        QUANT_OUT_T* out_quant,                                        \
                                                        bool         return_normed_output);

INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(float, int8_t);
INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(half, int8_t);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(__nv_bfloat16, int8_t);
#endif
#ifdef ENABLE_FP8
INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(float, __nv_fp8_e4m3);
INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_ADD_BIAS_RESIDUAL_LAYERNORM(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif
}  // namespace rtp_llm
