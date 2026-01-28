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
#include "rtp_llm/cpp/cuda/launch_utils.h"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/cpp/kernels/rmsnormKernels.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

template<typename Tf, typename T, bool IS_BETA>
__inline__ __device__ Tf compute_rmsnorm(Tf val, float s_variance, const T* gamma, const T* beta, size_t i) {
    Tf ret = val * s_variance * cuda_cast<Tf>(gamma[i]);
    if (IS_BETA) {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

/* Computes the rmsnorm https://pytorch.org/docs/stable/generated/torch.nn.rmsnorm.html
 * normed_output <- ( input / Sqrt(E[input²] + eps) ) * gamma + beta
 * input is [tokens, hidden_dim]. Mean and Variance are per-row (i.e. per-token)
 *
 * One CTA handles one row.
 *
 *
 * use_shmem controls if we cache input values into shared memory
 *
 * Optional: with dynamic scaling, the last pass doesn't write immediately but finds the
 *           amax per row. A final pass scales to int8 accordingly, and writes output to
 *           normed_output_quant.
 */
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, typename QUANT_OUT_T>
__global__ void generalRmsNorm(T*           output,
                               T*           normed_output,
                               const T*     input,
                               const T*     bias,
                               const T*     residual1,
                               const T*     residual2,
                               const T*     gamma,
                               const T*     beta,
                               const float  eps,
                               size_t       tokens,
                               size_t       hidden_dim,
                               const float* scale_orig_quant_per_tensor,
                               float*       scale_orig_quant_per_token,
                               QUANT_OUT_T* normed_output_quant) {
    constexpr auto num_elems_T = num_elems<T>::value;
    using quant_packed_t       = typename packed_as<QUANT_OUT_T, num_elems_T>::type;
    using float_packed_t       = typename packed_as<float, num_elems_T>::type;
    using T_scalar             = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_variance;

    const size_t tidx = threadIdx.x;
    const size_t bidx = blockIdx.x;

    float variance      = 0.0f;
    float local_var_sum = 0.0f;

    const size_t n_elems = hidden_dim / num_elems_T;

    const bool           with_per_token_scaling  = scale_orig_quant_per_token != nullptr;
    const bool           with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const bool           has_residual2           = residual2 != nullptr;
    const float_packed_t scale_orig_quant =
        cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
    T_scalar amax = getAmax<QUANT_OUT_T>();

    for (size_t i = tidx; i < n_elems; i += blockDim.x) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        asm volatile("griddepcontrol.wait;");
#endif
        const size_t index = bidx * n_elems + i;
        T            val   = cuda_cast<T>(0.0f);
        // const T val = input[index];
        if (IS_BIAS) {
            val = add(val, ldg(&bias[i]));
        }
        if (RESIDUAL) {
            val = add(val, ldg(&residual1[index]));
            if (has_residual2) {
                val = add(val, ldg(&residual2[index]));
            }
        }
        if (IS_OUTPUT) {
            T in_val = input[index];
            val      = add(val, in_val);
        }

        shmem[i] = val;

        if (IS_OUTPUT) {
            output[index] = val;
        }
        const float_packed_t val_f = cuda_cast<float_packed_t>(val);

        local_var_sum += cuda_sum<float>(val_f * val_f);
    }

    float packed[1] = {local_var_sum};
    blockReduceSumV2<float, 1>(packed);
    variance = packed[0];

    if (threadIdx.x == 0) {
        variance   = (variance / hidden_dim);  // Var[x] = E[x²]
        s_variance = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float scale_factor = getScaleFactor<QUANT_OUT_T>();
    for (size_t i = tidx; i < n_elems; i += blockDim.x) {
        const size_t         index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
        const T val = cuda_cast<T>(compute_rmsnorm<float_packed_t, T, IS_BETA>(val_f, s_variance, gamma, beta, i));

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
        const float dynamic_per_token_scale = scale_factor / abs_max_f;
        for (size_t i = tidx; i < n_elems; i += blockDim.x) {
            const size_t   index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
            reinterpret_cast<quant_packed_t*>(normed_output_quant)[index] =
                cuda_cast<quant_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0) {
            scale_orig_quant_per_token[bidx] = abs_max_f / scale_factor;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template<typename T, bool IS_BIAS>
__global__ void rmsNormWithStride(T* __restrict output,
                                  const size_t out_stride,
                                  const T* __restrict input,
                                  const size_t in_stride,
                                  const T* __restrict gamma,
                                  const T* __restrict bias,
                                  const float  eps,
                                  const size_t n,
                                  const size_t norm_size) {
    constexpr auto num_elems_T           = num_elems<T>::value;
    using float_packed_t                 = typename packed_as<float, num_elems_T>::type;
    constexpr size_t vec_size            = num_elems<T>::value;
    constexpr size_t warp_size           = 32;
    const size_t     elements_per_thread = norm_size / (warp_size * vec_size);

    const size_t sample_idx  = blockIdx.x / (n / norm_size);
    const size_t group_idx   = blockIdx.x % (n / norm_size);
    const T*     group_start = input + sample_idx * (in_stride / vec_size) + group_idx * (norm_size / vec_size);
    T*           dest_start  = output + sample_idx * (out_stride / vec_size) + group_idx * (norm_size / vec_size);

    __shared__ float smem_scale;

    float square_sum = 0.0f;
    for (size_t i = 0; i < elements_per_thread; ++i) {
        const size_t elem_idx   = i * warp_size + threadIdx.x;
        T            packed_val = group_start[elem_idx];
        auto         val        = cuda_cast<float_packed_t>(packed_val);

        square_sum += cuda_sum<float>(val * val);
    }

    float variance = warpReduceSum(square_sum) / norm_size;

    if (threadIdx.x == 0) {
        smem_scale = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (size_t i = 0; i < elements_per_thread; ++i) {
        const size_t elem_idx   = i * warp_size + threadIdx.x;
        T            packed_val = group_start[elem_idx];

        const float_packed_t val_f = cuda_cast<float_packed_t>(packed_val);
        const T              val =
            cuda_cast<T>(compute_rmsnorm<float_packed_t, T, IS_BIAS>(val_f, smem_scale, gamma, bias, elem_idx));
        dest_start[elem_idx] = cuda_cast<T>(val);
    }
}

template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, typename QUANT_OUT_T = int8_t>
void dispatch_rmsnorm_type_square_method(T*           output,
                                         T*           normed_output,
                                         const T*     input,
                                         const T*     bias,
                                         const T*     residual1,
                                         const T*     residual2,
                                         const T*     gamma,
                                         const T*     beta,
                                         const float  eps,
                                         size_t       tokens,
                                         size_t       hidden_dim,
                                         const float* scale_orig_quant_per_tensor,
                                         float*       scale_orig_quant_per_token,
                                         QUANT_OUT_T* normed_output_quant,
                                         const dim3   grid,
                                         const dim3   block,
                                         const size_t shmem_size,
                                         cudaStream_t stream) {
    if (shmem_size >= (48 << 10)) {
#if USING_CUDA
        check_cuda_value(cudaFuncSetAttribute(generalRmsNorm<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, QUANT_OUT_T>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              shmem_size));
#endif
    }
    LAUNCH_KERNEL_WITH_PDL((generalRmsNorm<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, QUANT_OUT_T>),
                           grid,
                           block,
                           shmem_size,
                           stream,
                           output,
                           normed_output,
                           input,
                           bias,
                           residual1,
                           residual2,
                           gamma,
                           beta,
                           eps,
                           tokens,
                           hidden_dim,
                           scale_orig_quant_per_tensor,
                           scale_orig_quant_per_token,
                           normed_output_quant);
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
}

template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, typename QUANT_OUT_T = int8_t>
void dispatch_rmsnorm_beta(T*           output,
                           T*           normed_output,
                           const T*     input,
                           const T*     bias,
                           const T*     residual1,
                           const T*     residual2,
                           const T*     gamma,
                           const T*     beta,
                           const float  eps,
                           size_t       tokens,
                           size_t       hidden_dim,
                           const float* scale_orig_quant_per_tensor,
                           float*       scale_orig_quant_per_token,
                           QUANT_OUT_T* normed_output_quant,
                           const dim3   grid,
                           const dim3   block,
                           const size_t shmem_size,
                           cudaStream_t stream) {
    if (beta != nullptr) {

        dispatch_rmsnorm_type_square_method<T, IS_OUTPUT, IS_BIAS, RESIDUAL, true, QUANT_OUT_T>(
            output,
            normed_output,
            input,
            bias,
            residual1,
            residual2,
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

        dispatch_rmsnorm_type_square_method<T, IS_OUTPUT, IS_BIAS, RESIDUAL, false, QUANT_OUT_T>(
            output,
            normed_output,
            input,
            bias,
            residual1,
            residual2,
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

template<typename T, bool IS_OUTPUT, bool IS_BIAS, typename QUANT_OUT_T>
void dispatch_rmsnorm_residual(T*           output,
                               T*           normed_output,
                               const T*     input,
                               const T*     bias,
                               const T*     residual1,
                               const T*     residual2,
                               const T*     gamma,
                               const T*     beta,
                               const float  eps,
                               size_t       tokens,
                               size_t       hidden_dim,
                               const float* scale_orig_quant_per_tensor,
                               float*       scale_orig_quant_per_token,
                               QUANT_OUT_T* normed_output_quant,
                               const dim3   grid,
                               const dim3   block,
                               const size_t shmem_size,
                               cudaStream_t stream) {
    if (residual1 != nullptr) {

        dispatch_rmsnorm_beta<T, IS_OUTPUT, IS_BIAS, true, QUANT_OUT_T>(output,
                                                                        normed_output,
                                                                        input,
                                                                        bias,
                                                                        residual1,
                                                                        residual2,
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

        dispatch_rmsnorm_beta<T, IS_OUTPUT, IS_BIAS, false, QUANT_OUT_T>(output,
                                                                         normed_output,
                                                                         input,
                                                                         bias,
                                                                         residual1,
                                                                         residual2,
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

template<typename T, bool IS_OUTPUT, typename QUANT_OUT_T>
void dispatch_rmsnorm_bias(T*           output,
                           T*           normed_output,
                           const T*     input,
                           const T*     bias,
                           const T*     residual1,
                           const T*     residual2,
                           const T*     gamma,
                           const T*     beta,
                           const float  eps,
                           size_t       tokens,
                           size_t       hidden_dim,
                           const float* scale_orig_quant_per_tensor,
                           float*       scale_orig_quant_per_token,
                           QUANT_OUT_T* normed_output_quant,
                           const dim3   grid,
                           const dim3   block,
                           const size_t shmem_size,
                           cudaStream_t stream) {
    if (bias != nullptr) {

        dispatch_rmsnorm_residual<T, IS_OUTPUT, true, QUANT_OUT_T>(output,
                                                                   normed_output,
                                                                   input,
                                                                   bias,
                                                                   residual1,
                                                                   residual2,
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

        dispatch_rmsnorm_residual<T, IS_OUTPUT, false, QUANT_OUT_T>(output,
                                                                    normed_output,
                                                                    input,
                                                                    bias,
                                                                    residual1,
                                                                    residual2,
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

template<typename T, typename QUANT_OUT_T>
void dispatch_rmsnorm_output(T*           output,
                             T*           normed_output,
                             const T*     input,
                             const T*     bias,
                             const T*     residual1,
                             const T*     residual2,
                             const T*     gamma,
                             const T*     beta,
                             const float  eps,
                             size_t       tokens,
                             size_t       hidden_dim,
                             const float* scale_orig_quant_per_tensor,
                             float*       scale_orig_quant_per_token,
                             QUANT_OUT_T* normed_output_quant,
                             const dim3   grid,
                             const dim3   block,
                             const size_t shmem_size,
                             cudaStream_t stream,
                             bool         is_output) {
    if (is_output) {

        dispatch_rmsnorm_bias<T, true, QUANT_OUT_T>(output,
                                                    normed_output,
                                                    input,
                                                    bias,
                                                    residual1,
                                                    residual2,
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
        dispatch_rmsnorm_bias<T, false, QUANT_OUT_T>(output,
                                                     normed_output,
                                                     input,
                                                     bias,
                                                     residual1,
                                                     residual2,
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

template<typename T, typename QUANT_OUT_T>
void invokeGeneralRmsNorm(T*           out,
                          const T*     input,
                          const T*     gamma,
                          const T*     beta,
                          const float  eps,
                          const size_t tokens,
                          const size_t hidden_dim,
                          cudaStream_t stream,
                          const float* scale,
                          float*       dynamic_scale,
                          QUANT_OUT_T* normed_output_quant) {
    constexpr size_t vec_size     = 2;
    const bool       use_vec_type = (hidden_dim % vec_size == 0)
                              && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
                                  || std::is_same<T, __nv_bfloat16>::value
#endif
                              );

    dim3 grid(tokens);
    dim3 block(std::min(hidden_dim, 1024ul));
    if (use_vec_type) {
        block.x = std::min(hidden_dim / vec_size, 1024ul);
    }
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    const size_t shmem_size = hidden_dim * sizeof(T);

    if (use_vec_type) {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_rmsnorm_output(reinterpret_cast<Tp*>(out),
                                reinterpret_cast<Tp*>(out),
                                reinterpret_cast<Tp*>(out),
                                (const Tp*)nullptr,
                                reinterpret_cast<const Tp*>(input),
                                (const Tp*)nullptr,
                                reinterpret_cast<const Tp*>(gamma),
                                reinterpret_cast<const Tp*>(beta),
                                eps,
                                tokens,
                                hidden_dim,
                                scale,
                                dynamic_scale,
                                normed_output_quant,
                                grid,
                                block,
                                shmem_size,
                                stream,
                                false);
    } else {
        dispatch_rmsnorm_output(out,
                                out,
                                (const T*)out,
                                (const T*)nullptr,
                                input,
                                (const T*)nullptr,
                                gamma,
                                beta,
                                eps,
                                tokens,
                                hidden_dim,
                                scale,
                                dynamic_scale,
                                normed_output_quant,
                                grid,
                                block,
                                shmem_size,
                                stream,
                                false);
    }
}

template<typename T>
void invokeRmsNormWithStride(T* __restrict output,
                             const size_t out_stride,
                             const T* __restrict input,
                             const size_t in_stride,
                             const T* __restrict gamma,
                             const T* __restrict beta,
                             const float  layernorm_eps,
                             const size_t m,
                             const size_t n,
                             const size_t norm_size,
                             cudaStream_t stream) {
    constexpr size_t vec_size  = 2;
    constexpr size_t warp_size = 32;

    // 参数校验
    if (n % norm_size != 0) {
        throw std::invalid_argument("n must be divisible by norm_size");
    }
    if (norm_size % (warp_size * vec_size) != 0) {
        throw std::invalid_argument("norm_size must be multiple of " + std::to_string(warp_size * vec_size));
    }

    const size_t num_heads = n / norm_size;
    dim3         grid(m * num_heads);  // 每个block处理一个样本的一个头
    dim3         block(warp_size);

    using Tp     = typename packed_as<T, vec_size>::type;
    bool is_bias = beta != nullptr;
    if (is_bias) {
        rmsNormWithStride<Tp, true><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(output),
                                                                out_stride,
                                                                reinterpret_cast<const Tp*>(input),
                                                                in_stride,
                                                                reinterpret_cast<const Tp*>(gamma),
                                                                reinterpret_cast<const Tp*>(beta),
                                                                layernorm_eps,
                                                                n,
                                                                norm_size);
    } else {
        rmsNormWithStride<Tp, false><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(output),
                                                                 out_stride,
                                                                 reinterpret_cast<const Tp*>(input),
                                                                 in_stride,
                                                                 reinterpret_cast<const Tp*>(gamma),
                                                                 reinterpret_cast<const Tp*>(beta),
                                                                 layernorm_eps,
                                                                 n,
                                                                 norm_size);
    }
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
}

template<typename T, typename QUANT_OUT_T>
void invokeAddBiasResidualRmsNorm(T*           output,
                                  T*           normed_output,
                                  const T*     input,
                                  const T*     bias,
                                  const T*     residual,
                                  const T*     residual2,
                                  const T*     gamma,
                                  const T*     beta,
                                  const float  eps,
                                  const size_t tokens,
                                  const size_t hidden_dim,
                                  cudaStream_t stream,
                                  const float* scale,
                                  float*       dynamic_scale,
                                  QUANT_OUT_T* normed_output_quant) {
    dim3 grid(tokens);
    dim3 block(std::min(hidden_dim, 1024ul));
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
        dispatch_rmsnorm_output(reinterpret_cast<Tp*>(output),
                                reinterpret_cast<Tp*>(normed_output),
                                reinterpret_cast<const Tp*>(input),
                                reinterpret_cast<const Tp*>(bias),
                                reinterpret_cast<const Tp*>(residual),
                                reinterpret_cast<const Tp*>(residual2),
                                reinterpret_cast<const Tp*>(gamma),
                                reinterpret_cast<const Tp*>(beta),
                                eps,
                                tokens,
                                hidden_dim,
                                scale,
                                dynamic_scale,
                                normed_output_quant,
                                grid,
                                block,
                                shmem_size,
                                stream,
                                true);
    } else {
        dispatch_rmsnorm_output(output,
                                normed_output,
                                input,
                                bias,
                                residual,
                                residual2,
                                gamma,
                                beta,
                                eps,
                                tokens,
                                hidden_dim,
                                scale,
                                dynamic_scale,
                                normed_output_quant,
                                grid,
                                block,
                                shmem_size,
                                stream,
                                true);
    }
}

#define INSTANTIATE_GENERAL_RMSNORM(T, QUANT_OUT_T)                                                                    \
    template void invokeGeneralRmsNorm(T*           out,                                                               \
                                       const T*     input,                                                             \
                                       const T*     gamma,                                                             \
                                       const T*     beta,                                                              \
                                       const float  eps,                                                               \
                                       const size_t tokens,                                                            \
                                       const size_t hidden_dim,                                                        \
                                       cudaStream_t stream,                                                            \
                                       const float* scale,                                                             \
                                       float*       dynamic_scale,                                                     \
                                       QUANT_OUT_T* normed_output_quant);

INSTANTIATE_GENERAL_RMSNORM(float, int8_t);
INSTANTIATE_GENERAL_RMSNORM(half, int8_t);

#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_RMSNORM(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_GENERAL_RMSNORM(float, __nv_fp8_e4m3);
INSTANTIATE_GENERAL_RMSNORM(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_RMSNORM(__nv_bfloat16, __nv_fp8_e4m3);
#endif  // ENABLE_BF16
#endif  // ENABLE_FP8

#define INSTANTIATE_ADD_BIAS_RESL_RMSNORM(T, QUANT_OUT_T)                                                              \
    template void invokeAddBiasResidualRmsNorm(T*           output,                                                    \
                                               T*           normed_output,                                             \
                                               const T*     input,                                                     \
                                               const T*     bias,                                                      \
                                               const T*     resiudal,                                                  \
                                               const T*     resiudal2,                                                 \
                                               const T*     gamma,                                                     \
                                               const T*     beta,                                                      \
                                               const float  eps,                                                       \
                                               const size_t tokens,                                                    \
                                               const size_t hidden_dim,                                                \
                                               cudaStream_t stream,                                                    \
                                               const float* scale,                                                     \
                                               float*       dynamic_scale,                                             \
                                               QUANT_OUT_T* normed_output_quant);

INSTANTIATE_ADD_BIAS_RESL_RMSNORM(float, int8_t);
INSTANTIATE_ADD_BIAS_RESL_RMSNORM(half, int8_t);
#ifdef ENABLE_BF16
INSTANTIATE_ADD_BIAS_RESL_RMSNORM(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_ADD_BIAS_RESL_RMSNORM(float, __nv_fp8_e4m3);
INSTANTIATE_ADD_BIAS_RESL_RMSNORM(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_ADD_BIAS_RESL_RMSNORM(__nv_bfloat16, __nv_fp8_e4m3);
#endif  // ENABLE_BF16
#endif  // ENABLE_FP8

#define INSTANTIATE_STRIDED_RMSNORM(T)                                                                                 \
    template void invokeRmsNormWithStride(T* __restrict output,                                                        \
                                          const size_t out_stride,                                                     \
                                          const T* __restrict input,                                                   \
                                          const size_t in_stride,                                                      \
                                          const T* __restrict gamma,                                                   \
                                          const T* __restrict beta,                                                    \
                                          const float  layernorm_eps,                                                  \
                                          const size_t m,                                                              \
                                          const size_t n,                                                              \
                                          const size_t norm_size,                                                      \
                                          cudaStream_t stream);
INSTANTIATE_STRIDED_RMSNORM(float);
INSTANTIATE_STRIDED_RMSNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_STRIDED_RMSNORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_STRIDED_RMSNORM

}  // namespace rtp_llm
