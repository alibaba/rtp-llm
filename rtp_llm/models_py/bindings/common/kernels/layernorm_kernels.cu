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

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/common/kernels/layernorm_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include <cuda_fp8.h>
#endif

#if USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

// wont't support new features
namespace rtp_llm {

template<typename Tf, typename T, bool IS_BETA>
__inline__ __device__ Tf
compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i) {
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (IS_BETA) {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

#if USING_CUDA
// Vision-BERT/DeepGEMM fast path.  This deliberately mirrors the existing
// USE_DIFF_OF_SQUARES AddBiasResidualLayerNorm reduction and BF16 rounding,
// then quantizes the rounded values from shared memory into DeepGEMM's
// per-128 UE8M0 TMA layout.  Keeping both outputs preserves the residual path
// while avoiding a second global-memory read and a separate quant kernel.
template<typename T>
__global__ void generalAddBiasResidualLayerNormQuantFp8(T*             output,
                                                        T*             normed_output,
                                                        const T*       input,
                                                        const T*       bias,
                                                        const T*       residual,
                                                        const T*       gamma,
                                                        const T*       beta,
                                                        float          eps,
                                                        int            hidden_dim,
                                                        __nv_fp8_e4m3* output_quant,
                                                        uint32_t*      scales,
                                                        int            scale_stride) {
    constexpr int packed_elements = num_elems<T>::value;
    using float_packed_t          = typename packed_as<float, packed_elements>::type;
    using scalar_t                = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char shared_storage[];
    T*                                              values = reinterpret_cast<T*>(shared_storage);
    __shared__ float                                shared_mean;
    __shared__ float                                shared_inv_std;

    const int row        = blockIdx.x;
    const int packed_dim = hidden_dim / packed_elements;
    float     local_sum  = 0.0f;
    float     local_sq   = 0.0f;

    for (int i = threadIdx.x; i < packed_dim; i += blockDim.x) {
        const int index = row * packed_dim + i;
        T         value = input[index];
        if (bias != nullptr) {
            value = add(value, ldg(&bias[i]));
        }
        value                        = add(value, ldg(&residual[index]));
        output[index]                = value;
        values[i]                    = value;
        const float_packed_t value_f = cuda_cast<float_packed_t>(value);
        local_sum += cuda_sum<float>(value_f);
        local_sq += cuda_sum<float>(value_f * value_f);
    }

    float moments[2] = {local_sum, local_sq};
    blockReduceSumV2<float, 2>(moments);
    if (threadIdx.x == 0) {
        shared_mean          = moments[0] / hidden_dim;
        const float variance = moments[1] / hidden_dim - shared_mean * shared_mean;
        shared_inv_std       = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < packed_dim; i += blockDim.x) {
        const int            index      = row * packed_dim + i;
        const float_packed_t value_f    = cuda_cast<float_packed_t>(values[i]);
        const T              normalized = cuda_cast<T>(
            compute_layernorm<float_packed_t, T, true>(value_f, shared_mean, shared_inv_std, gamma, beta, i));
        normed_output[index] = normalized;
        values[i]            = normalized;
    }
    __syncthreads();

    constexpr int group_size        = 128;
    constexpr int threads_per_group = 32;
    const int     groups_per_row    = hidden_dim / group_size;
    const int     group             = threadIdx.x / threads_per_group;
    const int     lane              = threadIdx.x % threads_per_group;
    if (group >= groups_per_row) {
        return;
    }

    const scalar_t* scalar_values = reinterpret_cast<const scalar_t*>(values) + group * group_size;
    float           local_absmax  = 1e-4f;
#pragma unroll
    for (int i = lane; i < group_size; i += threads_per_group) {
        local_absmax = fmaxf(local_absmax, fabsf(static_cast<float>(scalar_values[i])));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffffu, local_absmax, delta));
    }

    const float    raw_scale       = fmaxf(local_absmax / 448.0f, 1e-10f);
    const uint32_t scale_bits      = __float_as_uint(raw_scale);
    const int      floor_exponent  = static_cast<int>((scale_bits >> 23) & 0xff) - 127;
    const int      scale_exponent  = floor_exponent + ((scale_bits & 0x7fffffU) != 0U);
    const int      biased_exponent = scale_exponent + 127;
    const float    group_scale     = __int_as_float(biased_exponent << 23);
    if (lane == 0) {
        const int packed_col = group / 4;
        const int pack_idx   = group % 4;
        reinterpret_cast<uint8_t*>(scales)[packed_col * scale_stride * 4 + row * 4 + pack_idx] =
            static_cast<uint8_t>(biased_exponent);
    }

    __nv_fp8_e4m3* row_quant = output_quant + row * hidden_dim + group * group_size;
#pragma unroll
    for (int i = lane; i < group_size; i += threads_per_group) {
        const float value = static_cast<float>(scalar_values[i]);
        row_quant[i]      = __nv_fp8_e4m3(fminf(fmaxf(value / group_scale, -448.0f), 448.0f));
    }
}

template<typename T>
void invokeGeneralAddBiasResidualLayerNormQuantFp8(T*           out,
                                                   T*           norm_output,
                                                   const T*     input,
                                                   const T*     bias,
                                                   const T*     residual,
                                                   const T*     gamma,
                                                   const T*     beta,
                                                   float        eps,
                                                   int          tokens,
                                                   int          hidden_dim,
                                                   void*        out_quant,
                                                   uint32_t*    scales,
                                                   int          scale_stride,
                                                   cudaStream_t stream) {
    using packed_t                = typename packed_as<T, 2>::type;
    constexpr int packed_elements = 2;
    const int     packed_dim      = hidden_dim / packed_elements;
    dim3          block(32 * ((packed_dim + 31) / 32));
    dim3          grid(tokens);
    const size_t  shared_bytes = hidden_dim * sizeof(T);
    generalAddBiasResidualLayerNormQuantFp8<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<packed_t*>(out),
        reinterpret_cast<packed_t*>(norm_output),
        reinterpret_cast<const packed_t*>(input),
        reinterpret_cast<const packed_t*>(bias),
        reinterpret_cast<const packed_t*>(residual),
        reinterpret_cast<const packed_t*>(gamma),
        reinterpret_cast<const packed_t*>(beta),
        eps,
        hidden_dim,
        static_cast<__nv_fp8_e4m3*>(out_quant),
        scales,
        scale_stride);
}

template void invokeGeneralAddBiasResidualLayerNormQuantFp8<half>(half*,
                                                                  half*,
                                                                  const half*,
                                                                  const half*,
                                                                  const half*,
                                                                  const half*,
                                                                  const half*,
                                                                  float,
                                                                  int,
                                                                  int,
                                                                  void*,
                                                                  uint32_t*,
                                                                  int,
                                                                  cudaStream_t);
#ifdef ENABLE_BF16
template void invokeGeneralAddBiasResidualLayerNormQuantFp8<__nv_bfloat16>(__nv_bfloat16*,
                                                                           __nv_bfloat16*,
                                                                           const __nv_bfloat16*,
                                                                           const __nv_bfloat16*,
                                                                           const __nv_bfloat16*,
                                                                           const __nv_bfloat16*,
                                                                           const __nv_bfloat16*,
                                                                           float,
                                                                           int,
                                                                           int,
                                                                           void*,
                                                                           uint32_t*,
                                                                           int,
                                                                           cudaStream_t);
#endif
#endif

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
        check_cuda_value(cudaFuncSetAttribute(generalLayerNorm<T,
                                                               QUANT_OUT_T,
                                                               IS_OUTPUT,
                                                               IS_BIAS,
                                                               RESIDUAL,
                                                               IS_BETA,
                                                               RETURN_NORMED_OUTPUT,
                                                               USE_DIFF_OF_SQUARES>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              shmem_size));
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
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
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
