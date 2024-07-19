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

#if USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#endif

// wont't support new features
namespace fastertransformer{
#if USING_ROCM
using namespace rocm;
#endif

__device__ __forceinline__ int64_t loadOffset(int head_num,
                                              int size_per_head)
{
    // [[q_head_1],[q_head_2]...[k_head_1],[k_head_2]...[v_head_1],[v_head_2]...]
    int head_id = blockIdx.y;
    int batch_id = blockIdx.x;
    int offset = batch_id * head_num * size_per_head + size_per_head * head_id;
    return offset;
}

template<typename T>
__global__ void qkLayerNorm(T* __restrict qkv,
                            const T* __restrict gamma,
                            const float layernorm_eps,
                            int head_num,
                            int size_per_head)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    constexpr size_t warp_size = 32;
    const int vec_size_per_head = size_per_head / num_elems_T;
    const int n_elems = vec_size_per_head / warp_size;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;

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
        auto diff = val_f - s_mean;
        local_var_sum += cuda_sum<float>(diff * diff);
    }
    variance = warpReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / size_per_head + layernorm_eps);
    }
    __syncthreads();

    for (int i = 0; i < n_elems; i++) {
        auto index = loadOffset(head_num, vec_size_per_head) + tid * n_elems + i;
	auto gamma_index = blockIdx.y * vec_size_per_head + tid * n_elems + i;
        auto val_f = cuda_cast<float_packed_t>(ldg(&qkv[index]));
	auto val_gamma = cuda_cast<float_packed_t>(gamma[gamma_index]);
        qkv[index] = cuda_cast<T>((val_f - s_mean) * s_variance * val_gamma);
    }
}

template<typename T>
void invokeQkLayerNorm(T* __restrict qkv,
		       const T* __restrict gamma,
		       const float layernorm_eps,
		       const int tokens,
		       const int head_num,
		       const int head_num_kv,
		       const int size_per_head,
		       cudaStream_t stream)
{
    constexpr size_t vec_size = 2;
    constexpr size_t warp_size = 32;

    if (size_per_head % warp_size != 0) {
        throw std::invalid_argument("not supported size_per_head: " + std::to_string(size_per_head));
    }
    dim3 grid(tokens, head_num + head_num_kv);
    dim3 block(warp_size);

    int total_head_num = head_num + 2 * head_num_kv;
    using Tp = typename packed_as<T, vec_size>::type;
    qkLayerNorm<Tp><<<grid, block, 0, stream>>>(reinterpret_cast<Tp*>(qkv), reinterpret_cast<const Tp*>(gamma),
						layernorm_eps, total_head_num, size_per_head);
}

#define INSTANTIATE_QK_LAYERNORM(T)				\
  template void invokeQkLayerNorm(T* __restrict qkv,		\
				  const T* __restrict gamma,	\
				  const float layernorm_eps,	\
				  const int tokens,		\
				  const int head_num,		\
				  const int head_num_kv,	\
				  const int size_per_head,	\
				  cudaStream_t stream)
INSTANTIATE_QK_LAYERNORM(float);
INSTANTIATE_QK_LAYERNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_QK_LAYERNORM(__nv_bfloat16);
#endif


template <typename Tf, typename T, bool IS_BETA>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i)
{
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (IS_BETA)
    {
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
template <typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, bool RETURN_NORMED_OUTPUT, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
    T_scalar amax(1e-6f);

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        // const T val = input[bidx * n_elems + i];
        const int index = bidx * n_elems + i;
        T val = cuda_cast<T>(0.0f);
        // const T val = input[index];
        if (IS_BIAS)
        {
            val = add(val, ldg(&bias[i]));
        }
        if (RESIDUAL)
        {
            val = add(val, ldg(&residual[index]));
        }
        if (IS_OUTPUT)
        {
            T in_val;
            if (with_per_tensor_scaling)
            {
                in_val = cuda_cast<T>(cuda_cast<float_packed_t>(reinterpret_cast<const Int32_Packed_T*>(input)[index])
                    * scale_orig_quant);
            }
            else
            {
                in_val = input[index];
            }
            val = add(val, in_val);
            if (!RETURN_NORMED_OUTPUT) {
                output[index] = val;
            }
        }
        shmem[i] = val;

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = shmem[i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
        const T val
            = cuda_cast<T>(compute_layernorm<float_packed_t, T, IS_BETA>(val_f, s_mean, s_variance, gamma, beta, i));
        if (RETURN_NORMED_OUTPUT && IS_OUTPUT) {
            output[index] = val;
        }
        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            shmem[i] = val;
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(shmem[i]);
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, bool RETURN_NORMED_OUTPUT, bool USE_DIFF_OF_SQUARES>
void dispatch_layernorm_type_square_method(T* output, T* normed_output, const T* input, const T* bias,
    const T* residual, const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream)
{
    if (shmem_size >= (48 << 10))
    {
#if USING_CUDA
        cudaError_t ret
            = cudaFuncSetAttribute(generalLayerNorm<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, RETURN_NORMED_OUTPUT, USE_DIFF_OF_SQUARES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
#endif
    }
    generalLayerNorm<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, RETURN_NORMED_OUTPUT, USE_DIFF_OF_SQUARES>
        <<<grid, block, shmem_size, stream>>>(output, normed_output, input, bias, residual, gamma, beta, eps, tokens,
            hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant);
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA, bool RETURN_NORMED_OUTPUT>
void dispatch_layernorm_return_normed(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream, bool use_diff_of_squares)
{
    if (use_diff_of_squares)
    {
        dispatch_layernorm_type_square_method<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, RETURN_NORMED_OUTPUT, true>(output, normed_output,
            input, bias, residual, gamma, beta, eps, tokens, hidden_dim, scale_orig_quant_per_tensor,
            scale_orig_quant_per_token, normed_output_quant, grid, block, shmem_size, stream);
    }
    else
    {
        dispatch_layernorm_type_square_method<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, RETURN_NORMED_OUTPUT, false>(output, normed_output,
            input, bias, residual, gamma, beta, eps, tokens, hidden_dim, scale_orig_quant_per_tensor,
            scale_orig_quant_per_token, normed_output_quant, grid, block, shmem_size, stream);
    }
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIDUAL, bool IS_BETA>
void dispatch_layernorm_type(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream, bool use_diff_of_squares, bool return_normed_output)
{
    if (return_normed_output)
    {
        dispatch_layernorm_return_normed<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, true>(output, normed_output,
            input, bias, residual, gamma, beta, eps, tokens, hidden_dim, scale_orig_quant_per_tensor,
            scale_orig_quant_per_token, normed_output_quant, grid, block, shmem_size, stream, use_diff_of_squares);
    }
    else
    {
        dispatch_layernorm_return_normed<T, IS_OUTPUT, IS_BIAS, RESIDUAL, IS_BETA, false>(output, normed_output,
            input, bias, residual, gamma, beta, eps, tokens, hidden_dim, scale_orig_quant_per_tensor,
            scale_orig_quant_per_token, normed_output_quant, grid, block, shmem_size, stream, use_diff_of_squares);
    }
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS, bool RESIUDAL>
void dispatch_layernorm_beta(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream, bool use_diff_of_squares, bool return_normed_output)
{
    if (beta != nullptr)
    {
        dispatch_layernorm_type<T, IS_OUTPUT, IS_BIAS, RESIUDAL, true>(output, normed_output, input, bias, residual,
            gamma, beta, eps, tokens, hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token,
            normed_output_quant, grid, block, shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
    else
    {
        dispatch_layernorm_type<T, IS_OUTPUT, IS_BIAS, RESIUDAL, false>(output, normed_output, input, bias, residual,
            gamma, beta, eps, tokens, hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token,
            normed_output_quant, grid, block, shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS>
void dispatch_layernorm_residual(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream, bool use_diff_of_squares, bool return_normed_output)
{
    if (residual != nullptr)
    {
        dispatch_layernorm_beta<T, IS_OUTPUT, IS_BIAS, true>(output, normed_output, input, bias, residual, gamma, beta,
            eps, tokens, hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant, grid,
            block, shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
    else
    {
        dispatch_layernorm_beta<T, IS_OUTPUT, IS_BIAS, false>(output, normed_output, input, bias, residual, gamma, beta,
            eps, tokens, hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant, grid,
            block, shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
}

template <typename T, bool IS_OUTPUT>
void dispatch_layernorm_bias(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream, bool use_diff_of_squares, bool return_normed_output)
{
    if (bias != nullptr)
    {
        dispatch_layernorm_residual<T, IS_OUTPUT, true>(output, normed_output, input, bias, residual, gamma, beta, eps,
            tokens, hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant, grid,
            block, shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
    else
    {
        dispatch_layernorm_residual<T, IS_OUTPUT, false>(output, normed_output, input, bias, residual, gamma, beta, eps,
            tokens, hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant, grid,
            block, shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
}

template <typename T>
void dispatch_layernorm_output(T* output, T* normed_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, int tokens, int hidden_dim,
    const float* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token, int8_t* normed_output_quant,
    const dim3 grid, const dim3 block, const size_t shmem_size, cudaStream_t stream, bool use_diff_of_squares,
    bool is_output, bool return_normed_output)
{
    if (is_output)
    {
        dispatch_layernorm_bias<T, true>(output, normed_output, input, bias, residual, gamma, beta, eps, tokens,
            hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant, grid, block,
            shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
    else
    {
        dispatch_layernorm_bias<T, false>(output, normed_output, input, bias, residual, gamma, beta, eps, tokens,
            hidden_dim, scale_orig_quant_per_tensor, scale_orig_quant_per_token, normed_output_quant, grid, block,
            shmem_size, stream, use_diff_of_squares, return_normed_output);
    }
}

template <typename T>
void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream, bool use_diff_of_squares, const float* scale, float* dynamic_scale,
    int8_t* out_quant, bool return_normed_output)
{
    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size = 2;
    const size_t shmem_size = hidden_dim * sizeof(T);
    const bool use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_layernorm_output(reinterpret_cast<Tp*>(out), reinterpret_cast<Tp*>(out),
            reinterpret_cast<const Tp*>(out), (const Tp*) nullptr, reinterpret_cast<const Tp*>(input),
            reinterpret_cast<const Tp*>(gamma), reinterpret_cast<const Tp*>(beta), eps, tokens, hidden_dim, scale,
            dynamic_scale, out_quant, grid, block, shmem_size, stream, use_diff_of_squares, false, return_normed_output);
    }
    else
    {
        dispatch_layernorm_output(out, out, (const T*) out, (const T*) nullptr, input, gamma, beta, eps, tokens,
            hidden_dim, scale, dynamic_scale, out_quant, grid, block, shmem_size, stream, use_diff_of_squares, false, return_normed_output);
    }
}

template <typename T>
void invokeGeneralAddBiasResidualLayerNorm(T* out, T* norm_output, const T* input, const T* bias, const T* residual,
    const T* gamma, const T* beta, const float eps, const int tokens, const int hidden_dim, cudaStream_t stream,
    bool use_diff_of_squares, const float* scale, float* dynamic_scale, int8_t* out_quant, bool return_normed_output)
{
    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size = 2;
    const size_t shmem_size = hidden_dim * sizeof(T);
    const bool use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_layernorm_output(reinterpret_cast<Tp*>(out), reinterpret_cast<Tp*>(norm_output),
            reinterpret_cast<const Tp*>(input), reinterpret_cast<const Tp*>(bias),
            reinterpret_cast<const Tp*>(residual), reinterpret_cast<const Tp*>(gamma),
            reinterpret_cast<const Tp*>(beta), eps, tokens, hidden_dim, scale, dynamic_scale, out_quant, grid, block,
            shmem_size, stream, use_diff_of_squares, true, return_normed_output);
    }
    else
    {
        dispatch_layernorm_output(out, norm_output, input, bias, residual, gamma, beta, eps, tokens, hidden_dim, scale,
            dynamic_scale, out_quant, grid, block, shmem_size, stream, use_diff_of_squares, true, return_normed_output);
    }
}

#define INSTANTIATE_GENERAL_LAYERNORM(T)                                                                               \
    template void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps,       \
        const int tokens, const int hidden_dim, cudaStream_t stream, bool use_diff_of_squares, const float* scale,     \
        float* dynamic_scale, int8_t* out_quant, bool return_normed_output);

INSTANTIATE_GENERAL_LAYERNORM(float);
INSTANTIATE_GENERAL_LAYERNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_LAYERNORM(__nv_bfloat16);
#endif

#define INSTANTIATE_GENERAL_ADD_BIAS_RESDIAUL_LAYERNORM(T)                                                             \
    template void invokeGeneralAddBiasResidualLayerNorm(T* out, T* norm_output, const T* input, const T* bias,         \
        const T* residual, const T* gamma, const T* beta, const float eps, const int tokens, const int hidden_dim,     \
        cudaStream_t stream, bool use_diff_of_squares, const float* scale, float* dynamic_scale, int8_t* out_quant, bool return_normed_output);

INSTANTIATE_GENERAL_ADD_BIAS_RESDIAUL_LAYERNORM(float);
INSTANTIATE_GENERAL_ADD_BIAS_RESDIAUL_LAYERNORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_ADD_BIAS_RESDIAUL_LAYERNORM(__nv_bfloat16);
#endif

} // namespace fastertransformer
