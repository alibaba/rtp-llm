/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "cuda_fp8_utils.h"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include <algorithm>
#include <cstdio>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>

using namespace rtp_llm;
namespace tensorrt_llm
{
namespace common
{
#ifdef ENABLE_FP8

constexpr int CTA_SIZE = 256;

template <bool QUANTIZE>
__inline__ __device__ float scale(float a, float b)
{
    return QUANTIZE ? a / b : a * b;
}

template <QuantizeMode QUANTIZE_MODE, bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda)
{
    for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += blockDim.x * gridDim.x)
    {

        if (QUANTIZE_MODE == QuantizeMode::PER_CHANNEL)
        {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i % lda])));
        }
        else if (QUANTIZE_MODE == QuantizeMode::PER_TOKEN)
        {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i / lda])));
        }
        else if (QUANTIZE_MODE == QuantizeMode::PER_TENSOR)
        {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[0])));
	}     
    }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        scaleMatrix<QuantizeMode::PER_CHANNEL, true>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        scaleMatrix<QuantizeMode::PER_TOKEN, true><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        scaleMatrix<QuantizeMode::PER_TENSOR, true><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    sync_check_cuda_error();
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeDequantizeMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        scaleMatrix<QuantizeMode::PER_CHANNEL, false>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        scaleMatrix<QuantizeMode::PER_TOKEN, false><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        scaleMatrix<QuantizeMode::PER_TENSOR, false>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    sync_check_cuda_error();
}

template <typename T_FAKE, typename T_OUT, typename T_IN>
__global__ void fakeQuantize(T_OUT* dst, const T_IN* src, const int64_t numel)
{
    for (int64_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        T_FAKE tmp = (T_FAKE) (static_cast<float>(src[tid]));
        dst[tid] = (T_OUT) (static_cast<float>(tmp));
    }
}

template <typename T_FAKE, typename T_OUT, typename T_IN>
void invokeFakeQuantize(T_OUT* dst, const T_IN* src, const int64_t numel, cudaStream_t stream)
{
    fakeQuantize<T_FAKE><<<1024, CTA_SIZE, 0, stream>>>(dst, src, numel);
    sync_check_cuda_error();
}

template void invokeFakeQuantize<__nv_fp8_e4m3, float, float>(
    float* dst, float const* src, const int64_t numel, cudaStream_t stream);
template void invokeFakeQuantize<float, float, __nv_fp8_e4m3>(
    float* dst, __nv_fp8_e4m3 const* src, const int64_t numel, cudaStream_t stream);
template void invokeFakeQuantize<__nv_fp8_e4m3, half, half>(
    half* dst, half const* src, const int64_t numel, cudaStream_t stream);
template void invokeFakeQuantize<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* dst, __nv_bfloat16 const* src, const int64_t numel, cudaStream_t stream);

template void invokeFakeQuantize<float, half, float>(
    half* dst, float const* src, const int64_t numel, cudaStream_t stream);

__device__ float atomicMaxExtd(float* address, float val)
{
    assert(val >= 0);
    unsigned int* address_as_u = reinterpret_cast<unsigned int*>(address);
    unsigned int old = atomicMax(address_as_u, __float_as_uint(val));
    return __uint_as_float(old);
}

template <typename T>
inline __device__ T atomicMaxExtdV2(T* address, T val)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, __nv_bfloat16>, "T needs to be either half or bfloat16");
    // The address in 64 bits.
    uint64_t address_u64 = reinterpret_cast<uint64_t const&>(address);

    // Pack the input value into 32 bits.
    union
    {
        T v[2];
        uint16_t u[2];
    } old, tmp = {};

    int const loc = (address_u64 & 0x2) >> 1;
    tmp.v[loc] = val;

    // 4B aligned pointer.
    auto aligned_address = reinterpret_cast<T*>(address_u64 & ~0x3ull);

    if constexpr (std::is_same_v<T, half>)
    {
        asm volatile("atom.global.v2.f16.max.noftz {%0, %1}, [%2], {%3, %4};"
                     : "=h"(old.u[0]), "=h"(old.u[1])
                     : "l"(aligned_address), "h"(tmp.u[0]), "h"(tmp.u[1]));
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        asm volatile("atom.global.v2.bf16.max.noftz {%0, %1}, [%2], {%3, %4};"
                     : "=h"(old.u[0]), "=h"(old.u[1])
                     : "l"(aligned_address), "h"(tmp.u[0]), "h"(tmp.u[1]));
    }

    // Return the correct half.
    return old.v[loc];
#endif
}

__device__ half atomicMaxExtd(half* address, half val)
{
    unsigned short int* address_as_u = reinterpret_cast<unsigned short int*>(address);
    unsigned short int old = *address_as_u, assumed;

    while (val > __ushort_as_half(old))
    {
        assumed = old;
        old = atomicCAS(address_as_u, assumed, __half_as_ushort(val));
    }

    return __ushort_as_half(old);
}

__device__ __nv_bfloat16 atomicMaxExtd(__nv_bfloat16* address, __nv_bfloat16 val)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    unsigned short int* address_as_u = reinterpret_cast<unsigned short int*>(address);
    unsigned short int old = *address_as_u, assumed;

    while (val > __ushort_as_bfloat16(old))
    {
        assumed = old;
        old = atomicCAS(address_as_u, assumed, __bfloat16_as_ushort(val));
    }

    return __ushort_as_bfloat16(old);
#else
    assert(0);
    asm volatile("brkpt;\n" ::);
    return __nv_bfloat16(0);
#endif
}

template <QuantizeMode QUANTIZE_MODE, typename T_S, typename T_W>
__global__ void computeFP8QuantizeScale(T_S* quant_ptr, const T_W* weights, const int64_t size, const int64_t n)
{
    constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);
    if (QUANTIZE_MODE == QuantizeMode::PER_CHANNEL)
    {
        for (int64_t col = threadIdx.x; col < n; col += blockDim.x)
        {
            float max = 0.f;
            for (int64_t i = col + n * blockIdx.x; i < size; i += gridDim.x * n)
            {
                auto val = fabs(static_cast<float>(weights[i]));
                max = max > val ? max : val;
            }
            auto const scale = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
            if constexpr (std::is_same_v<T_S, float>)
            {
                atomicMaxExtd(quant_ptr + col, scale);
            }
            else
            {
                auto const address_u64 = reinterpret_cast<uint64_t>(quant_ptr + col);
                if ((col == 0 && address_u64 % 4 != 0) || (col == n - 1 && address_u64 % 4 == 0))
                    atomicMaxExtd(quant_ptr + col, scale);
                else
                    atomicMaxExtdV2(quant_ptr + col, scale);
            }
#else // Vector atomics require __CUDA_ARCH__ >= 900
            atomicMaxExtd(quant_ptr + col, scale);
#endif
        }
    }
    else if (QUANTIZE_MODE == QuantizeMode::PER_TOKEN)
    {
        auto const nrows = size / n;
        for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x)
        {
            float max = 0.f;
            for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
            {
                auto val = fabs(static_cast<float>(weights[row * n + i]));
                max = max > val ? max : val;
            }
            max = blockReduceMax<float>(max);
            if (threadIdx.x == 0)
            {
                auto const scale = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
                quant_ptr[row] = scale;
            }
        }
    }
    else if (QUANTIZE_MODE == QuantizeMode::PER_TENSOR)
    {
        float max = 0.f;
        for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x)
        {
            auto val = fabs(static_cast<float>(weights[i]));
            max = max > val ? max : val;
        }
        max = blockReduceMax<float>(max);
        if (threadIdx.x == 0)
        {
            auto const scale = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
            atomicMaxExtd(quant_ptr, scale);
        }
    }
}

template <typename T_S, typename T_W>
void invokeComputeFP8QuantizeScale(T_S* quant_ptr, const T_W* weights, const int64_t numel, const int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        dim3 block(CTA_SIZE);
        dim3 grid(numel / lda);
        computeFP8QuantizeScale<QuantizeMode::PER_TOKEN><<<grid, block, 0, stream>>>(quant_ptr, weights, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        dim3 block(CTA_SIZE);
        dim3 grid((lda + CTA_SIZE - 1) / CTA_SIZE);
        cudaMemsetAsync(quant_ptr, 0, lda * sizeof(T_S), stream);
        sync_check_cuda_error();
        computeFP8QuantizeScale<QuantizeMode::PER_CHANNEL><<<grid, block, 0, stream>>>(quant_ptr, weights, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        dim3 block(1024);
        dim3 grid(1024);
        cudaMemsetAsync(quant_ptr, 0, sizeof(T_S), stream);
        sync_check_cuda_error();
        computeFP8QuantizeScale<QuantizeMode::PER_TENSOR><<<grid, block, 0, stream>>>(quant_ptr, weights, numel, lda);
    }
    sync_check_cuda_error();
}

#define DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(type_scale, type_in)                                                  \
    template void invokeComputeFP8QuantizeScale<type_scale, type_in>(type_scale * input_scale, type_in const* weights, \
        int64_t numel, int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);

DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(half, half);
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(float, half);
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(float, float);
#ifdef ENABLE_BF16
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(__nv_bfloat16, __nv_bfloat16);
DEFINE_INVOKE_COMPUTE_FP8_QUANTIZE_SCALE(float, __nv_bfloat16);
#endif

template <typename T_OUT, typename T_S, typename T_IN>
__global__ void dynamicQuantizeMatrixPerToken(
    T_OUT* output, T_S* quant_ptr, T_IN const* input, int64_t numel, int64_t lda)
{
    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T_IN* shmem = reinterpret_cast<T_IN*>(_shmem);
    constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);
    auto const nrows = numel / lda;
    for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x)
    {
        float max = 0.f;
        for (int64_t i = threadIdx.x; i < lda; i += blockDim.x)
        {
            auto const in = input[row * lda + i];
            shmem[i] = in;
            auto val = fabs(static_cast<float>(in));
            max = max > val ? max : val;
        }
        max = blockAllReduceMax<float>(max); // __syncthreads() called so we can read shmem
        auto const s = (T_S) std::max(max / FP8_E4M3_MAX, min_scaling_factor);
        for (int64_t i = threadIdx.x; i < lda; i += blockDim.x)
        {
            // true means we are quantizing
            output[row * lda + i] = (T_OUT) scale<true>(static_cast<float>(shmem[i]), static_cast<float>(s));
        }
        if (threadIdx.x == 0)
        {
            quant_ptr[row] = s;
        }
    }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeComputeScalesAndQuantizeMatrix(T_OUT* output, T_S* quant_ptr, const T_IN* input, const int64_t numel,
    const int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream)
{
    if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        dim3 grid(numel / lda);
        bool use_shmem = true;
        auto const shmem_size = lda * sizeof(T_IN);
        if (shmem_size >= (48 << 10))
        {
            cudaError_t ret = cudaFuncSetAttribute(dynamicQuantizeMatrixPerToken<T_OUT, T_S, T_IN>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            use_shmem = ret == cudaSuccess;
        }
        if (use_shmem)
        {
            // ensure the threadblock is as large as possible to increase occupancy
            dim3 block(std::min((lda + 31) / 32 * 32, static_cast<int64_t>(1024)));
            dynamicQuantizeMatrixPerToken<<<grid, block, shmem_size, stream>>>(output, quant_ptr, input, numel, lda);
        }
        else
        {
            dim3 block(CTA_SIZE);
            computeFP8QuantizeScale<QuantizeMode::PER_TOKEN><<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
            sync_check_cuda_error();
            invokeQuantizeMatrix(output, quant_ptr, input, numel, lda, quantize_mode, stream);
        }
    }
    else if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        dim3 block(CTA_SIZE);
        dim3 grid((lda + CTA_SIZE - 1) / CTA_SIZE);
        cudaMemsetAsync(quant_ptr, 0, lda * sizeof(T_S), stream);
        sync_check_cuda_error();
        computeFP8QuantizeScale<QuantizeMode::PER_CHANNEL><<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
        sync_check_cuda_error();
        invokeQuantizeMatrix(output, quant_ptr, input, numel, lda, quantize_mode, stream);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        dim3 block(1024);
        dim3 grid(1024);
        cudaMemsetAsync(quant_ptr, 0, sizeof(T_S), stream);
        sync_check_cuda_error();
        computeFP8QuantizeScale<QuantizeMode::PER_TENSOR><<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
        sync_check_cuda_error();
        invokeQuantizeMatrix(output, quant_ptr, input, numel, lda, quantize_mode, stream);
    }
    sync_check_cuda_error();
}

#define DEFINE_INVOKE_QUANTIZE_MATRIX(type_out, type_scale, type_in)                                                   \
    template void invokeQuantizeMatrix<type_out, type_scale, type_in>(type_out * output,                               \
        type_scale const* input_scale, type_in const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode,   \
        cudaStream_t stream);                                                                                          \
    template void invokeDequantizeMatrix<type_out, type_scale, type_in>(type_out * output,                             \
        type_scale const* input_scale, type_in const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode,   \
        cudaStream_t stream);                                                                                          \
    template void invokeComputeScalesAndQuantizeMatrix<type_out, type_scale, type_in>(type_out * output,               \
        type_scale * input_scale, type_in const* input, int64_t numel, int64_t lda, QuantizeMode quantize_mode,        \
        cudaStream_t stream);

#ifdef ENABLE_FP8
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float, float);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float, half);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, half, half);
DEFINE_INVOKE_QUANTIZE_MATRIX(half, half, __nv_fp8_e4m3);
DEFINE_INVOKE_QUANTIZE_MATRIX(float, float, __nv_fp8_e4m3);
DEFINE_INVOKE_QUANTIZE_MATRIX(half, float, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float, __nv_bfloat16);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3);
DEFINE_INVOKE_QUANTIZE_MATRIX(__nv_bfloat16, float, __nv_fp8_e4m3);
#endif
#endif

#endif // ENABLE_FP8

#ifdef ENABLE_FP8
inline __device__ __nv_bfloat16 max_abs_op(bf16_4_t v) {
    return cuda_max(cuda_max<__nv_bfloat16>(cuda_abs(v.x)), cuda_max<__nv_bfloat16>(cuda_abs(v.y)));
}

inline __device__ __nv_bfloat16 max_abs_op(bf16_8_t v) {
    return cuda_max<__nv_bfloat16>(max_abs_op(bf16_4_t{v.x, v.y}), max_abs_op(bf16_4_t{v.z, v.w}));
}

inline __device__ __nv_bfloat162 mul(__nv_bfloat162 v, __nv_bfloat16 scale) {
    return bf16hmul2(v, bf162bf162(scale));
}

inline __device__ bf16_4_t mul(bf16_4_t v, __nv_bfloat16 scale) {
    bf16_4_t n;
    n.x = mul(v.x, scale);
    n.y = mul(v.y, scale);
    return n;
}

inline __device__ bf16_8_t mul(bf16_8_t v, __nv_bfloat16 scale) {
    bf16_8_t n;
    n.x = mul(v.x, scale);
    n.y = mul(v.y, scale);
    n.z = mul(v.z, scale);
    n.w = mul(v.w, scale);
    return n;
}

inline __device__ void convert_to_fp8(fp8_4_t* v, const bf16_4_t u) {
    v[0] = fp8_4_t(u.x, u.y);
}

inline __device__ void convert_to_fp8(fp8_8_t* v, const bf16_8_t u) {
    v[0].x = fp8_2_t(u.x);
    v[0].y = fp8_2_t(u.y);
    v[0].z = fp8_2_t(u.z);
    v[0].w = fp8_2_t(u.w);
}

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

static __device__ __forceinline__ __nv_bfloat162 silu(const __nv_bfloat162& val)
{
    return make_bfloat162((__nv_bfloat16)((float)val.x / (1.0f + __expf((float)-val.x))),
                          (__nv_bfloat16)((float)val.y / (1.0f + __expf((float)-val.y))));
}

inline __device__ bf16_8_t act_and_mul(bf16_8_t v, bf16_8_t u) {
    bf16_8_t n;
    n.x = bf16hmul2(silu(v.x), u.x);
    n.y = bf16hmul2(silu(v.y), u.y);
    n.z = bf16hmul2(silu(v.z), u.z);
    n.w = bf16hmul2(silu(v.w), u.w);
    return n;
}


template<typename T_S, bool COL_MAJOR_SCALE, int ELEM_PER_THREAD>
__global__ void computeFP8Quantize128Kernel(__nv_fp8_e4m3*       fp8_output,
                                            T_S*                 quant_ptr,
                                            const __nv_bfloat16* weights,
                                            const int64_t        dim0,
                                            const int64_t        dim1,
                                            const int64_t        size) {
    const int64_t       global_idx     = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    using InputElem = typename packed_type<__nv_bfloat16, ELEM_PER_THREAD>::type;
    using OutputElem = typename packed_type<__nv_fp8_e4m3, ELEM_PER_THREAD>::type;
    auto weights_vec = reinterpret_cast<InputElem const*>(weights);
    auto output_vec = reinterpret_cast<OutputElem *>(fp8_output);

    if (global_idx * ELEM_PER_THREAD >= size) {
        return;
    }
    auto w8 = weights_vec[global_idx];
    float scale = cuda_max((float)1e-4, (float)max_abs_op(w8));
    static constexpr int THREADS_PER_ROW = 128 / ELEM_PER_THREAD;
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        scale = max(scale, __shfl_xor_sync(0xFFFFFFFF, scale, mask, THREADS_PER_ROW));
    }
    scale = scale / FP8_E4M3_MAX;
    w8 = mul(w8, (__nv_bfloat16)(1 / scale));
    convert_to_fp8(output_vec + global_idx, w8);
    if (threadIdx.x % THREADS_PER_ROW == 0) {
        if constexpr (COL_MAJOR_SCALE) {
            const int64_t now_idx = global_idx / THREADS_PER_ROW;
            const int64_t row_idx = now_idx / dim1;
            const int64_t col_idx = now_idx % dim1;
            quant_ptr[col_idx * dim0 + row_idx] = scale;
        } else {
            quant_ptr[global_idx / THREADS_PER_ROW] = scale;
        }
    }
}

void invokeComputeFP8Quantize128(__nv_fp8_e4m3*       fp8_output,
                                 float*               quant_ptr,
                                 const __nv_bfloat16* weights,
                                 const int64_t        dim0,
                                 const int64_t        dim1,
                                 const int64_t        size,
                                 bool                 col_major_scale,
                                 cudaStream_t         stream) {
    RTP_LLM_CHECK(dim1 % 128 == 0);
    static constexpr int ELEM_PER_THREAD = 8;
    const int num_per_grid = CTA_SIZE / (128 / ELEM_PER_THREAD);
    dim3      grid((size / 128 + num_per_grid - 1) / num_per_grid);
    dim3      block(CTA_SIZE);
    if (col_major_scale) {
        computeFP8Quantize128Kernel<float, true, ELEM_PER_THREAD><<<grid, block, 0, stream>>>(fp8_output, quant_ptr, weights, dim0, dim1 / 128, size);
    } else {
        computeFP8Quantize128Kernel<float, false, ELEM_PER_THREAD><<<grid, block, 0, stream>>>(fp8_output, quant_ptr, weights, dim0, dim1, size);
    }
}

template<typename T_S, bool COL_MAJOR_SCALE, int ELEM_PER_THREAD>
__global__ void computeFP8ActivationAndQuantizeKernel(__nv_fp8_e4m3*       fp8_output,
                                                      T_S*                 quant_ptr,
                                                      const __nv_bfloat16* gate_up_output,
                                                      const int64_t        dim0,
                                                      const int64_t        dim1) {
    const int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t size       = dim0 * dim1;
    const int64_t padded_dim0 = (dim0 + 63) / 64 * 64;
    if (global_idx * ELEM_PER_THREAD >= size) {
        return;
    }

    const int64_t row_idx = global_idx * ELEM_PER_THREAD / dim1;
    const int64_t col_idx = global_idx * ELEM_PER_THREAD % dim1;
    const int64_t row_stride = dim1 * 2;
    using InputElem = typename packed_type<__nv_bfloat16, ELEM_PER_THREAD>::type;
    using OutputElem = typename packed_type<__nv_fp8_e4m3, ELEM_PER_THREAD>::type;
    auto weights_vec = reinterpret_cast<InputElem const*>(gate_up_output);
    auto output_vec = reinterpret_cast<OutputElem *>(fp8_output);

    auto gate8 = weights_vec[(row_idx * row_stride + col_idx) / ELEM_PER_THREAD], 
         up8   = weights_vec[(row_idx * row_stride + col_idx + dim1) / ELEM_PER_THREAD];

    auto w8 = act_and_mul(gate8, up8);
    float scale = cuda_max((float)1e-4, (float)max_abs_op(w8));
    static constexpr int THREADS_PER_ROW = 128 / ELEM_PER_THREAD;
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        scale = max(scale, __shfl_xor_sync(0xFFFFFFFF, scale, mask, THREADS_PER_ROW));
    }
    scale = scale / FP8_E4M3_MAX;
    w8 = mul(w8, (__nv_bfloat16)(1 / scale));
    convert_to_fp8(output_vec + global_idx, w8);
    if (threadIdx.x % THREADS_PER_ROW == 0) {
        const int64_t dim = dim1 / 128;
        const int64_t now_idx = global_idx / THREADS_PER_ROW;
        const int64_t row_idx = now_idx / dim;
        const int64_t col_idx = now_idx % dim;
        quant_ptr[col_idx * padded_dim0 + row_idx] = scale;
    }
}

void computeFP8ActivationAndQuantize(__nv_fp8_e4m3*       fp8_output,
                                     float*               quant_ptr,
                                     const __nv_bfloat16* weights,
                                     const int64_t        dim0,
                                     const int64_t        dim1,
                                     cudaStream_t         stream) {
    RTP_LLM_CHECK(dim1 % 128 == 0);
    static constexpr int ELEM_PER_THREAD = 8;
    const int num_per_grid = CTA_SIZE / (128 / ELEM_PER_THREAD);
    const int size = dim0 * dim1;
    dim3      grid((size / 128 + num_per_grid - 1) / num_per_grid);
    dim3      block(CTA_SIZE);
    computeFP8ActivationAndQuantizeKernel<float, true, ELEM_PER_THREAD><<<grid, block, 0, stream>>>(fp8_output, quant_ptr, weights, dim0, dim1);
}

#endif // ENABLE_FP8
} // namespace common
} // namespace tensorrt_llm
