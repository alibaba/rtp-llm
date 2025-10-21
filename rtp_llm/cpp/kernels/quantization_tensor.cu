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

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/cpp/kernels/quantization_tensor.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

__global__ void quantizedKernel(char4* dst, const float4* src, const int64_t sizeDiv4, const float* scalePtr) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x) {
        const float  scale = __ldg(scalePtr);
        char4        tmp;
        const float4 floatTmp = __ldg(src + idx);
        tmp.x                 = cuda_cast<int8_t>(floatTmp.x * scale);
        tmp.y                 = cuda_cast<int8_t>(floatTmp.y * scale);
        tmp.z                 = cuda_cast<int8_t>(floatTmp.z * scale);
        tmp.w                 = cuda_cast<int8_t>(floatTmp.w * scale);
        dst[idx]              = tmp;
    }
}

__global__ void quantizedKernel(char4* dst, const half2* src, const int64_t sizeDiv4, const float* scalePtr) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x) {
        const float scale = __ldg(scalePtr);
        char4       tmp;
        int         srcId = idx << 1;

        const uint2 h2 = __ldg(reinterpret_cast<const uint2*>(src + srcId));

        const half2 half2Tmp  = reinterpret_cast<const half2&>(h2.x);
        const half2 half2Tmp2 = reinterpret_cast<const half2&>(h2.y);

        tmp.x    = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
        tmp.y    = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
        tmp.z    = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
        tmp.w    = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
        dst[idx] = tmp;
    }
}

template<typename T>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize) {
    RTP_LLM_CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    if (maxGridSize == -1) {
        maxGridSize = numBlocks;
    }
    dim3 grid(std::min(numBlocks, maxGridSize));
    RTP_LLM_CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>) {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*)dst, (const float4*)src, size / 4, scalePtr);
    } else if (std::is_same_v<T, half>) {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*)dst, (const half2*)src, size / 4, scalePtr);
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#define INSTANTIATE_INVOKE_QUANTIZATION(T)                                                                             \
    template void invokeQuantization(                                                                                  \
        int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize);

INSTANTIATE_INVOKE_QUANTIZATION(float);
INSTANTIATE_INVOKE_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_QUANTIZATION(__nv_bfloat16);
#endif

template<typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perTokenQuantization(int8_t*       dst,
                                     const T*      src,
                                     const int64_t numRows,
                                     const int64_t numCols,
                                     float*        scalePtr,
                                     const float*  smoother,
                                     const float*  shift) {
    const T* srcRow = src + blockIdx.x * numCols;
    int8_t*  dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x) {
        T val = srcRow[i];
        if (IS_SMOOTHER) {
            val = cuda_cast<T>(val / cuda_cast<T>(smoother[i]));
        }
        if (IS_SHIFT) {
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0) {
        scalePtr[blockIdx.x] = rowMax / 127.f;
    }

    const float scaleOrigQuant = 127.f / rowMax;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x) {
        T val = srcRow[i];
        if (IS_SMOOTHER) {
            val = val / cuda_cast<T>(smoother[i]);
        }
        if (IS_SHIFT) {
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        dstRow[i] = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
    }
}

template<typename T, bool IS_SMOOTHER>
void dispatch_per_token_quantization_shift(int8_t*       dst,
                                           const T*      src,
                                           const int64_t numRows,
                                           const int64_t numCols,
                                           float*        scalePtr,
                                           const float*  smoother,
                                           const float*  shift,
                                           cudaStream_t  stream) {
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    if (shift != nullptr) {
        perTokenQuantization<T, IS_SMOOTHER, true>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift);
    } else {
        perTokenQuantization<T, IS_SMOOTHER, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr);
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T>
void invokePerTokenQuantization(int8_t*       dst,
                                const T*      src,
                                const int64_t numRows,
                                const int64_t numCols,
                                float*        scalePtr,
                                const float*  smoother,
                                const float*  shift,
                                cudaStream_t  stream) {
    if (smoother != nullptr) {
        dispatch_per_token_quantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    } else {
        dispatch_per_token_quantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T)                                                                   \
    template void invokePerTokenQuantization(int8_t*       dst,                                                        \
                                             const T*      src,                                                        \
                                             const int64_t numRows,                                                    \
                                             const int64_t numCols,                                                    \
                                             float*        scalePtr,                                                   \
                                             const float*  smoother,                                                   \
                                             const float*  shift,                                                      \
                                             cudaStream_t  stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16);
#endif

#ifdef ENABLE_FP8
template<typename T>
__global__ void
perTokenBlockQuantization(int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr) {
    const T* srcRow = src + blockIdx.x * numCols;
    int8_t*  dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x) {
        T val    = srcRow[i];
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0) {
        scalePtr[blockIdx.x] = rowMax / 127.f;
    }

    const float scaleOrigQuant = 127.f / rowMax;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x) {
        T val     = srcRow[i];
        dstRow[i] = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
    }
}

template<typename T>
void invokePerTokenBlockQuantization(__nv_fp8_e4m3* dst,
                                     const T*       src,
                                     const int64_t  numRows,
                                     const int64_t  numCols,
                                     float*         scalePtr,
                                     cudaStream_t   stream) {
    const dim3 block(512);
    const dim3 grid(numRows);
}
#define INSTANTIATE_INVOKE_PER_TOKEN_BLOCK_QUANTIZATION(T)                                                             \
    template void invokePerTokenBlockQuantization(__nv_fp8_e4m3* dst,                                                  \
                                                  const T*       src,                                                  \
                                                  const int64_t  numRows,                                              \
                                                  const int64_t  numCols,                                              \
                                                  float*         scalePtr,                                             \
                                                  cudaStream_t   stream)

#endif

}  // namespace rtp_llm
