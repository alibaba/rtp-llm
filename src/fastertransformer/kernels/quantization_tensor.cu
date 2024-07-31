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

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/kernels/quantization_tensor.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/cuda/cuda_utils.h"
#endif
#if USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#endif

namespace fastertransformer
{
#if USING_ROCM
using namespace rocm;
#endif

__global__ void quantizedKernel(char4* dst, const float4* src, const int64_t sizeDiv4, const float* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        const float scale = __ldg(scalePtr);
        char4 tmp;
        const float4 floatTmp = __ldg(src + idx);
        tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
        tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
        tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
        tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
        dst[idx] = tmp;
    }
}

__global__ void quantizedKernel(char4* dst, const half2* src, const int64_t sizeDiv4, const float* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        const float scale = __ldg(scalePtr);
        char4 tmp;
        int srcId = idx << 1;

        const uint2 h2 = __ldg(reinterpret_cast<const uint2*>(src + srcId));

        const half2 half2Tmp = reinterpret_cast<const half2&>(h2.x);
        const half2 half2Tmp2 = reinterpret_cast<const half2&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
        dst[idx] = tmp;
    }
}

template <typename T>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize)
{
    FT_CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    if (maxGridSize == -1) {
        maxGridSize = numBlocks;
    }
    dim3 grid(std::min(numBlocks, maxGridSize));
    FT_CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (const float4*) src, size / 4, scalePtr);
    }
    else if (std::is_same_v<T, half>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (const half2*) src, size / 4, scalePtr);
    }
}

#define INSTANTIATE_INVOKE_QUANTIZATION(T)                                                                        \
template void invokeQuantization(                                                                                 \
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize);

INSTANTIATE_INVOKE_QUANTIZATION(float);
INSTANTIATE_INVOKE_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_QUANTIZATION(__nv_bfloat16);
#endif

template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift)
{
    const T* srcRow = src + blockIdx.x * numCols;
    int8_t* dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        T val = srcRow[i];
        if(IS_SMOOTHER){
            val = cuda_cast<T>(val / cuda_cast<T>(smoother[i]));
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[blockIdx.x] = rowMax / 127.f;
    }

    const float scaleOrigQuant = 127.f / rowMax;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        T val = srcRow[i];
        if(IS_SMOOTHER){
            val = val / cuda_cast<T>(smoother[i]);
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        dstRow[i] = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_token_quantization_shift(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    if(shift != nullptr){
        perTokenQuantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift);
    }
    else{
        perTokenQuantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_token_quantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_token_quantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }

}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T)                                                                   \
    template void invokePerTokenQuantization(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// int8 col quant ///////////////////////////////////////////////////////////////////////////////
template<typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColQuantization(int8_t*       dst,
                                   const T*      src,
                                   const int64_t numRows,
                                   const int64_t numCols,
                                   half*         scalePtr,
                                   const float*  smoother,
                                   const float*  shift,
                                   float*        dbgfp  = nullptr,
                                   int*          dbgint = nullptr) {
    uint32_t colIdx = blockIdx.x;
    const T* srcCol = src + colIdx;
    int8_t*  dstCol = dst + colIdx;

    T localMax = 1e-6f;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x) {
        T val = srcCol[rowIdx * numCols];
        if (IS_SMOOTHER) {
            val = cuda_cast<T>(val / cuda_cast<T>(smoother[rowIdx]));
        }
        if (IS_SHIFT) {
            val = cuda_cast<T>(val + cuda_cast<T>(shift[rowIdx]));
        }
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float colMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0) {
        scalePtr[colIdx] = cuda_cast<half>(colMax / 128.f);
    }

    const float scaleOrigQuant = 128.f / colMax;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x) {
        T val = srcCol[rowIdx * numCols];
        if (IS_SMOOTHER) {
            val = val / cuda_cast<T>(smoother[rowIdx]);
        }
        if (IS_SHIFT) {
            val = cuda_cast<T>(val + cuda_cast<T>(shift[rowIdx]));
        }
        dstCol[rowIdx * numCols] = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
    }
}

template<typename T, bool IS_SMOOTHER>
void dispatch_per_col_quantization_shift(int8_t*       dst,
                                         const T*      src,
                                         const int64_t numRows,
                                         const int64_t numCols,
                                         half*         scalePtr,
                                         const float*  smoother,
                                         const float*  shift,
                                         cudaStream_t  stream) {
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numCols);

    if (shift != nullptr) {
        perColQuantization<T, IS_SMOOTHER, true>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift);
    } else {
        perColQuantization<T, IS_SMOOTHER, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerColQuantizationInt8(int8_t*       dst,
                                  const T*      src,
                                  const int64_t numRows,
                                  const int64_t numCols,
                                  half*         scalePtr,
                                  const float*  smoother,
                                  const float*  shift,
                                  cudaStream_t  stream) {
    if (smoother != nullptr) {
        dispatch_per_col_quantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    } else {
        dispatch_per_col_quantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(T)                                                                \
    template void invokePerColQuantizationInt8(int8_t*       dst,                                                      \
                                               const T*      src,                                                      \
                                               const int64_t numRows,                                                  \
                                               const int64_t numCols,                                                  \
                                               half*         scalePtr,                                                 \
                                               const float*  smoother,                                                 \
                                               const float*  shift,                                                    \
                                               cudaStream_t  stream)

INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(float);
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(__nv_bfloat16);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// int8 col dequant /////////////////////////////////////////////////////////////////////////////
template<typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColDequantization(T*            dst,
                                     const int8_t* src,
                                     const int64_t numRows,
                                     const int64_t numCols,
                                     const half*   scalePtr,
                                     const float*  smoother,
                                     const float*  shift,
                                     float*        dbgfp  = nullptr,
                                     int*          dbgint = nullptr) {
    uint32_t      colIdx = blockIdx.x;
    const int8_t* srcRow = src + colIdx;
    T*            dstRow = dst + colIdx;

    float scaleOrigQuant = cuda_cast<float>(scalePtr[colIdx]);
    if (IS_SMOOTHER) {
        scaleOrigQuant = scaleOrigQuant * smoother[colIdx];
    }
    if (IS_SHIFT) {
        scaleOrigQuant = scaleOrigQuant - shift[colIdx];
    }

    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x) {
        uint8_t tmpi8 = srcRow[rowIdx * numCols];

        T val = cuda_cast<T>(cuda_cast<float>(tmpi8) * scaleOrigQuant);

        if (IS_SMOOTHER) {
            val = val * cuda_cast<T>(smoother[rowIdx]);
        }
        if (IS_SHIFT) {
            val = cuda_cast<T>(val - cuda_cast<T>(shift[rowIdx]));
        }

        dstRow[rowIdx * numCols] = val;
    }
}

template<typename T, bool IS_SMOOTHER>
void dispatch_per_col_dequantization_shift(T*            dst,
                                           const int8_t* src,
                                           const int64_t numRows,
                                           const int64_t numCols,
                                           half*         scalePtr,
                                           const float*  smoother,
                                           const float*  shift,
                                           cudaStream_t  stream) {
    // each block is responsible for a single col
    const dim3 block(512);
    const dim3 grid(numCols);

    if (shift != nullptr) {
        perColDequantization<T, IS_SMOOTHER, true>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift);
    } else {
        perColDequantization<T, IS_SMOOTHER, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerColDequantizationInt8(T*            dst,
                                    const int8_t* src,
                                    const int64_t numRows,
                                    const int64_t numCols,
                                    half*         scalePtr,
                                    const float*  smoother,
                                    const float*  shift,
                                    cudaStream_t  stream) {
    if (smoother != nullptr) {
        dispatch_per_col_dequantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    } else {
        dispatch_per_col_dequantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(T)                                                              \
    template void invokePerColDequantizationInt8(T*            dst,                                                    \
                                                 const int8_t* src,                                                    \
                                                 const int64_t numRows,                                                \
                                                 const int64_t numCols,                                                \
                                                 half*         scalePtr,                                               \
                                                 const float*  smoother,                                               \
                                                 const float*  shift,                                                  \
                                                 cudaStream_t  stream)
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(float);
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(__nv_bfloat16);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// int4 col quant ///////////////////////////////////////////////////////////////////////////////
template<typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColQuantization(char4*        dst,
                                   const T*      src,
                                   const int64_t numRows,
                                   const int64_t numCols,
                                   const int64_t numColsBlk,
                                   half*         scalePtr,
                                   const float*  smoother,
                                   const float*  shift,
                                   float*        dbgfp  = nullptr,
                                   int*          dbgint = nullptr) {
    uint8_t* pDst      = (uint8_t*)dst;
    uint32_t colBlkIdx = blockIdx.x;
    const T* srcCol    = src + colBlkIdx * numColsBlk;
    uint8_t* dstCol    = pDst + colBlkIdx * numColsBlk / 2;

    T localMax = 1e-6f;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x) {
        for (int colInBlkIdx = 0; colInBlkIdx < numColsBlk; colInBlkIdx++) {
            T val = srcCol[rowIdx * numCols + colInBlkIdx];
            if (IS_SMOOTHER) {
                val = cuda_cast<T>(val / cuda_cast<T>(smoother[colBlkIdx]));
            }
            if (IS_SHIFT) {
                val = cuda_cast<T>(val + cuda_cast<T>(shift[colBlkIdx]));
            }
            localMax = cuda_max(localMax, cuda_abs(val));
        }
    }
    const float colBlkMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0) {
        scalePtr[colBlkIdx] = colBlkMax / 8.0f;
    }

    const float scaleOrigQuant = 8.f / colBlkMax;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x) {
        // one loop process 2 cols of intput, and 1 col of uint8_t output
        for (int colInBlkIdx = 0; colInBlkIdx < numColsBlk / 2; colInBlkIdx++) {
            T vall = srcCol[rowIdx * numCols + colInBlkIdx * 2];
            T valh = srcCol[rowIdx * numCols + colInBlkIdx * 2 + 1];
            if (IS_SMOOTHER) {
                vall = vall / cuda_cast<T>(smoother[colBlkIdx]);
                valh = valh / cuda_cast<T>(smoother[colBlkIdx]);
            }
            if (IS_SHIFT) {
                vall = cuda_cast<T>(vall + cuda_cast<T>(shift[colBlkIdx]));
                valh = cuda_cast<T>(valh + cuda_cast<T>(shift[colBlkIdx]));
            }

            int8_t tmpi8l = cuda_cast<int8_t>(cuda_cast<float>(vall) * scaleOrigQuant);
            int8_t tmpi8h = cuda_cast<int8_t>(cuda_cast<float>(valh) * scaleOrigQuant);
            int8_t tmpi4l = tmpi8l & 0x0F;
            int8_t tmpi4h = tmpi8h & 0x0F;

            uint8_t tmpuint = tmpi4l;
            tmpuint         = tmpuint << 4;
            tmpuint         = tmpuint | tmpi4h;

            dstCol[rowIdx * numCols / 2 + colInBlkIdx] = tmpuint;
        }
    }
}

template<typename T, bool IS_SMOOTHER>
void dispatch_per_col_quantization_shift(char4*        dst,
                                         const T*      src,
                                         const int64_t numRows,
                                         const int64_t numCols,
                                         half*         scalePtr,
                                         const float*  smoother,
                                         const float*  shift,
                                         cudaStream_t  stream) {
    // each block is responsible for a block cols, share the same scale
    const int colBlk = 2;
    assert(colBlk % 2 == 0);
    assert(numCols % colBlk == 0);

    const dim3 block(512);
    const dim3 grid(numCols / colBlk);

    if (shift != nullptr) {
        perColQuantization<T, IS_SMOOTHER, true>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, shift);
    } else {
        perColQuantization<T, IS_SMOOTHER, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerColQuantizationInt4x2(int8_t*       dst,
                                    const T*      src,
                                    const int64_t numRows,
                                    const int64_t numCols,
                                    half*         scalePtr,
                                    const float*  smoother,
                                    const float*  shift,
                                    cudaStream_t  stream) {
    if (smoother != nullptr) {
        dispatch_per_col_quantization_shift<T, true>(
            (char4*)dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    } else {
        dispatch_per_col_quantization_shift<T, false>(
            (char4*)dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(T)                                                              \
    template void invokePerColQuantizationInt4x2(int8_t*       dst,                                                    \
                                                 const T*      src,                                                    \
                                                 const int64_t numRows,                                                \
                                                 const int64_t numCols,                                                \
                                                 half*         scalePtr,                                               \
                                                 const float*  smoother,                                               \
                                                 const float*  shift,                                                  \
                                                 cudaStream_t  stream)
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(float);
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(__nv_bfloat16);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// int4 col dequant /////////////////////////////////////////////////////////////////////////////
template<typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColDequantization(T*            dst,
                                     const char4*  src,
                                     const int64_t numRows,
                                     const int64_t numCols,
                                     const int64_t numColsBlk,
                                     const half*   scalePtr,
                                     const float*  smoother,
                                     const float*  shift,
                                     float*        dbgfp  = nullptr,
                                     int*          dbgint = nullptr) {
    const uint8_t* pSrc      = (const uint8_t*)src;
    uint32_t       colBlkIdx = blockIdx.x;

    float scaleOrigQuant = scalePtr[colBlkIdx];
    if (IS_SMOOTHER) {
        scaleOrigQuant = scaleOrigQuant * smoother[colBlkIdx];
    }
    if (IS_SHIFT) {
        scaleOrigQuant = scaleOrigQuant - shift[colBlkIdx];
    }

    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x) {
        // one loop process 1 col uint8 input, and 2 cols of output
        for (int colInBlkIdx = 0; colInBlkIdx < numColsBlk / 2; colInBlkIdx++) {
            uint8_t tmpu8 = pSrc[rowIdx * numCols / 2 + colBlkIdx * numColsBlk / 2 + colInBlkIdx];

            uint8_t tmpi4l = tmpu8 & 0x0F;
            uint8_t tmpi4h = (tmpu8 >> 4) & 0x0F;

            T vall = cuda_cast<T>(cuda_cast<float>(tmpi4l) * scaleOrigQuant);
            T valh = cuda_cast<T>(cuda_cast<float>(tmpi4h) * scaleOrigQuant);

            if (IS_SMOOTHER) {
                vall = vall * cuda_cast<T>(smoother[colBlkIdx]);
                valh = valh * cuda_cast<T>(smoother[colBlkIdx]);
            }
            if (IS_SHIFT) {
                vall = cuda_cast<T>(vall - cuda_cast<T>(shift[colBlkIdx]));
                valh = cuda_cast<T>(valh - cuda_cast<T>(shift[colBlkIdx]));
            }

            dst[rowIdx * numCols + colBlkIdx * numColsBlk + colInBlkIdx * 2 + 0] = valh;
            dst[rowIdx * numCols + colBlkIdx * numColsBlk + colInBlkIdx * 2 + 1] = vall;
        }
    }
}

template<typename T, bool IS_SMOOTHER>
void dispatch_per_col_dequantization_shift(T*            dst,
                                           const char4*  src,
                                           const int64_t numRows,
                                           const int64_t numCols,
                                           half*         scalePtr,
                                           const float*  smoother,
                                           const float*  shift,
                                           cudaStream_t  stream) {
    // each block is responsible for a block cols, share the same scale
    const int colBlk = 2;
    assert(colBlk % 2 == 0);
    assert(numCols % colBlk == 0);

    const dim3 block(512);
    const dim3 grid(numCols / colBlk);

    if (shift != nullptr) {
        perColDequantization<T, IS_SMOOTHER, true>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, shift);
    } else {
        perColDequantization<T, IS_SMOOTHER, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerColDequantizationInt4x2(T*            dst,
                                      const int8_t* src,
                                      const int64_t numRows,
                                      const int64_t numCols,
                                      half*         scalePtr,
                                      const float*  smoother,
                                      const float*  shift,
                                      cudaStream_t  stream) {
    if (smoother != nullptr) {
        dispatch_per_col_dequantization_shift<T, true>(
            dst, (char4*)src, numRows, numCols, scalePtr, smoother, shift, stream);
    } else {
        dispatch_per_col_dequantization_shift<T, false>(
            dst, (char4*)src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(T)                                                            \
    template void invokePerColDequantizationInt4x2(T*            dst,                                                  \
                                                   const int8_t* src,                                                  \
                                                   const int64_t numRows,                                              \
                                                   const int64_t numCols,                                              \
                                                   half*         scalePtr,                                             \
                                                   const float*  smoother,                                             \
                                                   const float*  shift,                                                \
                                                   cudaStream_t  stream)
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(float);
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(__nv_bfloat16);
#endif

}  // namespace fastertransformer
