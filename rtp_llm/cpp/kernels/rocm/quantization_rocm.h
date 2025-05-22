#pragma once

#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"

namespace rtp_llm {
template<typename T>
void invokePerColQuantizationInt4x2(const T*      src,
                                    const int64_t numRows,
                                    const int64_t numCols,
                                    const int64_t groupSize,
                                    uint8_t*      weightPtr,
                                    half*         scalePtr,
                                    half*         zerosPtr,
                                    cudaStream_t  stream = 0);

template<typename T>
void invokePerColDequantizationInt4x2(T*            dst,
                                      const int64_t numRows,
                                      const int64_t numCols,
                                      const int64_t groupSize,
                                      const int8_t* weightPtr,
                                      half*         scalePtr,
                                      half*         zerosPtr,
                                      cudaStream_t  stream = 0);

template<typename T>
void invokePerTokenQuantization(int8_t*       dst,
                                const T*      src,
                                const int64_t numRows,
                                const int64_t numCols,
                                float*        scalePtr,
                                const float*  smoother,
                                const float*  shift,
                                cudaStream_t  stream = 0);

template<typename T>
void invokePerColQuantizationInt8(int8_t*       dst,
                                  const T*      src,
                                  const int64_t numRows,
                                  const int64_t numCols,
                                  half*         scalePtr,
                                  const float*  smoother,
                                  const float*  shift,
                                  cudaStream_t  stream = 0);

template<typename T>
void invokePerColDequantizationInt8(T*            dst,
                                    const int8_t* src,
                                    const int64_t numRows,
                                    const int64_t numCols,
                                    half*         scalePtr,
                                    const float*  smoother,
                                    const float*  shift,
                                    cudaStream_t  stream = 0);

template<typename T>
void invokePerRowDequantizationInt4x2(T*            dst,
                                      const int8_t* src,
                                      const int64_t numRows,
                                      const int64_t numCols,
                                      half*         scalePtr,
                                      half*         zerosPtr,
                                      const int64_t groupSize,
                                      cudaStream_t  stream = 0);
}  // namespace rtp_llm
