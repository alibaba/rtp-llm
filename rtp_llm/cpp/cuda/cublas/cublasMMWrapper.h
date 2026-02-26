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

#pragma once

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "cublasAlgoMap.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>

namespace rtp_llm {

template<typename T>
CublasDataType getCublasDataType() {
    if (std::is_same<T, half>::value) {
        return HALF_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return BFLOAT16_DATATYPE;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return FLOAT_DATATYPE;
    } else {
        RTP_LLM_CHECK(false);
        return FLOAT_DATATYPE;
    }
}

class cublasMMWrapper {
protected:
    cublasHandle_t   cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;

    cudaDataType_t Atype_;
    cudaDataType_t Btype_;
    cudaDataType_t Ctype_;
    cudaDataType_t computeType_;

    cudaStream_t   stream_;
    cublasAlgoMap* cublas_algo_map_;
    std::mutex*    mutex_;

    IAllocator*                        allocator_        = nullptr;
    void*                              cublas_workspace_ = nullptr;
    bool                               deterministic_gemm_ = false;
    std::vector<void*>                 additional_cublas_workspaces_;
    std::unordered_map<void*, int32_t> cublas_workspces_map_;

    friend class cublasINT8MMWrapper;

    void _Int8Gemm(const int     m,
                   const int     n,
                   const int     k,
                   const int8_t* A,
                   const int     lda,
                   const int8_t* B,
                   const int     ldb,
                   void*         C,
                   const int     ldc,
                   const void*   alpha,
                   const int     mode,
                   const bool    per_column_scaling);

    void cublasLtGemm(cublasHandle_t          handle,
                      cublasOperation_t       transa,
                      cublasOperation_t       transb,
                      int                     m,
                      int                     n,
                      int                     k,
                      const void*             alpha, /* host or device pointer */
                      const void*             A,
                      const void*             A_scale,
                      cudaDataType            Atype,
                      int                     lda,
                      const void*             B,
                      const void*             B_scale,
                      cudaDataType            Btype,
                      int                     ldb,
                      const void*             beta, /* host or device pointer */
                      void*                   C,
                      cudaDataType            Ctype,
                      int                     ldc,
                      bool                    is_fp16_computeType,
                      cublasLtMatmulAlgo_info info,
                      bool                    findAlgo,
                      int                     math_sm_count,
                      int8_t                  fast_accum,
                      cudaStream_t            stream);

public:
    cublasMMWrapper(cublasHandle_t   cublas_handle_,
                    cublasLtHandle_t cublaslt_handle_,
                    cudaStream_t     stream,
                    cublasAlgoMap*   map,
                    std::mutex*      mu,
                    IAllocator*      allocator);

    virtual ~cublasMMWrapper();

    cublasMMWrapper(const cublasMMWrapper& wrapper);

    virtual void cublasVersionCheck() {
        return;
    };

    void setDeterministicGemm(bool enable) {
        deterministic_gemm_ = enable;
    }
    cublasStatus_t cublasLtMatmulWrapper(cublasLtHandle_t            lightHandle,
                                         cublasLtMatmulDesc_t        computeDesc,
                                         const void*                 alpha,
                                         const void*                 A,
                                         cublasLtMatrixLayout_t      Adesc,
                                         const void*                 B,
                                         cublasLtMatrixLayout_t      Bdesc,
                                         const void*                 beta,
                                         const void*                 C,
                                         cublasLtMatrixLayout_t      Cdesc,
                                         void*                       D,
                                         cublasLtMatrixLayout_t      Ddesc,
                                         const cublasLtMatmulAlgo_t* algo,
                                         void*                       workspace,
                                         size_t                      workspaceSizeInBytes,
                                         cudaStream_t                stream,
                                         bool                        findBest = true);

    std::pair<bool, cublasLtMatmulAlgo_t> findHeuristicAlgo(cublasLtHandle_t       lightHandle,
                                                            cublasLtMatmulDesc_t   computeDesc,
                                                            const void*            alpha,
                                                            const void*            A,
                                                            cublasLtMatrixLayout_t Adesc,
                                                            const void*            B,
                                                            cublasLtMatrixLayout_t Bdesc,
                                                            const void*            beta,
                                                            const void*            C,
                                                            cublasLtMatrixLayout_t Cdesc,
                                                            void*                  D,
                                                            cublasLtMatrixLayout_t Ddesc);

    // Selects the first heuristic candidate with splitK<=1 to guarantee deterministic
    // results. Aborts if no such candidate exists among the top-64 candidates.
    std::pair<bool, cublasLtMatmulAlgo_t> findDeterministicAlgo(cublasLtHandle_t       lightHandle,
                                                                 cublasLtMatmulDesc_t   computeDesc,
                                                                 cublasLtMatrixLayout_t Adesc,
                                                                 cublasLtMatrixLayout_t Bdesc,
                                                                 cublasLtMatrixLayout_t Cdesc,
                                                                 cublasLtMatrixLayout_t Ddesc);

    std::pair<bool, cublasLtMatmulAlgo_t> findBestAlgo(cublasLtHandle_t       lightHandle,
                                                       cublasLtMatmulDesc_t   computeDesc,
                                                       const void*            alpha,
                                                       const void*            A,
                                                       cublasLtMatrixLayout_t Adesc,
                                                       const void*            B,
                                                       cublasLtMatrixLayout_t Bdesc,
                                                       const void*            beta,
                                                       const void*            C,
                                                       cublasLtMatrixLayout_t Cdesc,
                                                       void*                  D,
                                                       cublasLtMatrixLayout_t Ddesc,
                                                       cudaStream_t           stream);

    using MatrixLayout = std::tuple<cudaDataType_t, cublasLtOrder_t, uint64_t, uint64_t>;
    using cache_idx_t  = std::tuple<cublasLtMatmulDesc_t, std::array<MatrixLayout, 4>>;
    std::map<cache_idx_t, cublasLtMatmulAlgo_t> algo_cache;

    MatrixLayout createMatrixLayout(cublasLtMatrixLayout_t Mdesc);

    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       A,
              cudaDataType_t    Atype,
              const int         lda,
              const void*       B,
              cudaDataType_t    Btype,
              const int         ldb,
              void*             C,
              cudaDataType_t    Ctype,
              const int         ldc,
              cudaDataType_t    computeType,
              float             f_alpha,
              float             f_beta,
              const float*      A_scale,
              const float*      B_scale,
              int               math_sm_count,
              int8_t            fast_accum,
              cudaStream_t      stream);

    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       A,
              const int         lda,
              const void*       B,
              const int         ldb,
              void*             C,
              const int         ldc,
              const float       f_alpha       = 1.0f,
              const float       f_beta        = 0.0f,
              int               math_sm_count = 0,
              cudaStream_t      stream        = 0);

    void Int8Gemm(const int     m,
                  const int     n,
                  const int     k,
                  const int8_t* A,
                  const int     lda,
                  const int8_t* B,
                  const int     ldb,
                  int8_t*       C,
                  const int     ldc,
                  const float*  alpha,
                  const bool    per_column_scaling = false);

    void Int8Gemm(const int     m,
                  const int     n,
                  const int     k,
                  const int8_t* A,
                  const int     lda,
                  const int8_t* B,
                  const int     ldb,
                  int32_t*      C,
                  const int     ldc);

    void setFP32GemmConfig();
    void setFP16GemmConfig();

    void setBF16GemmConfig();

    void setStream(cudaStream_t stream);

    void setGemmConfig(cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType);

#ifdef ENABLE_FP8
    void setFP8GemmConfig(cudaDataType_t outputType = CUDA_R_16F);
#endif
    CublasDataType getCublasDataType(cudaDataType_t data_type);

#if (CUDART_VERSION >= 11000)
    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       A,
              const int         lda,
              const void*       B,
              const int         ldb,
              const void*       bias,
              void*             C,
              const int         ldc);
#endif

    void stridedBatchedGemm(cublasOperation_t transa,
                            cublasOperation_t transb,
                            const int         m,
                            const int         n,
                            const int         k,
                            const void*       A,
                            const int         lda,
                            const int64_t     strideA,
                            const void*       B,
                            const int         ldb,
                            const int64_t     strideB,
                            void*             C,
                            const int         ldc,
                            const int64_t     strideC,
                            const int         batchCount,
                            const float       f_alpha = 1.0f,
                            const float       f_beta  = 0.0f);

    void batchedGemm(cublasOperation_t  transa,
                     cublasOperation_t  transb,
                     const int          m,
                     const int          n,
                     const int          k,
                     const void* const* A,
                     const int          lda,
                     const void* const* B,
                     const int          ldb,
                     void* const*       C,
                     const int          ldc,
                     const int          batch_count,
                     const float        f_alpha = 1.0f,
                     const float        f_beta  = 0.0f);

    bool isFuseBatchGemm(const int batch_count, const int m, const int k, const int n);
};

}  // namespace rtp_llm
