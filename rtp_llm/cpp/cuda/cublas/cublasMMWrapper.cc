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

#include "cublasMMWrapper.h"
#include "autil/Scope.h"
#include <algorithm>
#include <cuda.h>

namespace rtp_llm {
cublasMMWrapper::cublasMMWrapper(cublasHandle_t   cublas_handle,
                                 cublasLtHandle_t cublaslt_handle,
                                 cudaStream_t     stream,
                                 cublasAlgoMap*   cublas_algo_map,
                                 std::mutex*      mu,
                                 IAllocator*      allocator):
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle),
    stream_(stream),
    cublas_algo_map_(cublas_algo_map),
    mutex_(mu),
    allocator_(allocator) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        cublas_workspace_ = allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE);
    }
}

cublasMMWrapper::~cublasMMWrapper() {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mutex_ = nullptr;
    if (allocator_ != nullptr) {
        allocator_->free((void**)(&cublas_workspace_));
        for (size_t i = 0; i < additional_cublas_workspaces_.size(); i++) {
            allocator_->free((void**)(&additional_cublas_workspaces_[i]));
        }
        allocator_ = nullptr;
    }
}

cublasMMWrapper::cublasMMWrapper(const cublasMMWrapper& wrapper):
    cublas_handle_(wrapper.cublas_handle_),
    cublaslt_handle_(wrapper.cublaslt_handle_),
    stream_(wrapper.stream_),
    cublas_algo_map_(wrapper.cublas_algo_map_),
    mutex_(wrapper.mutex_),
    allocator_(wrapper.allocator_) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        cublas_workspace_ = allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE);
    }
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
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
                           float             f_alpha,
                           float             f_beta,
                           int               math_sm_count,
                           cudaStream_t      stream) {
    Gemm(transa,
         transb,
         m,
         n,
         k,
         A,
         Atype_,
         lda,
         B,
         Btype_,
         ldb,
         C,
         Ctype_,
         ldc,
         computeType_,
         f_alpha,
         f_beta,
         nullptr,
         nullptr,
         math_sm_count,
         0,
         stream);
}

void cublasMMWrapper::cublasLtGemm(cublasHandle_t          handle,
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
                                   cudaStream_t            stream) {
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cublasLtMatrixLayout_t Ddesc;
    cublasLtMatmulDesc_t   operationDesc;

    cudaDataType_t scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    } else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_value(
        cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    autil::ScopeGuard guard1([&]() { cublasLtMatrixLayoutDestroy(Adesc); });
    check_cuda_value(
        cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    autil::ScopeGuard guard2([&]() { cublasLtMatrixLayoutDestroy(Bdesc); });
    check_cuda_value(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));
    autil::ScopeGuard guard3([&]() { cublasLtMatrixLayoutDestroy(Cdesc); });
    check_cuda_value(cublasLtMatrixLayoutCreate(&Ddesc, Btype, m, n, ldc));
    autil::ScopeGuard guard4([&]() { cublasLtMatrixLayoutDestroy(Ddesc); });
#if (CUDART_VERSION >= 11000)
    check_cuda_value(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#else
    check_cuda_value(cublasLtMatmulDescCreate(&operationDesc, computeType));
#endif
    autil::ScopeGuard guard5([&]() { cublasLtMatmulDescDestroy(operationDesc); });

    if (math_sm_count > 0) {
        check_cuda_value(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count, sizeof(math_sm_count)));
    }

    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(int8_t)));
    if (A_scale != nullptr) {
        check_cuda_value(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_scale, sizeof(void*)));
    }
    if (B_scale != nullptr) {
        check_cuda_value(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_scale, sizeof(void*)));
    }

    cublasLtMatmulAlgo_t algo;
    void*                workSpace     = cublas_workspace_;
    uint64_t             workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (stream != stream_) {
        if (cublas_workspces_map_.count(stream) == 0) {
            void* additional_cublas_workspace = nullptr;
            additional_cublas_workspace = allocator_->reMalloc(additional_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
            additional_cublas_workspaces_.push_back(additional_cublas_workspace);
            cublas_workspces_map_[stream] = additional_cublas_workspaces_.size() - 1;
        }
        workSpace     = additional_cublas_workspaces_[cublas_workspces_map_[stream]];
        workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
        RTP_LLM_LOG_DEBUG("stream %d, idx %d", stream, cublas_workspces_map_[stream]);
    }
    if (findAlgo) {
        if (info.workspaceSize > workspaceSize) {
            findAlgo = 0;
        } else {
            check_cuda_value(cublasLtMatmulAlgoInit(
                cublaslt_handle_, computeType, scaleType, Atype, Btype, Ctype, Ctype, info.algoId, &algo));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme)));

#if (CUDART_VERSION >= 11000)
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages)));
#endif

#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId)));
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &(info.mma_shapeId), sizeof(info.mma_shapeId)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &(info.cga_shapeId), sizeof(info.cga_shapeId)));
            check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &(info.sche_mode), sizeof(info.sche_mode)));
#endif
        }
    }

    check_cuda_value(cublasLtMatmulWrapper(cublaslt_handle_,
                                           operationDesc,
                                           alpha,
                                           A,
                                           Adesc,
                                           B,
                                           Bdesc,
                                           beta,
                                           C,
                                           Cdesc,
                                           C,
                                           Cdesc,
                                           (findAlgo == 1 ? (&algo) : NULL),
                                           workSpace,
                                           workspaceSize,
                                           stream,
                                           /* find_best = */ false));

    check_cuda_error();
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
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
                           cudaStream_t      stream) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::lock_guard<std::mutex> lock(*mutex_);

    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);
    // TODO: default cublas libs
    bool is_fp16_computeType = computeType == CUDA_R_16F ? true : false;
    bool using_cublasLt      = (Atype == CUDA_R_16F || Atype == CUDA_R_8F_E4M3 || Atype == CUDA_R_16BF) ? true : false;

    int batch_count = 1;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(Atype));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype));
    if (findAlgo) {
        RTP_LLM_LOG_DEBUG("Using pre-tuned cublasLt algorithm");
        if (info.stages != -1) {
            using_cublasLt = true;
        } else {
            using_cublasLt = false;
        }
    } else {
        RTP_LLM_LOG_DEBUG("Fallback to default cublas algorithm");
    }
    RTP_LLM_LOG_DEBUG("using cublasLt: %d", using_cublasLt);

    try {
        if (using_cublasLt) {
            const void* A_scale_ptr = static_cast<const void*>(A_scale);
            const void* B_scale_ptr = static_cast<const void*>(B_scale);
            cublasLtGemm(cublas_handle_,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         A,
                         A_scale_ptr,
                         Atype,
                         lda,
                         B,
                         B_scale_ptr,
                         Btype,
                         ldb,
                         beta,
                         C,
                         Ctype,
                         ldc,
                         is_fp16_computeType,
                         info,
                         findAlgo,
                         math_sm_count,
                         fast_accum,
                         stream);
        } else {
            int cublasAlgo = info.algoId;
            check_cuda_value(cublasGemmEx(cublas_handle_,
                                          transa,
                                          transb,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          A,
                                          Atype,
                                          lda,
                                          B,
                                          Btype,
                                          ldb,
                                          beta,
                                          C,
                                          Ctype,
                                          ldc,
                                          computeType,
                                          static_cast<cublasGemmAlgo_t>(cublasAlgo)));
        }
        check_cuda_error();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("cublasMMWrapper::Gemm exception %s", e.what());
        throw;
    }
}

void cublasMMWrapper::setFP32GemmConfig() {
    Atype_       = CUDA_R_32F;
    Btype_       = CUDA_R_32F;
    Ctype_       = CUDA_R_32F;
    computeType_ = CUDA_R_32F;
}

void cublasMMWrapper::setFP16GemmConfig() {
    Atype_       = CUDA_R_16F;
    Btype_       = CUDA_R_16F;
    Ctype_       = CUDA_R_16F;
    computeType_ = CUDA_R_32F;
}

void cublasMMWrapper::setBF16GemmConfig() {
    Atype_       = CUDA_R_16BF;
    Btype_       = CUDA_R_16BF;
    Ctype_       = CUDA_R_16BF;
    computeType_ = CUDA_R_32F;
}

#ifdef ENABLE_FP8
void cublasMMWrapper::setFP8GemmConfig(cudaDataType_t outputType) {
    setGemmConfig(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, outputType, CUDA_R_32F);
}
#endif
void cublasMMWrapper::setGemmConfig(cudaDataType_t aType,
                                    cudaDataType_t bType,
                                    cudaDataType_t cType,
                                    cudaDataType_t computeType) {
    Atype_       = aType;
    Btype_       = bType;
    Ctype_       = cType;
    computeType_ = computeType;
}

CublasDataType cublasMMWrapper::getCublasDataType(cudaDataType_t data_type) {
    if (data_type == CUDA_R_16F) {
        return HALF_DATATYPE;
    } else if (data_type == CUDA_R_32F) {
        return FLOAT_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (data_type == CUDA_R_16BF) {
        return BFLOAT16_DATATYPE;
    }
#endif
    return FLOAT_DATATYPE;
}

#if (CUDART_VERSION >= 11000)
// input, weight, output are row-major
// only works for cublas 11.x
void cublasMMWrapper::Gemm(cublasOperation_t transa,
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
                           const int         ldc) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    cudaDataType_t      Atype, Btype, Ctype;
    cublasComputeType_t computeType;
    cudaDataType_t      scaleType;
    float               alpha_float = 1.0f;
    float               beta_float  = 0.0f;
    void *              alpha, *beta;

    // int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    if (Atype_ == CUDA_R_32F) {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        Atype       = CUDA_R_32F;
        Btype       = CUDA_R_32F;
        Ctype       = CUDA_R_32F;
        scaleType   = CUDA_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    } else if (Atype_ == CUDA_R_16BF) {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        Atype       = CUDA_R_16BF;
        Btype       = CUDA_R_16BF;
        Ctype       = CUDA_R_16BF;
        scaleType   = CUDA_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    } else {
        computeType = CUBLAS_COMPUTE_32F;
        Atype       = CUDA_R_16F;
        Btype       = CUDA_R_16F;
        Ctype       = CUDA_R_16F;
        scaleType   = CUDA_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    }

    int                     findAlgo = cublas_algo_map_->isExist(1, m, n, k, getCublasDataType(Atype_));
    cublasLtMatmulAlgo_info info     = cublas_algo_map_->getAlgo(1, m, n, k, getCublasDataType(Atype_));
    cublasLtMatmulAlgo_t    algo;

    void*    workSpace     = cublas_workspace_;
    uint64_t workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    if (findAlgo && info.stages != -1 && info.workspaceSize <= workspaceSize) {
        check_cuda_value(cublasLtMatmulAlgoInit(
            cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, info.algoId, &algo));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption)));
        check_cuda_value(
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile)));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val)));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle)));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme)));
#if (CUDART_VERSION >= 11000)
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages)));
#endif

#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId)));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId)));
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &(info.mma_shapeId), sizeof(info.mma_shapeId)));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &(info.cga_shapeId), sizeof(info.cga_shapeId)));
        check_cuda_value(cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &(info.sche_mode), sizeof(info.sche_mode)));
#endif
    } else {
        findAlgo = false;
    }

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtEpilogue_t     epi = CUBLASLT_EPILOGUE_BIAS;
    check_cuda_value(cublasLtMatrixLayoutCreate(
        &Adesc, Atype, (transa == CUBLAS_OP_N) ? m : k, (transa == CUBLAS_OP_N) ? k : m, lda));
    autil::ScopeGuard guard1([&]() { cublasLtMatrixLayoutDestroy(Adesc); });
    check_cuda_value(cublasLtMatrixLayoutCreate(
        &Bdesc, Btype, (transb == CUBLAS_OP_N) ? k : n, (transb == CUBLAS_OP_N) ? n : k, ldb));
    autil::ScopeGuard guard2([&]() { cublasLtMatrixLayoutDestroy(Bdesc); });
    check_cuda_value(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));
    autil::ScopeGuard guard3([&]() { cublasLtMatrixLayoutDestroy(Cdesc); });

    check_cuda_value(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
    autil::ScopeGuard guard4([&]() { cublasLtMatmulDescDestroy(operationDesc); });
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t)));
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(const void*)));
    check_cuda_value(cublasLtMatmul(cublaslt_handle_,
                                    operationDesc,
                                    alpha,
                                    A,
                                    Adesc,
                                    B,
                                    Bdesc,
                                    beta,
                                    C,
                                    Cdesc,
                                    C,
                                    Cdesc,
                                    (findAlgo == 1 ? (&algo) : NULL),
                                    workSpace,
                                    workspaceSize,
                                    stream_));
    check_cuda_error();
}
#endif
void cublasMMWrapper::setStream(cudaStream_t stream) {
    stream_ = stream;
}

void cublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa,
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
                                         const int         batch_count,
                                         const float       f_alpha,
                                         const float       f_beta) {
    std::lock_guard<std::mutex> lock(*mutex_);

    half h_alpha = (half)f_alpha;
    half h_beta  = (half)f_beta;

    int         is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha =
        is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_value(cublasGemmStridedBatchedEx(cublas_handle_,
                                                transa,
                                                transb,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                A,
                                                Atype_,
                                                lda,
                                                strideA,
                                                B,
                                                Btype_,
                                                ldb,
                                                strideB,
                                                beta,
                                                C,
                                                Ctype_,
                                                ldc,
                                                strideC,
                                                batch_count,
                                                computeType_,
                                                static_cast<cublasGemmAlgo_t>(info.algoId)));
}

void cublasMMWrapper::batchedGemm(cublasOperation_t  transa,
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
                                  const float        alpha,
                                  const float        beta) {
    std::lock_guard<std::mutex> lock(*mutex_);

    float f_alpha = static_cast<float>(alpha);
    float f_beta  = static_cast<float>(beta);

    half h_alpha = (half)alpha;
    half h_beta  = (half)beta;

    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;

    const void* r_alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* r_beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_value(cublasGemmBatchedEx(cublas_handle_,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         r_alpha,
                                         A,
                                         Atype_,
                                         lda,
                                         B,
                                         Btype_,
                                         ldb,
                                         r_beta,
                                         C,
                                         Ctype_,
                                         ldc,
                                         batch_count,
                                         computeType_,
                                         static_cast<cublasGemmAlgo_t>(info.algoId)));
}

bool cublasMMWrapper::isFuseBatchGemm(const int batch_count, const int m, const int k, const int n) {
    CublasDataType data_type = getCublasDataType(Atype_);

    if (cublas_algo_map_->isExist(batch_count, m, k, n, data_type) == false
        || cublas_algo_map_->isExist(1, m, k, n, data_type) == false) {
        return false;
    } else {
        return cublas_algo_map_->getAlgo(batch_count, m, k, n, data_type).exec_time
               < 3 * cublas_algo_map_->getAlgo(1, m, k, n, data_type).exec_time;
    }
}

std::pair<bool, cublasLtMatmulAlgo_t> cublasMMWrapper::findHeuristicAlgo(cublasLtHandle_t       lightHandle,
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
                                                                         cublasLtMatrixLayout_t Ddesc) {
#if (CUBLAS_VERSION) <= 11402
    RTP_LLM_FAIL("CUBLAS version too low.");
    return {false, cublasLtMatmulAlgo_t{}};
#else
    size_t  returnSize;
    int32_t pointer_mode;
    check_cuda_value(cublasLtMatmulDescGetAttribute(
        computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize));

    cublasLtMatmulHeuristicResult_t result;
    cublasLtMatmulPreference_t      preference;
    check_cuda_value(cublasLtMatmulPreferenceCreate(&preference));
    autil::ScopeGuard guard1([&]() { cublasLtMatmulPreferenceDestroy(preference); });
    check_cuda_value(cublasLtMatmulPreferenceInit(preference));
    uint64_t workspace_size = CUBLAS_WORKSPACE_SIZE;
    check_cuda_value(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
    uint32_t pointer_mode_mask = 0;
    check_cuda_value(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif

    int  return_count = 0;
    auto ret          = cublasLtMatmulAlgoGetHeuristic(
        lightHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &result, &return_count);
    check_cuda_value(ret);

    return {return_count != 0, result.algo};
#endif
}

std::pair<bool, cublasLtMatmulAlgo_t> cublasMMWrapper::findBestAlgo(cublasLtHandle_t       lightHandle,
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
                                                                    cudaStream_t           stream) {
#if (CUBLAS_VERSION) <= 11402
    RTP_LLM_FAIL("CUBLAS version too low.");
    return {false, cublasLtMatmulAlgo_t{}};
#else
    size_t  returnSize;
    int32_t pointer_mode;
    check_cuda_value(cublasLtMatmulDescGetAttribute(
        computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t                   preference;
    check_cuda_value(cublasLtMatmulPreferenceCreate(&preference));
    autil::ScopeGuard guard1([&]() { cublasLtMatmulPreferenceDestroy(preference); });
    check_cuda_value(cublasLtMatmulPreferenceInit(preference));
    uint64_t workspace_size = CUBLAS_WORKSPACE_SIZE;
    check_cuda_value(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
    uint32_t pointer_mode_mask = 0;
    check_cuda_value(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif

    int  return_count = 0;
    auto ret          = cublasLtMatmulAlgoGetHeuristic(lightHandle,
                                              computeDesc,
                                              Adesc,
                                              Bdesc,
                                              Cdesc,
                                              Ddesc,
                                              preference,
                                              heuristics.size(),
                                              heuristics.data(),
                                              &return_count);
    check_cuda_value(ret);
    heuristics.resize(return_count);

    std::map<int, std::vector<float>> algo_results;
    cudaEvent_t                       start_event, stop_event;
    check_cuda_value(cudaEventCreate(&start_event));
    autil::ScopeGuard guard4([&]() { cudaEventDestroy(start_event); });
    check_cuda_value(cudaEventCreate(&stop_event));
    autil::ScopeGuard guard5([&]() { cudaEventDestroy(stop_event); });

    for (int i = 0; i < int(heuristics.size()); i++) {
        const auto&          heuristic = heuristics[i];
        cublasLtMatmulAlgo_t algo      = heuristic.algo;
        int32_t              algo_id;
        check_cuda_value(cublasLtMatmulAlgoConfigGetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize));

        for (int i = 0; i < 11; i++) {
            float duration_ms;
            cudaEventRecord(start_event, stream);
            check_cuda_value(cublasLtMatmul(lightHandle,
                                            computeDesc,
                                            alpha,
                                            A,
                                            Adesc,
                                            B,
                                            Bdesc,
                                            beta,
                                            C,
                                            Cdesc,
                                            D,
                                            Ddesc,
                                            &algo,
                                            cublas_workspace_,
                                            CUBLAS_WORKSPACE_SIZE,
                                            stream));
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&duration_ms, start_event, stop_event);

            algo_results[algo_id].push_back(duration_ms);
        }
        std::sort(algo_results[algo_id].begin(), algo_results[algo_id].end());
    }

    cublasLtMatmulHeuristicResult_t result;
    float                           best_time = INFINITY;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t              algo_id;
        check_cuda_value(cublasLtMatmulAlgoConfigGetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize));
        const auto& results = algo_results[algo_id];

        if (results.size() > 0 && results[5] < best_time) {
            best_time = results[5];
            result    = heuristic;
        }
    }

    return {best_time != INFINITY, result.algo};
#endif
}

cublasMMWrapper::MatrixLayout cublasMMWrapper::createMatrixLayout(cublasLtMatrixLayout_t Mdesc) {
    size_t       returnSize;
    MatrixLayout m_layout;

    check_cuda_value(cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &std::get<0>(m_layout), sizeof(std::get<0>(m_layout)), &returnSize));
    check_cuda_value(cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &std::get<1>(m_layout), sizeof(std::get<1>(m_layout)), &returnSize));
    check_cuda_value(cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &std::get<2>(m_layout), sizeof(std::get<2>(m_layout)), &returnSize));
    check_cuda_value(cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &std::get<3>(m_layout), sizeof(std::get<3>(m_layout)), &returnSize));

    return m_layout;
}

cublasStatus_t cublasMMWrapper::cublasLtMatmulWrapper(cublasLtHandle_t            lightHandle,
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
                                                      bool                        findBest) {
    cache_idx_t cache_idx{
        computeDesc,
        {createMatrixLayout(Adesc), createMatrixLayout(Bdesc), createMatrixLayout(Cdesc), createMatrixLayout(Ddesc)}};

    cublasLtMatmulAlgo_t algo_value;
    bool                 found_algo = false;
    if (algo == nullptr) {
        auto it = algo_cache.find(cache_idx);
        if (it == algo_cache.end()) {
            std::pair<bool, cublasLtMatmulAlgo_t> result;
            if (findBest) {
                result =
                    findBestAlgo(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, stream);
            } else {
                result =
                    findHeuristicAlgo(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc);
            }
            if (result.first) {
                algo_cache[cache_idx] = result.second;
                algo_value            = result.second;
                found_algo            = true;
            }
        } else {
            algo_value = it->second;
            found_algo = true;
        }
    }

    return cublasLtMatmul(lightHandle,
                          computeDesc,
                          alpha,
                          A,
                          Adesc,
                          B,
                          Bdesc,
                          beta,
                          C,
                          Cdesc,
                          D,
                          Ddesc,
                          found_algo ? &algo_value : algo,
                          workspace,
                          workspaceSizeInBytes,
                          stream);
}

void cublasMMWrapper::_Int8Gemm(const int     m,
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
                                const bool    per_column_scaling) {
    /* mode:
     *  - 0: int8 * int8 -> int32 -> int8
     *  - 1: int8 * int8 -> int32 -> int32
     */
#if (CUBLAS_VERSION) <= 11402
    RTP_LLM_FAIL("CUBLAS version too low.");
#else

    std::lock_guard<std::mutex> lock(*mutex_);
    const auto                  op_a        = CUBLAS_OP_T;
    const auto                  op_b        = CUBLAS_OP_N;
    const auto                  dataType    = CUDA_R_8I;
    const auto                  resultType  = mode == 0 ? CUDA_R_8I : CUDA_R_32I;
    const auto                  computeType = CUBLAS_COMPUTE_32I;
    const auto                  scaleType   = mode == 0 ? CUDA_R_32F : CUDA_R_32I;
    const void*                 beta;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_value(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, lda));
    autil::ScopeGuard guard1([&]() { cublasLtMatrixLayoutDestroy(Adesc); });
    check_cuda_value(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, ldb));
    autil::ScopeGuard guard2([&]() { cublasLtMatrixLayoutDestroy(Bdesc); });
    check_cuda_value(cublasLtMatrixLayoutCreate(&Cdesc, resultType, m, n, ldc));
    autil::ScopeGuard guard3([&]() { cublasLtMatrixLayoutDestroy(Cdesc); });

    check_cuda_value(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
    autil::ScopeGuard guard4([&]() { cublasLtMatmulDescDestroy(operationDesc); });

    auto pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    if (mode == 0) {
        pointer_mode =
            per_column_scaling ? CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST : CUBLASLT_POINTER_MODE_DEVICE;
    }
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(cublasOperation_t)));
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(cublasOperation_t)));
    check_cuda_value(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &op_b, sizeof(cublasOperation_t)));
    check_cuda_value(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

    const int32_t int_one    = 1;
    const int32_t int_zero   = 0;
    const float   float_zero = 0;
    if (mode == 0) {
        beta = per_column_scaling ? &float_zero : NULL;
    } else {
        alpha = &int_one;
        beta  = &int_zero;
    }

    void*    workSpace     = cublas_workspace_;
    uint64_t workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    check_cuda_error();
    auto ret = cublasLtMatmulWrapper(cublaslt_handle_,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     NULL,
                                     workSpace,
                                     workspaceSize,
                                     stream_);
    check_cuda_value(ret);
    check_cuda_error();
#endif
}

void cublasMMWrapper::Int8Gemm(const int     m,
                               const int     n,
                               const int     k,
                               const int8_t* A,
                               const int     lda,
                               const int8_t* B,
                               const int     ldb,
                               int8_t*       C,
                               const int     ldc,
                               const float*  alpha,
                               const bool    per_column_scaling) {
    return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, alpha, 0, per_column_scaling);
}

void cublasMMWrapper::Int8Gemm(const int     m,
                               const int     n,
                               const int     k,
                               const int8_t* A,
                               const int     lda,
                               const int8_t* B,
                               const int     ldb,
                               int32_t*      C,
                               const int     ldc) {
    return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, (float*)nullptr, 1, false);
}

}  // namespace rtp_llm
