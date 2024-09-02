#include "hipblasMMWrapper.h"

namespace fastertransformer {
namespace rocm {
#define HIPBLAS_WORKSPACE_SIZE 33554432  // 32MB
hipblasMMWrapper::hipblasMMWrapper(hipblasHandle_t   hipblas_handle,
                                   hipblasLtHandle_t hipblaslt_handle,
                                   hipStream_t       stream,
                                   hipblasAlgoMap*   hipblas_algo_map,
                                   std::mutex*       mu,
                                   IAllocator*       allocator):
    hipblas_handle_(hipblas_handle),
    hipblaslt_handle_(hipblaslt_handle),
    stream_(stream),
    hipblas_algo_map_(hipblas_algo_map),
    mu_(mu),
    allocator_(allocator) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        // FIXME: reMalloc NOT supported yet
        // hipblas_workspace_ = allocator_->reMalloc(hipblas_workspace_, HIPBLAS_WORKSPACE_SIZE);
        hipblas_workspace_ = allocator_->malloc(HIPBLAS_WORKSPACE_SIZE);
    }
}
hipblasMMWrapper::hipblasMMWrapper(const hipblasMMWrapper& wrapper):
    hipblas_handle_(wrapper.hipblas_handle_),
    hipblaslt_handle_(wrapper.hipblaslt_handle_),
    stream_(wrapper.stream_),
    hipblas_algo_map_(wrapper.hipblas_algo_map_),
    mu_(wrapper.mu_),
    allocator_(wrapper.allocator_) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        // FIXME: reMalloc NOT supported yet
        // hipblas_workspace_ = allocator_->reMalloc(hipblas_workspace_, HIPBLAS_WORKSPACE_SIZE);
        hipblas_workspace_ = allocator_->malloc(HIPBLAS_WORKSPACE_SIZE);
    }
}

hipblasMMWrapper::~hipblasMMWrapper() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_ = nullptr;
    if (allocator_ != nullptr) {
        allocator_->free((void**)(&hipblas_workspace_));
        allocator_ = nullptr;
    }
}

hipDataType hipblasMMWrapper::getHipDataType(hipblasDatatype_t data_type) {
    if (data_type == HIPBLAS_R_16F) {
        return HIP_R_16F;
    } else if (data_type == HIPBLAS_R_32F) {
        return HIP_R_32F;
    }
#ifdef ENABLE_BF16
    else if (data_type == HIPBLAS_R_16B) {
        return HIP_R_16BF;
    }
#endif
    return HIP_R_32F;
}
hipblasComputeType_t hipblasMMWrapper::getHipblasLtComputeType(hipblasDatatype_t data_type) {
    if (data_type == HIPBLAS_R_16F) {
        return HIPBLAS_COMPUTE_16F;
    } else if (data_type == HIPBLAS_R_32F) {
        return HIPBLAS_COMPUTE_32F;
    }
    return HIPBLAS_COMPUTE_32F;
}

void hipblasMMWrapper::setStream(hipStream_t stream) {
    stream_ = stream;
}

void hipblasMMWrapper::setFP32GemmConfig() {
    Atype_       = HIPBLAS_R_32F;
    Btype_       = HIPBLAS_R_32F;
    Ctype_       = HIPBLAS_R_32F;
    computeType_ = HIPBLAS_R_32F;
}
void hipblasMMWrapper::setFP16GemmConfig() {
    Atype_       = HIPBLAS_R_16F;
    Btype_       = HIPBLAS_R_16F;
    Ctype_       = HIPBLAS_R_16F;
    computeType_ = HIPBLAS_R_32F;
}
#ifdef ENABLE_BF16
void hipblasMMWrapper::setBF16GemmConfig() {
    Atype_       = HIPBLAS_R_16B;
    Btype_       = HIPBLAS_R_16B;
    Ctype_       = HIPBLAS_R_16B;
    computeType_ = HIPBLAS_R_32F;
}
#endif
void hipblasMMWrapper::setGemmConfig(hipblasDatatype_t aType,
                                     hipblasDatatype_t bType,
                                     hipblasDatatype_t cType,
                                     hipblasDatatype_t computeType) {
    Atype_       = aType;
    Btype_       = bType;
    Ctype_       = cType;
    computeType_ = computeType;
}

// =========================================== gemm =================================================
void hipblasMMWrapper::Gemm(hipblasOperation_t transa,
                            hipblasOperation_t transb,
                            const int          m,
                            const int          n,
                            const int          k,
                            const void*        alpha,
                            const void*        A,
                            hipblasDatatype_t  Atype,
                            int                lda,
                            const void*        B,
                            hipblasDatatype_t  Btype,
                            int                ldb,
                            const void*        beta,
                            void*              C,
                            hipblasDatatype_t  Ctype,
                            int                ldc,
                            hipblasDatatype_t  computeType,
                            hipblasGemmAlgo_t  algo) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_->lock();
    check_hip_error(hipblasGemmEx(hipblas_handle_,
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
                                  algo));
    mu_->unlock();
}

void hipblasMMWrapper::Gemm(hipblasOperation_t transa,
                            hipblasOperation_t transb,
                            const int          m,
                            const int          n,
                            const int          k,
                            const void*        A,
                            const int          lda,
                            const void*        B,
                            const int          ldb,
                            void*              C,
                            const int          ldc) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f);
}

void hipblasMMWrapper::Gemm(hipblasOperation_t transa,
                            hipblasOperation_t transb,
                            const int          m,
                            const int          n,
                            const int          k,
                            const void*        A,
                            const int          lda,
                            const void*        B,
                            const int          ldb,
                            void*              C,
                            const int          ldc,
                            float              f_alpha,
                            float              f_beta) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);

    mu_->lock();

    int  is_fp16_computeType = computeType_ == HIPBLAS_R_16F ? 1 : 0;
    bool using_hipblasLt     = (Atype_ == HIPBLAS_R_16F) ? true : false;
    int  batch_count         = 1;

    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    int findAlgo = hipblas_algo_map_->isExist(batch_count, m, n, k, Atype_);
    hipblasLtMatmulAlgo_info info = hipblas_algo_map_->getAlgo(batch_count, m, n, k, Atype_);

    if (findAlgo) {
        using_hipblasLt = true;
    } else {
        using_hipblasLt = false;
    }

    if (using_hipblasLt) {
        hipblasLtMatmulDesc_t   operationDesc = NULL;
        hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        hipDataType             scaleType   = HIP_R_32F;
        hipblasComputeType_t    computeType = HIPBLAS_COMPUTE_32F;

        // --------------------------------------
        // Create descriptors for the original matrices
        hipblasLtMatrixLayoutCreate(
            &Adesc, getHipDataType(Atype_), transa == HIPBLAS_OP_N ? m : k, transa == HIPBLAS_OP_N ? k : m, lda);
        hipblasLtMatrixLayoutCreate(
            &Bdesc, getHipDataType(Btype_), transb == HIPBLAS_OP_N ? k : n, transb == HIPBLAS_OP_N ? n : k, ldb);
        hipblasLtMatrixLayoutCreate(&Cdesc, getHipDataType(Ctype_), m, n, ldc);
        hipblasLtMatmulDescCreate(&operationDesc, computeType, scaleType);

        hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(int32_t));

        hipblasLtMatmulAlgo_t algo;
        void*                 workSpace     = hipblas_workspace_;
        int                   workspaceSize = hipblas_workspace_ == NULL ? 0 : HIPBLAS_WORKSPACE_SIZE;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(1);
        if (findAlgo) {
            if (info.workspaceSize > workspaceSize) {
                findAlgo = 0;
            } else {
                std::vector<int> algoIndex(1);
                algoIndex[0] = info.algoId;
                hipblaslt_ext::getAlgosFromIndex(hipblaslt_handle_, algoIndex, heuristicResult);
            }
        }

        hipblasLtMatmul(hipblaslt_handle_,
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
                        (findAlgo == 1 ? (&heuristicResult[0].algo) : NULL),
                        workSpace,
                        workspaceSize,
                        stream_);

        hipblasLtMatmulDescDestroy(operationDesc);
        hipblasLtMatrixLayoutDestroy(Adesc);
        hipblasLtMatrixLayoutDestroy(Bdesc);
        hipblasLtMatrixLayoutDestroy(Cdesc);
    } else {
        int hipblasAlgo = info.algoId;
        check_hip_error(hipblasGemmEx(hipblas_handle_,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      Atype_,
                                      lda,
                                      B,
                                      Btype_,
                                      ldb,
                                      beta,
                                      C,
                                      Ctype_,
                                      ldc,
                                      computeType_,
                                      static_cast<hipblasGemmAlgo_t>(hipblasAlgo)));
    }
    mu_->unlock();
}

void hipblasMMWrapper::Gemm(hipblasOperation_t transa,
                            hipblasOperation_t transb,
                            const int          m,
                            const int          n,
                            const int          k,
                            const void*        A,
                            const int          lda,
                            const void*        B,
                            const int          ldb,
                            const void*        bias,
                            void*              C,
                            const int          ldc) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    hipblasDatatype_t Atype, Btype, Ctype;
    hipblasDatatype_t computeType;
    hipblasDatatype_t scaleType;
    float             alpha_float = 1.0f;
    float             beta_float  = 0.0f;
    half              alpha_half  = half(1.0f);
    half              beta_half   = half(0.0f);
    void *            alpha, *beta;

    // int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    if (Atype_ == HIPBLAS_R_32F) {
        computeType = HIPBLAS_C_32F;
        Atype       = HIPBLAS_R_32F;
        Btype       = HIPBLAS_R_32F;
        Ctype       = HIPBLAS_R_32F;
        scaleType   = HIPBLAS_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    } else if (Atype_ == HIPBLAS_R_16B) {
        computeType = HIPBLAS_C_32F;
        Atype       = HIPBLAS_R_16B;
        Btype       = HIPBLAS_R_16B;
        Ctype       = HIPBLAS_R_16B;
        scaleType   = HIPBLAS_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    } else {
        computeType = HIPBLAS_C_16F;
        Atype       = HIPBLAS_R_16F;
        Btype       = HIPBLAS_R_16F;
        Ctype       = HIPBLAS_R_16F;
        scaleType   = HIPBLAS_R_16F;
        alpha       = &alpha_half;
        beta        = &beta_half;
    }

    hipblasLtMatmulDesc_t   operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    hipblasLtEpilogue_t     epi = HIPBLASLT_EPILOGUE_BIAS;
    hipblasLtMatrixLayoutCreate(
        &Adesc, getHipDataType(Atype), (transa == HIPBLAS_OP_N) ? m : k, (transa == HIPBLAS_OP_N) ? k : m, lda);
    hipblasLtMatrixLayoutCreate(
        &Bdesc, getHipDataType(Btype), (transb == HIPBLAS_OP_N) ? k : n, (transb == HIPBLAS_OP_N) ? n : k, ldb);
    hipblasLtMatrixLayoutCreate(&Cdesc, getHipDataType(Ctype), m, n, ldc);

    hipblasLtMatmulDescCreate(&operationDesc, getHipblasLtComputeType(computeType), getHipDataType(scaleType));
    hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(hipblasOperation_t));
    hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(hipblasOperation_t));
    hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(hipblasLtEpilogue_t));
    hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(const void*));
    check_hip_error(hipblasLtMatmul(
        hipblaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, NULL, 0, stream_));
    hipblasLtMatrixLayoutDestroy(Adesc);
    hipblasLtMatrixLayoutDestroy(Bdesc);
    hipblasLtMatrixLayoutDestroy(Cdesc);
    hipblasLtMatmulDescDestroy(operationDesc);
}

// =========================================== Batch =================================================
void hipblasMMWrapper::stridedBatchedGemm(hipblasOperation_t transa,
                                          hipblasOperation_t transb,
                                          const int          m,
                                          const int          n,
                                          const int          k,
                                          const void*        A,
                                          const int          lda,
                                          const int64_t      strideA,
                                          const void*        B,
                                          const int          ldb,
                                          const int64_t      strideB,
                                          void*              C,
                                          const int          ldc,
                                          const int64_t      strideC,
                                          const int          batch_count,
                                          const float        f_alpha,
                                          const float        f_beta) {
    half h_alpha = (half)f_alpha;
    half h_beta  = (half)f_beta;

    mu_->lock();
    int         is_fp16_computeType = computeType_ == HIPBLAS_R_16F ? 1 : 0;
    const void* alpha =
        is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    hipblasLtMatmulAlgo_info info = hipblas_algo_map_->getAlgo(batch_count, m, n, k, Atype_);

    check_hip_error(hipblasGemmStridedBatchedEx(hipblas_handle_,
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
                                                static_cast<hipblasGemmAlgo_t>(info.algoId)));

    mu_->unlock();
}

void hipblasMMWrapper::stridedBatchedGemm(hipblasOperation_t transa,
                                          hipblasOperation_t transb,
                                          const int          m,
                                          const int          n,
                                          const int          k,
                                          const float        f_alpha,
                                          const void*        A,
                                          hipblasDatatype_t  AType,
                                          const int          lda,
                                          const int64_t      strideA,
                                          const void*        B,
                                          hipblasDatatype_t  BType,
                                          const int          ldb,
                                          const int64_t      strideB,
                                          const float        f_beta,
                                          void*              C,
                                          hipblasDatatype_t  CType,
                                          const int          ldc,
                                          const int64_t      strideC,
                                          const int          batch_count,
                                          hipblasDatatype_t  computeType) {
    half h_alpha = (half)f_alpha;
    half h_beta  = (half)f_beta;

    mu_->lock();
    int         is_fp16_computeType = computeType == HIPBLAS_R_16F ? 1 : 0;
    const void* alpha =
        is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    hipblasLtMatmulAlgo_info info = hipblas_algo_map_->getAlgo(batch_count, m, n, k, Atype_);

    check_hip_error(hipblasGemmStridedBatchedEx(hipblas_handle_,
                                                transa,
                                                transb,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                A,
                                                AType,
                                                lda,
                                                strideA,
                                                B,
                                                BType,
                                                ldb,
                                                strideB,
                                                beta,
                                                C,
                                                CType,
                                                ldc,
                                                strideC,
                                                batch_count,
                                                computeType,
                                                static_cast<hipblasGemmAlgo_t>(info.algoId)));

    mu_->unlock();
}

void hipblasMMWrapper::batchedGemm(hipblasOperation_t transa,
                                   hipblasOperation_t transb,
                                   const int          m,
                                   const int          n,
                                   const int          k,
                                   const void* const* A,
                                   const int          lda,
                                   const void* const* B,
                                   const int          ldb,
                                   void* const*       C,
                                   const int          ldc,
                                   const int          batch_count) {
    float f_alpha = static_cast<float>(1.0f);
    float f_beta  = static_cast<float>(0.0f);

    half h_alpha = (half)1.0f;
    half h_beta  = (half)0.0f;

    mu_->lock();
    int         is_fp16_computeType = computeType_ == HIPBLAS_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    hipblasLtMatmulAlgo_info info = hipblas_algo_map_->getAlgo(batch_count, m, n, k, Atype_);

    check_hip_error(hipblasGemmBatchedEx(hipblas_handle_,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         (const void**)A,
                                         Atype_,
                                         lda,
                                         (const void**)B,
                                         Btype_,
                                         ldb,
                                         beta,
                                         (void**)C,
                                         Ctype_,
                                         ldc,
                                         batch_count,
                                         computeType_,
                                         static_cast<hipblasGemmAlgo_t>(info.algoId)));
    mu_->unlock();
}

// =========================================== int8 =================================================
// TODO: int8 NOT supported yet
#if (FALSE)
std::pair<bool, cublasLtMatmulAlgo_t> hipblasMMWrapper::findBestAlgo(cublasLtHandle_t       lightHandle,
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
                                                                     hipStream_t            stream) {

    return {false, cublasLtMatmulAlgo_t{}};
#if (FALSE)
#if (CUBLAS_VERSION) < 11601
    FT_CHECK_WITH_INFO(false, "CUBLAS version too low.");
    return {false, cublasLtMatmulAlgo_t{}};
#else
    size_t  returnSize;
    int32_t pointer_mode;
    cublasLtMatmulDescGetAttribute(
        computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize);

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t                   preference;
    check_hip_error(cublasLtMatmulPreferenceCreate(&preference));
    check_hip_error(cublasLtMatmulPreferenceInit(preference));
    uint64_t workspace_size = HIPBLAS_WORKSPACE_SIZE;
    check_hip_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
    uint32_t pointer_mode_mask = 0;
    check_hip_error(cublasLtMatmulPreferenceSetAttribute(
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
    heuristics.resize(return_count);

    std::map<int, std::vector<float>> algo_results;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t              algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);

        hipEvent_t start_event, stop_event;
        hipEventCreate(&start_event);
        hipEventCreate(&stop_event);

        float my_alpha = 1.0f;
        float my_beta  = 0.0f;

        for (int i = 0; i < 11; i++) {
            float duration_ms;
            hipEventRecord(start_event, stream);
            check_hip_error(cublasLtMatmul(lightHandle,
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
                                           hipblas_workspace_,
                                           HIPBLAS_WORKSPACE_SIZE,
                                           stream));
            hipEventRecord(stop_event, stream);
            hipEventSynchronize(stop_event);
            hipEventElapsedTime(&duration_ms, start_event, stop_event);

            algo_results[algo_id].push_back(duration_ms);
        }
        std::sort(algo_results[algo_id].begin(), algo_results[algo_id].end());
    }

    cublasLtMatmulHeuristicResult_t result;
    float                           best_time = INFINITY;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t              algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);
        const auto& results = algo_results[algo_id];

        if (results.size() > 0 && results[5] < best_time) {
            best_time = results[5];
            result    = heuristic;
        }
    }

    return {best_time != INFINITY, result.algo};
#endif
#endif
}
hipblasMMWrapper::MatrixLayout hipblasMMWrapper::createMatrixLayout(hipblasLtMatrixLayout_t Mdesc) {
    size_t       returnSize;
    MatrixLayout m_layout;

    hipblasLtMatrixLayoutGetAttribute(
        Mdesc, HIPBLASLT_MATRIX_LAYOUT_TYPE, &std::get<0>(m_layout), sizeof(std::get<0>(m_layout)), &returnSize);
    hipblasLtMatrixLayoutGetAttribute(
        Mdesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &std::get<1>(m_layout), sizeof(std::get<1>(m_layout)), &returnSize);
    hipblasLtMatrixLayoutGetAttribute(
        Mdesc, HIPBLASLT_MATRIX_LAYOUT_ROWS, &std::get<2>(m_layout), sizeof(std::get<2>(m_layout)), &returnSize);
    hipblasLtMatrixLayoutGetAttribute(
        Mdesc, HIPBLASLT_MATRIX_LAYOUT_COLS, &std::get<3>(m_layout), sizeof(std::get<3>(m_layout)), &returnSize);

    return m_layout;
}
hipblasStatus_t hipblasMMWrapper::cublasLtMatmulWrapper(cublasLtHandle_t            lightHandle,
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
                                                        hipStream_t                 stream) {
    return HIPBLAS_STATUS_SUCCESS;
#if (FALSE)
    cache_idx_t cache_idx{
        computeDesc,
        {createMatrixLayout(Adesc), createMatrixLayout(Bdesc), createMatrixLayout(Cdesc), createMatrixLayout(Ddesc)}};

    cublasLtMatmulAlgo_t algo_value;
    bool                 found_algo = false;
    if (algo == nullptr) {
        if (algo_cache.find(cache_idx) == algo_cache.end()) {
            auto result =
                findBestAlgo(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, stream);
            if (result.first) {
                algo_cache[cache_idx] = result.second;
                algo_value            = result.second;
                found_algo            = true;
            }
        } else {
            algo_value = algo_cache[cache_idx];
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
#endif
}

void hipblasMMWrapper::_Int8Gemm(const int     m,
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
    // TODO : in8 NOT supported.
    /* mode:
     *  - 0: int8 * int8 -> int32 -> int8
     *  - 1: int8 * int8 -> int32 -> int32
     */
#if (CUBLAS_VERSION) < 11601
    FT_CHECK_WITH_INFO(false, "CUBLAS version too low.");
#else

    mu_->lock();
    const auto  op_a        = HIPBLAS_OP_T;
    const auto  op_b        = HIPBLAS_OP_N;
    const auto  dataType    = HIPBLAS_R_8I;
    const auto  resultType  = mode == 0 ? HIPBLAS_R_8I : HIPBLAS_R_32I;
    const auto  computeType = CUBLAS_COMPUTE_32I;
    const auto  scaleType   = mode == 0 ? HIPBLAS_R_32F : HIPBLAS_R_32I;
    const int   batch_count = 1;
    const void* beta;

    int findAlgo = hipblas_algo_map_->isExist(batch_count, m, n, k, dataType);

    hipblasLtMatmulAlgo_info info = hipblas_algo_map_->getAlgo(batch_count, m, n, k, dataType);

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    // --------------------------------------
    // Create descriptors for the original matrices
    check_hip_error(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, lda));
    check_hip_error(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, ldb));
    check_hip_error(cublasLtMatrixLayoutCreate(&Cdesc, resultType, m, n, ldc));

    check_hip_error(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));

    auto pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    if (mode == 0) {
        pointer_mode =
            per_column_scaling ? CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST : CUBLASLT_POINTER_MODE_DEVICE;
    }
    check_hip_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(hipblasOperation_t)));
    check_hip_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(hipblasOperation_t)));
    check_hip_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &op_b, sizeof(hipblasOperation_t)));
    check_hip_error(cublasLtMatmulDescSetAttribute(
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

    cublasLtMatmulAlgo_t algo;
    void*                workSpace     = hipblas_workspace_;
    int                  workspaceSize = hipblas_workspace_ == NULL ? 0 : HIPBLAS_WORKSPACE_SIZE;

    auto ret = cublasLtMatmulWrapper(hipblaslt_handle_,
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
    check_hip_error(ret);

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    mu_->unlock();
#endif
}

void hipblasMMWrapper::Int8Gemm(const int     m,
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

void hipblasMMWrapper::Int8Gemm(const int     m,
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
#endif

}  // namespace rocm
}  // namespace fastertransformer
