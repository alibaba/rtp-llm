#include "hipblasMMWrapper.h"

namespace rtp_llm {
namespace rocm {

hipblasMMWrapper::hipblasMMWrapper(hipblasHandle_t   hipblas_handle,
                                   hipblasLtHandle_t hipblaslt_handle,
                                   hipStream_t       stream,
                                   IAllocator*       allocator):
    hipblas_handle_(hipblas_handle), hipblaslt_handle_(hipblaslt_handle), stream_(stream), allocator_(allocator) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    hipblas_workspace_ = allocator_->malloc(HIPBLAS_WORKSPACE_SIZE);
    const char* config_path = std::getenv("ROCM_HIPBLASLT_CONFIG");
    if (config_path == nullptr) {
        RTP_LLM_LOG_WARNING("ROCM_HIPBLASLT_CONFIG not set. Defaulting to gemm_config.csv.");
        config_path = "gemm_config.csv";
    }
    hipblas_algo_map_.loadGemmConfig(config_path, hipblaslt_handle);

    int   workspaceSize = HIPBLAS_WORKSPACE_SIZE;
    ROCM_CHECK(hipblasLtMatmulPreferenceCreate(&blasLtPrefer));
    ROCM_CHECK(hipblasLtMatmulPreferenceSetAttribute(blasLtPrefer,
                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspaceSize,
                                          sizeof(workspaceSize)));
}

hipblasMMWrapper::~hipblasMMWrapper() {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    ROCM_CHECK(hipblasLtMatmulPreferenceDestroy(blasLtPrefer));
    allocator_->free((void**)(&hipblas_workspace_));
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
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    float f_alpha(1.0);
    float f_beta(0.0);
    half  h_alpha = (half)(f_alpha);
    half  h_beta  = (half)(f_beta);

    int  is_fp16_computeType = computeType_ == HIPBLAS_R_16F ? 1 : 0;
    bool using_hipblasLt     = (Atype_ == HIPBLAS_R_16F) ? true : false;
    int  batch_count         = 1;

    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    void* workSpace     = hipblas_workspace_;
    int   workspaceSize = HIPBLAS_WORKSPACE_SIZE;

    const auto* info = hipblas_algo_map_.getAlgo(transa,
                                                 transb,
                                                 m,
                                                 n,
                                                 k,
                                                 getHipDataType(Atype_),
                                                 lda,
                                                 0,
                                                 getHipDataType(Btype_),
                                                 ldb,
                                                 0,
                                                 getHipDataType(Ctype_),
                                                 ldc,
                                                 0,
                                                 HIPBLAS_COMPUTE_32F,
                                                 1,
                                                 HIPBLASLT_EPILOGUE_DEFAULT);
    if (info) {
        ROCM_CHECK(hipblasLtMatmul(hipblaslt_handle_,
                        info->opDesc.get(),
                        alpha,
                        A,
                        info->ADesc.get(),
                        B,
                        info->BDesc.get(),
                        beta,
                        C,
                        info->CDesc.get(),
                        C,
                        info->CDesc.get(),
                        &info->algo,
                        workSpace,
                        workspaceSize,
                        stream_));
    } else {
        hipblasLtMatrixLayout_t ADesc, BDesc, CDesc;
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&ADesc, getHipDataType(Atype_), transa == HIPBLAS_OP_N ? m : k, transa == HIPBLAS_OP_N ? k : m, lda));
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&BDesc, getHipDataType(Btype_), transb == HIPBLAS_OP_N ? k : n, transb == HIPBLAS_OP_N ? n : k, ldb));
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&CDesc, getHipDataType(Ctype_), m, n, ldc));

        hipblasLtMatmulDesc_t matmul;
        ROCM_CHECK(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, getHipDataType(computeType_)));
        hipblasOperation_t trans_a = transa;
        hipblasOperation_t trans_b = transb;
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

        hipblasStatus_t blaslt_status = hipblasLtMatmul(hipblaslt_handle_,
                                                        matmul,
                                                        alpha,
                                                        A,
                                                        ADesc,
                                                        B,
                                                        BDesc,
                                                        beta,
                                                        C,
                                                        CDesc,
                                                        C,
                                                        CDesc,
                                                        NULL,
                                                        workSpace,
                                                        workspaceSize,
                                                        stream_);

        if (blaslt_status != HIPBLAS_STATUS_SUCCESS) {
            RTP_LLM_LOG_WARNING("[BLAS] blaslt failed, back to blas.");
            RTP_LLM_LOG_WARNING("[BLAS] trans = %c%c, mnk = [%d,%d,%d], ld = [%d,%d,%d]",
                           transa == HIPBLAS_OP_N ? 'N' : 'T',
                           transb == HIPBLAS_OP_N ? 'N' : 'T',
                           m,
                           n,
                           k,
                           lda,
                           ldb,
                           ldc);
            ROCM_CHECK(hipblasGemmEx(hipblas_handle_,
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
                                     HIPBLAS_GEMM_DEFAULT));
        }

        ROCM_CHECK(hipblasLtMatrixLayoutDestroy(ADesc));
        ROCM_CHECK(hipblasLtMatrixLayoutDestroy(BDesc));
        ROCM_CHECK(hipblasLtMatrixLayoutDestroy(CDesc));
        ROCM_CHECK(hipblasLtMatmulDescDestroy(matmul));
    }
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

    int         is_fp16_computeType = computeType == HIPBLAS_R_16F ? 1 : 0;
    const void* alpha =
        is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    
    ROCM_CHECK(hipblasGemmStridedBatchedEx(hipblas_handle_,
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
                                                HIPBLAS_GEMM_DEFAULT));
}

void hipblasMMWrapper::GemmBiasAct(hipblasOperation_t transa,
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
                            const void*        bias,
                            const hipblasLtEpilogue_t epilogue) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    float f_alpha(1.0);
    float f_beta(0.0);
    half  h_alpha = (half)(f_alpha);
    half  h_beta  = (half)(f_beta);

    int  is_fp16_computeType = computeType_ == HIPBLAS_R_16F ? 1 : 0;
    bool using_hipblasLt     = (Atype_ == HIPBLAS_R_16F) ? true : false;
    int  batch_count         = 1;

    void* workSpace     = hipblas_workspace_;
    int   workspaceSize = HIPBLAS_WORKSPACE_SIZE;

    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    const auto* info = hipblas_algo_map_.getAlgo(transa,
                                                 transb,
                                                 m,
                                                 n,
                                                 k,
                                                 getHipDataType(Atype_),
                                                 lda,
                                                 0,
                                                 getHipDataType(Btype_),
                                                 ldb,
                                                 0,
                                                 getHipDataType(Ctype_),
                                                 ldc,
                                                 0,
                                                 HIPBLAS_COMPUTE_32F,
                                                 1,
                                                 epilogue);

    if(info)
    {
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(info->opDesc.get(), HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));

        ROCM_CHECK(hipblasLtMatmul(hipblaslt_handle_,
                        info->opDesc.get(),
                        alpha,
                        A,
                        info->ADesc.get(),
                        B,
                        info->BDesc.get(),
                        beta,
                        C,
                        info->CDesc.get(),
                        C,
                        info->CDesc.get(),
                        &info->algo,
                        workSpace,
                        workspaceSize,
                        stream_));
    }
    else
    {
        hipblasLtMatrixLayout_t ADesc, BDesc, CDesc;
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&ADesc, getHipDataType(Atype_), m, k, lda));
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&BDesc, getHipDataType(Btype_), k, n, ldb));
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&CDesc, getHipDataType(Ctype_), m, n, ldc));

        hipblasLtMatmulDesc_t matmul;
        ROCM_CHECK(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t trans_a = transa;
        hipblasOperation_t trans_b = transb;
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

        hipblasLtEpilogue_t epilogue_ = epilogue;
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_, sizeof(epilogue_)));
        int32_t bias_data_type = getHipDataType(Ctype_);
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        ROCM_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
        
        const int                        request_solutions = 1;
        hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
        int                              returnedAlgoCount = 0;
        ROCM_CHECK(hipblasLtMatmulAlgoGetHeuristic(hipblaslt_handle_,
                                        matmul,
                                        ADesc,
                                        BDesc,
                                        CDesc,
                                        CDesc,
                                        blasLtPrefer,
                                        request_solutions,
                                        heuristicResult,
                                        &returnedAlgoCount)); 

        ROCM_CHECK(hipblasLtMatmul(hipblaslt_handle_,
                        matmul,
                        alpha,
                        A,
                        ADesc,
                        B,
                        BDesc,
                        beta,
                        C,
                        CDesc,
                        C,
                        CDesc,
                        &heuristicResult[0].algo,
                        workSpace,
                        workspaceSize,
                        stream_));

        ROCM_CHECK(hipblasLtMatrixLayoutDestroy(ADesc));
        ROCM_CHECK(hipblasLtMatrixLayoutDestroy(BDesc));
        ROCM_CHECK(hipblasLtMatrixLayoutDestroy(CDesc));
        ROCM_CHECK(hipblasLtMatmulDescDestroy(matmul));
    }
}

}  // namespace rocm
}  // namespace rtp_llm
