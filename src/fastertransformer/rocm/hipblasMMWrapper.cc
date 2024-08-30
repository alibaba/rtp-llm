#include "hipblasMMWrapper.h"

namespace fastertransformer {
namespace rocm {

hipblasMMWrapper::hipblasMMWrapper(hipblasHandle_t   hipblas_handle,
                                   hipblasLtHandle_t hipblaslt_handle,
                                   hipStream_t       stream,
                                   IAllocator*       allocator):
    hipblas_handle_(hipblas_handle), hipblaslt_handle_(hipblaslt_handle), stream_(stream), allocator_(allocator) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
#define HIPBLAS_WORKSPACE_SIZE 33554432  // 32MB
    hipblas_workspace_ = allocator_->malloc(HIPBLAS_WORKSPACE_SIZE);
    const char* config_path = std::getenv("ROCM_HIPBLASLT_CONFIG");
    if (config_path == nullptr) {
        FT_LOG_WARNING("ROCM_HIPBLASLT_CONFIG not set. Defaulting to gemm_config.csv.");
        config_path = "gemm_config.csv";
    }
    hipblas_algo_map_.loadGemmConfig(config_path, hipblaslt_handle);
}

hipblasMMWrapper::~hipblasMMWrapper() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
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
    float f_alpha(1.0);
    float f_beta(0.0);
    half  h_alpha = (half)(f_alpha);
    half  h_beta  = (half)(f_beta);

    int  is_fp16_computeType = computeType_ == HIPBLAS_R_16F ? 1 : 0;
    bool using_hipblasLt     = (Atype_ == HIPBLAS_R_16F) ? true : false;
    int  batch_count         = 1;

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
                                                 1);
    static bool disable_hipblasLt = []() {
        auto env = std::getenv("ROCM_DISABLE_HIPBLASLT");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();

    void* workSpace     = hipblas_workspace_;
    int   workspaceSize = HIPBLAS_WORKSPACE_SIZE;

    if (info && !disable_hipblasLt) {
        //printf("DEBUG -> Calling the hipblasLtMatmul\n");
        hipblasLtMatmul(hipblaslt_handle_,
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
                        stream_);
    } else {
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

}  // namespace rocm
}  // namespace fastertransformer
