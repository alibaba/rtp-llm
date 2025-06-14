#pragma once

#include "hip_utils.h"
#include "hipblasAlgoMap.h"
#include "rtp_llm/cpp/core/allocator.h"

#include <map>
#include <mutex>
#include <string>

namespace rtp_llm {
namespace rocm {

class hipblasMMWrapper {

protected:
    hipblasHandle_t   hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;

    hipDataType Atype_;
    hipDataType Btype_;
    hipDataType Ctype_;
    hipDataType computeType_;

    hipStream_t          stream_;
    rocm::hipblasAlgoMap hipblas_algo_map_;

    IAllocator* allocator_         = nullptr;
    void*       hipblas_workspace_ = nullptr;

    hipblasLtMatmulPreference_t blasLtPrefer;

    // friend class cublasINT8MMWrapper;

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

public:
    hipblasMMWrapper(hipblasHandle_t   hipblas_handle_,
                     hipblasLtHandle_t hipblaslt_handle_,
                     hipStream_t       stream,
                     IAllocator*       allocator);

    ~hipblasMMWrapper();
    hipblasMMWrapper(const hipblasMMWrapper& wrapper) = delete;

    void Gemm(hipblasOperation_t transa,
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
              float              alpha_ = float(1.0f),
              float              beta_ = float(0.0f));
              
    void GemmBiasAct(hipblasOperation_t transa,
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
                     const hipblasLtEpilogue_t epilogue);

    void setFP32GemmConfig();
    void setFP16GemmConfig();
#ifdef ENABLE_BF16
    void setBF16GemmConfig();
#endif
    void setGemmConfig(hipDataType aType,
                       hipDataType bType,
                       hipDataType cType,
                       hipDataType computeType);

    hipblasDatatype_t getHipBlasDataType(hipDataType data_type);
    hipblasComputeType_t getHipblasLtComputeType(hipDataType data_type);

    void stridedBatchedGemm(hipblasOperation_t transa,
                            hipblasOperation_t transb,
                            const int          m,
                            const int          n,
                            const int          k,
                            const float        f_alpha,
                            const void*        A,
                            hipDataType        AType,
                            const int          lda,
                            const int64_t      strideA,
                            const void*        B,
                            hipDataType        BType,
                            const int          ldb,
                            const int64_t      strideB,
                            const float        f_beta,
                            void*              C,
                            hipDataType        CType,
                            const int          ldc,
                            const int64_t      strideC,
                            const int          batch_count,
                            hipDataType        computeType);
    void setStream(hipStream_t stream) {
        stream_ = stream;
        hipblasSetStream(hipblas_handle_, stream_);
    }
};
}  // namespace rocm
}  // namespace rtp_llm
