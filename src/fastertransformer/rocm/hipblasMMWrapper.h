#pragma once

#include "hip_utils.h"
#include "hipblasAlgoMap.h"
#include "src/fastertransformer/core/allocator.h"

#include <map>
#include <mutex>
#include <string>

namespace fastertransformer {
namespace rocm {

class hipblasMMWrapper {

protected:
    hipblasHandle_t   hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;

    hipblasDatatype_t Atype_;
    hipblasDatatype_t Btype_;
    hipblasDatatype_t Ctype_;
    hipblasDatatype_t computeType_;

    hipStream_t           stream_;
    rocm::hipblasAlgoMap* hipblas_algo_map_;
    std::mutex*           mu_;

    IAllocator* allocator_         = nullptr;
    void*       hipblas_workspace_ = nullptr;

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
    hipblasMMWrapper(hipblasHandle_t       hipblas_handle_,
                     hipblasLtHandle_t     hipblaslt_handle_,
                     hipStream_t           stream,
                     rocm::hipblasAlgoMap* map,
                     std::mutex*           mu,
                     IAllocator*           allocator);

    ~hipblasMMWrapper();
    hipblasMMWrapper(const hipblasMMWrapper& wrapper);
    virtual void hipblasVersionCheck() {
        return;
    };

    void Gemm(hipblasOperation_t transa,
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
              hipblasGemmAlgo_t  algo);

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
              const int          ldc);

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
              float              f_alpha,
              float              f_beta);

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

    void setStream(hipStream_t stream);
    void setFP32GemmConfig();
    void setFP16GemmConfig();
#ifdef ENABLE_BF16
    void setBF16GemmConfig();
#endif
    void setGemmConfig(hipblasDatatype_t aType,
                       hipblasDatatype_t bType,
                       hipblasDatatype_t cType,
                       hipblasDatatype_t computeType);

    hipDataType          getHipDataType(hipblasDatatype_t data_type);
    hipblasComputeType_t getHipblasLtComputeType(hipblasDatatype_t data_type);

    void Gemm(hipblasOperation_t transa,
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
              const int          ldc);

    void stridedBatchedGemm(hipblasOperation_t transa,
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
                            const int          batchCount,
                            const float        f_alpha = 1.0f,
                            const float        f_beta  = 0.0f);

    void stridedBatchedGemm(hipblasOperation_t transa,
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
                            hipblasDatatype_t  computeType);

    void batchedGemm(hipblasOperation_t transa,
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
                     const int          batch_count);
};
}  // namespace rocm
}  // namespace fastertransformer
