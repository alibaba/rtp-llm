#pragma once

#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/LoRAWeight.h"
#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include <string>

namespace fastertransformer {

template<typename T>
class GemmRunner {
private:
    bool                                                  sparse_ = false;
    cudaStream_t                                          stream_;
    cublasMMWrapper*                                      cublas_wrapper_;
    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner_;
    static constexpr int                                  SMALL_M_FAST_PATH = 4;
    bool                                                  weight_only_cuda_kernel_enabled_;


public:
    GemmRunner(bool                                                  sparse,
               cudaStream_t                                          stream,
               cublasMMWrapper*                                      cublas_wrapper,
               std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner):
        sparse_(sparse),
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        weight_only_int8_fc_runner_(weight_only_int8_fc_runner) {
#if defined(USE_WEIGHT_ONLY) && USE_WEIGHT_ONLY == 1
        weight_only_cuda_kernel_enabled_ = fastertransformer::kernels::isWeightOnlyBatchedGemvEnabled(
            fastertransformer::kernels::WeightOnlyQuantType::Int8b);
#else
        weight_only_cuda_kernel_enabled_ = false;
#endif
    }

    ~GemmRunner() = default;

    void Gemm(int                      m,
              int                      n,
              int                      k,
              const T*                 input,
              const DenseWeight<T, T>* weight,
              T*                       output,
              int                      int8_mode,
              bool                     use_sparse,
              char*                    mixed_gemm_workspace,
              size_t                   mixed_gemm_ws_bytes,
              int                      m_padded);
};

}  // namespace fastertransformer