#pragma once

#include <string>
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/LoRAWeight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/cutlass/interface.h"

namespace fastertransformer {

template<typename T>
class GemmRunner {
private:
    bool         sparse_ = false;
    cudaStream_t stream_;
    IAllocator*  allocator_;
    cublasMMWrapper*                                      cublas_wrapper_;
    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner_;
    std::shared_ptr<CutlassGroupGemmRunner<T>>         group_gemm_runner_;
    static constexpr int SMALL_M_FAST_PATH = 4;
    static constexpr int MAX_BATCH_SIZE = 1024;
    bool weight_only_cuda_kernel_enabled_;

    T* input_array_cpu[MAX_BATCH_SIZE]    = {};
    T* lora_a_array_cpu[MAX_BATCH_SIZE]   = {};
    T* lora_b_array_cpu[MAX_BATCH_SIZE]   = {};
    T* lora_buf_array_cpu[MAX_BATCH_SIZE] = {};
    T* output_array_cpu[MAX_BATCH_SIZE]   = {};
    int m_array_cpu[MAX_BATCH_SIZE]       = {};
    int n_array_cpu[MAX_BATCH_SIZE]       = {};
    int k_array_cpu[MAX_BATCH_SIZE]       = {};
    int r_array_cpu[MAX_BATCH_SIZE]       = {};

    T* lora_buf_ = nullptr;

public:
    GemmRunner(bool                                                  sparse,
               cudaStream_t                                          stream,
               IAllocator*                                           allocator,
               cublasMMWrapper*                                      cublas_wrapper,
               std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner):
        sparse_(sparse),
        stream_(stream),
        allocator_(allocator),
        cublas_wrapper_(cublas_wrapper),
        weight_only_int8_fc_runner_(weight_only_int8_fc_runner)
    {
#if defined (USE_WEIGHT_ONLY) && USE_WEIGHT_ONLY == 1
        weight_only_cuda_kernel_enabled_ = fastertransformer::kernels::isWeightOnlyBatchedGemvEnabled(fastertransformer::kernels::WeightOnlyQuantType::Int8b);
#else
        weight_only_cuda_kernel_enabled_ = false;
#endif
        group_gemm_runner_ = std::make_shared<CutlassGroupGemmRunner<T>>();

    }

    ~GemmRunner()
    {
        freeBuffer();
    }
    void freeBuffer();
    bool useLoRA(const int batch_size, const int* lora_ids, const LoRAWeight<T>* lora_weights);
    void applyLoRA(const int            s,
                   const int            b,
                   const int*           lora_input_lengths,
                   const int            k,
                   const int            n,
                   const int*           lora_ids,
                   const LoRAWeight<T>* lora_weights,
                   const T*             input,
                   T*                   output);

    void Gemm(int                      batch_size,
              const int*               lora_input_lengths,
              int                      m,
              int                      n,
              int                      k,
              const T*                 input,
              const DenseWeight<T, T>* weight,
              T*                       output,
              const int*               lora_ids,
              int                      int8_mode,
              bool                     use_sparse,
              char*                    mixed_gemm_workspace,
              size_t                   mixed_gemm_ws_bytes,
              int                      m_padded);

private:
    void allocateBuffer(size_t s, size_t k, size_t n, size_t r);

    void LoRAGemm(const int m,
                  const int n,
                  const int k,
                  const int r,
                  const T*  input,
                  const T*  lora_a,
                  const T*  lora_b,
                  T*        lora_buf,
                  T*        output);
    
    void BatchLoraGemm(const int b, const int k, const int n);
    
    void GroupLoraGemm(int* m,
                       int* n,
                       int* k,
                       int* r, 
                       T**  input,
                       T**  lora_a,
                       T**  lora_b,
                       T**  lora_buf,
                       T**  output,
                       int count);
    
    void ForLoraGemm(int* m,
                    int* n,
                    int* k,
                    int* r, 
                    T**  input,
                    T**  lora_a,
                    T**  lora_b,
                    T**  lora_buf,
                    T**  output,
                    int count);
};
}  // namespace fastertransformer