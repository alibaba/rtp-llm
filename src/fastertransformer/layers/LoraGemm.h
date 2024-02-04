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
class LoraGemm {
private:
    bool                                                  sparse_ = false;
    cudaStream_t                                          stream_;
    IAllocator*                                           allocator_;
    cublasMMWrapper*                                      cublas_wrapper_;
    std::shared_ptr<CutlassGroupGemmRunner<T>>            group_gemm_runner_;
    static constexpr int                                  MAX_BATCH_SIZE    = 1024;

    T*  input_array_cpu[MAX_BATCH_SIZE]    = {};
    T*  lora_a_array_cpu[MAX_BATCH_SIZE]   = {};
    T*  lora_b_array_cpu[MAX_BATCH_SIZE]   = {};
    T*  lora_buf_array_cpu[MAX_BATCH_SIZE] = {};
    T*  output_array_cpu[MAX_BATCH_SIZE]   = {};
    int m_array_cpu[MAX_BATCH_SIZE]        = {};
    int n_array_cpu[MAX_BATCH_SIZE]        = {};
    int k_array_cpu[MAX_BATCH_SIZE]        = {};
    int r_array_cpu[MAX_BATCH_SIZE]        = {};

    T* lora_buf_ = nullptr;

public:
    LoraGemm(cudaStream_t                                          stream,
               IAllocator*                                           allocator,
               cublasMMWrapper*                                      cublas_wrapper):
        stream_(stream),
        allocator_(allocator),
        cublas_wrapper_(cublas_wrapper) {
        group_gemm_runner_ = std::make_shared<CutlassGroupGemmRunner<T>>();
    }

    ~LoraGemm() {
        freeBuffer();
    }
    void freeBuffer();
    bool useLoRA(const int batch_size, const int* lora_ids, const LoRAWeight<T>* lora_weights);
    void applyLoRA(const int            token_num,
                   const int            batch_size,
                   const int*           lora_input_lengths,
                   const int            k,
                   const int            n,
                   const int*           lora_ids,
                   const LoRAWeight<T>* lora_weights,
                   const T*             input,
                   T*                   output);


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

    void GroupLoraGemm(
        int* m, int* n, int* k, int* r, T** input, T** lora_a, T** lora_b, T** lora_buf, T** output, int count);

    void ForLoraGemm(int* m, int* n, int* k, int* r, T** input, T** lora_a, T** lora_b, T** lora_buf, T** output, int count);
};
}  // namespace fastertransformer