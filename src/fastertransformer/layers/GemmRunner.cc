#include "src/fastertransformer/layers/GemmRunner.h"
#include <fstream>
namespace fastertransformer {

template<typename T>
void GemmRunner<T>::allocateBuffer(size_t s, size_t r)
{
    int lora_buf_size = s * r;
    lora_buf_       = (T*)allocator_->reMalloc(lora_buf_, sizeof(T) * lora_buf_size);
}

template<typename T>
bool GemmRunner<T>::useLoRA(const int batch_size, const int* lora_ids, const LoRAWeight<T>* lora_weights)
{
    if (lora_ids == nullptr || lora_weights == nullptr)
        return false;
    for (int i = 0; i < batch_size; i++) {
        const auto& w = lora_weights->getLoRAWeight(lora_ids[i]);
        if (w.first != nullptr && w.second != nullptr)
            return true;
    }
    return false;
}
template<typename T>
void GemmRunner<T>::freeBuffer()
{
    allocator_->free((void**)&lora_buf_);
}
template<typename T>
void GemmRunner<T>::applyLoRA(const int            s,
                              const int            b,
                              const int*           input_lengths,
                              const int            k,
                              const int            n,
                              const int*           lora_ids,
                              const LoRAWeight<T>* lora_weights,
                              const T*             input,
                              T*                   output)
{
    int r = lora_weights->max_rank;
    // FT_LOG_INFO("apply lora Gemm: {s : %d, b : %d, k : %d, n : %d, r : %d,}", s, b, k, n, r);
    if (r == 0)
        return;
    if (finished_copy)
        cudaEventSynchronize(finished_copy);
    else
        cudaEventCreate(&finished_copy);
    // input: [s, k]
    // lora_buf: [s, r]
    // output: [s, n]
    // for context decoder s = b * m - padding_size
    // so all lora_ids should be the same
    // for decoder, m = 1, s = b * m
    // lora_ids can be different
    bool use_lora        = false;
    bool use_single_lora = true;
    int  pre_lora_id     = lora_ids[0];
    for (int i = 0; i < b; i++) {
        // check if all lora_ids are the same
        if (pre_lora_id != lora_ids[i])
            use_single_lora = false;
        // check if any lora_id is valid
        const auto& w = lora_weights->getLoRAWeight(lora_ids[i]);
        if (w.first != nullptr && w.second != nullptr)
            use_lora = true;
    }
    if (!use_lora) {
        return;
    }
    allocateBuffer(s, r);

    if (!use_single_lora) {
        FT_CHECK(input_lengths != nullptr);
        int total_token_num = 0;
        for (int i = 0; i < b; i++) {
            const auto& w  = lora_weights->getLoRAWeight(lora_ids[i]);
            const int rank = lora_weights->getLoRARank(lora_ids[i]);
            if (rank != 0) {
                const T*    lora_a = nullptr;
                const T*    lora_b = nullptr;
                if (w.first != nullptr && w.second != nullptr) {
                    lora_a = w.first;
                    lora_b = w.second;
                    LoRAGemm(input_lengths[i], n, k, rank, input, lora_a, lora_b, output);
                }
            }
            input = input + input_lengths[i] * k;
            output = output + input_lengths[i] * n;
            total_token_num = total_token_num + input_lengths[i];
              
        }
        FT_CHECK(total_token_num == s);
    }
    else {
        const auto& w      = lora_weights->getLoRAWeight(lora_ids[0]);
        const int rank     = lora_weights->getLoRARank(lora_ids[0]);
        if (rank == 0) {
            return;
        }
        const T*    lora_a = w.first;
        const T*    lora_b = w.second;
        LoRAGemm(s, n, k, rank, input, lora_a, lora_b, output);
    }
}


template<typename T>
void GemmRunner<T>::Gemm(int                   batch_size,
                         const int*            input_lengths,
                         int                   m,
                         int                   n,
                         int                   k,
                         const T*              input,
                         const DenseWeight<T>* weight,
                         T*                    output,
                         const int*            lora_ids,
                         int                   int8_mode,
                         bool                  use_sparse,
                         char*                 mixed_gemm_workspace,
                         size_t                mixed_gemm_ws_bytes,
                         int                   m_padded) {
    // input: [m, k]
    // weight: [k, n]
    // output: [m, n]
    
    if (use_sparse) {
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, weight->sp_kernel, input, output);
#endif
    } else {
        if (int8_mode == 1) {
            if (m < SMALL_M_FAST_PATH && weight_only_cuda_kernel_enabled_ && k <= 20480 && n <= 25600 ) {
#if defined(USE_WEIGHT_ONLY) && USE_WEIGHT_ONLY == 1
                // PUSH_RANGE(stream_, fmtstr("weight only batched gemv: [%d,%d,%d]", m, n, k));
                fastertransformer::kernels::WeightOnlyParams ffn_gemm_params{
                    reinterpret_cast<const uint8_t*>(weight->int8_kernel),
                    reinterpret_cast<const half*>(weight->weight_only_quant_scale),
                    nullptr,
                    reinterpret_cast<const half*>(input),
                    nullptr,
                    reinterpret_cast<half*>(output),
                    m,
                    n,
                    k,
                    0};
                fastertransformer::kernels::weight_only_batched_gemv_launcher(
                    fastertransformer::kernels::WeightOnlyQuantType::Int8b,
                    fastertransformer::kernels::WeightOnlyType::PerChannel,
                    fastertransformer::kernels::WeightOnlyActivationType::Identity,
                    ffn_gemm_params,
                    stream_);
                // POP_RANGE;
#endif
            } else {
                // Otherwise, let FT handle activation
                // PUSH_RANGE(stream_, fmtstr("weight_only_int8_fc gemm: [%d,%d,%d]", m, n, k));
                weight_only_int8_fc_runner_->gemm(
                    input,
                    reinterpret_cast<const uint8_t*>(weight->int8_kernel),
                    weight->weight_only_quant_scale,
                    output,
                    m,
                    n,
                    k,
                    mixed_gemm_workspace,
                    mixed_gemm_ws_bytes,
                    stream_);
                // POP_RANGE;
            }
        } else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, weight->kernel, n, input, k, output, n);
            sync_check_cuda_error();
        }
    }

    // lora_gemm
    if (lora_ids) {
        applyLoRA(m, batch_size, input_lengths, k, n, lora_ids, weight->lora_weights, input, output);
        sync_check_cuda_error();
    }
}

template<typename T>
void GemmRunner<T>::LoRAGemm(
    const int m, const int n, const int k, const int r, const T* input, const T* lora_a, const T* lora_b, T* output)
{
    // X = [m, k]
    // A = [k, r]
    // B = [r, n]
    // M = [m, r]
    // Y = [m, n]
    // M = X * A
    cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, r, m, k, lora_a, r, input, k, lora_buf_, r, 1.0f, 0.0f);
    // Y = M * B + Y
    cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, r, lora_b, n, lora_buf_, r, output, n, 1.0f, 1.0f);
}


template class GemmRunner<float>;
template class GemmRunner<half>;
#ifdef ENABLE_BF16
template class GemmRunner<__nv_bfloat16>;
#endif
}  // namespace fastertransformer