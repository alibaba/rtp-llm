#include "src/fastertransformer/layers/GemmRunner.h"
#include <fstream>
namespace fastertransformer {

template<typename T>
void GemmRunner<T>::allocateBuffer(size_t s, size_t k, size_t n, size_t r) {
    lora_buf_ = (T*)allocator_->reMalloc(lora_buf_, sizeof(T) * s * r, true);
}

template<typename T>
bool GemmRunner<T>::useLoRA(const int batch_size, const int* lora_ids, const LoRAWeight<T>* lora_weights) {
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
void GemmRunner<T>::freeBuffer() {
    allocator_->free((void**)&lora_buf_);
}

/// @brief apply different lora or unique lora to multiple tokens.
/// @tparam s: the token num of inputs.
/// @tparam b: the batch size of inputs.
/// @tparam input_lengths: [batch] the token num for each batch,
///         the sum of input_lengths must be equal to s.
/// @tparam k: the hidden units
/// @tparam n: the intersize
/// @tparam lora_ids: [batch] the lora model id for each batch.
/// @tparam lora_weights: all lora model weights ref.

template<typename T>
void GemmRunner<T>::applyLoRA(const int            s,
                              const int            b,
                              const int*           input_lengths,
                              const int            k,
                              const int            n,
                              const int*           lora_ids,
                              const LoRAWeight<T>* lora_weights,
                              const T*             input,
                              T*                   output) {
    int r = lora_weights->max_rank;
    // FT_LOG_INFO("apply lora Gemm: {s : %d, b : %d, k : %d, n : %d, r : %d,}", s, b, k, n, r);
    if (r == 0)
        return;
    // input: [s, k]
    // lora_buf: [s, r]
    // output: [s, n]
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
    allocateBuffer(s, k, n, r);

    if (!use_single_lora) {
        FT_CHECK(input_lengths != nullptr);
        FT_CHECK(b < MAX_BATCH_SIZE);
        int total_token_num = 0;
        int extent          = 0;  // the number of zero lora
        int index           = 0;

        for (int i = 0; i < b; i++) {
            const auto& w    = lora_weights->getLoRAWeight(lora_ids[i]);
            const int   rank = lora_weights->getLoRARank(lora_ids[i]);
            if (rank != 0 && w.first != nullptr && w.second != nullptr) {
                lora_a_array_cpu[index]   = w.first;
                lora_b_array_cpu[index]   = w.second;
                r_array_cpu[index]        = rank;
                input_array_cpu[index]    = const_cast<T*>(input);
                output_array_cpu[index]   = output;
                lora_buf_array_cpu[index] = lora_buf_;
                m_array_cpu[index]        = input_lengths[i];
                n_array_cpu[index]        = n;
                k_array_cpu[index]        = k;
                index                     = index + 1;
            } else {
                extent = extent + 1;
            }

            input           = input + input_lengths[i] * k;
            output          = output + input_lengths[i] * n;
            lora_buf_       = lora_buf_ + input_lengths[i] * r;
            total_token_num = total_token_num + input_lengths[i];
        }
        FT_CHECK(total_token_num == s);
        BatchLoraGemm(b - extent, n, k);

    } else {
        const auto& w    = lora_weights->getLoRAWeight(lora_ids[0]);
        const int   rank = lora_weights->getLoRARank(lora_ids[0]);
        if (rank == 0) {
            return;
        }
        const T* lora_a = w.first;
        const T* lora_b = w.second;
        LoRAGemm(s, n, k, rank, input, lora_a, lora_b, lora_buf_, output);
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
            if (m < SMALL_M_FAST_PATH && weight_only_cuda_kernel_enabled_ && k <= 20480 && n <= 25600) {
#if defined(USE_WEIGHT_ONLY) && USE_WEIGHT_ONLY == 1
                // PUSH_RANGE(stream_, fmtstr("weight only batched gemv: [%d,%d,%d]", m, n, k));
                fastertransformer::kernels::WeightOnlyActivationType weight_only_act_type;

                if (std::is_same<T, half>::value) {
                    weight_only_act_type = fastertransformer::kernels::WeightOnlyActivationType::FP16;
                }
#ifdef ENABLE_BF16
                else if (std::is_same<T, __nv_bfloat16>::value) {
                    weight_only_act_type = fastertransformer::kernels::WeightOnlyActivationType::BF16;
                }
#endif
                else {
                    FT_LOG_ERROR("weight only batched gemv only support half and bf16");
                }
                fastertransformer::kernels::WeightOnlyParams weight_only_batched_gemv_params{
                    reinterpret_cast<const uint8_t*>(weight->int8_kernel),
                    reinterpret_cast<const void*>(weight->weight_only_quant_scale),
                    nullptr,
                    reinterpret_cast<const void*>(input),
                    nullptr,
                    reinterpret_cast<void*>(output),
                    m,
                    n,
                    k,
                    0,
                    fastertransformer::kernels::WeightOnlyQuantType::Int8b,
                    fastertransformer::kernels::WeightOnlyType::PerChannel,
                    fastertransformer::kernels::WeightOnlyActivationFunctionType::Identity,
                    weight_only_act_type};
                fastertransformer::kernels::weight_only_batched_gemv_launcher(weight_only_batched_gemv_params, stream_);
                // POP_RANGE;
#endif
            } else {
                // Otherwise, let FT handle activation
                // PUSH_RANGE(stream_, fmtstr("weight_only_int8_fc gemm: [%d,%d,%d]", m, n, k));
                weight_only_int8_fc_runner_->gemm(input,
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
    if (useLoRA(batch_size, lora_ids, weight->lora_weights)) {
        applyLoRA(m, batch_size, input_lengths, k, n, lora_ids, weight->lora_weights, input, output);
        sync_check_cuda_error();
    }
}

template<typename T>
void GemmRunner<T>::LoRAGemm(const int m,
                             const int n,
                             const int k,
                             const int r,
                             const T*  input,
                             const T*  lora_a,
                             const T*  lora_b,
                             T*        lora_buf_,
                             T*        output) {
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

template<typename T>
void GemmRunner<T>::BatchLoraGemm(const int b, const int k, const int n) {
    bool same_rank = false;
    int  rank      = r_array_cpu[0];
    int  m         = m_array_cpu[0];
    for (int i = 0; i < b; i++) {
        same_rank = (r_array_cpu[0] == rank) && (m == m_array_cpu[i]);
    }

    // TODO(lidongjin): (32x32) can not use group gemm.
    if (n < 32 && k < 32) {
        ForLoraGemm(m_array_cpu,
                    n_array_cpu,
                    k_array_cpu,
                    r_array_cpu,
                    input_array_cpu,
                    lora_a_array_cpu,
                    lora_b_array_cpu,
                    lora_buf_array_cpu,
                    output_array_cpu,
                    b);

    } else if (same_rank) {
        // TODO(lidongjin): when token num and rank id same, use batch gemm.
        GroupLoraGemm(m_array_cpu,
                      n_array_cpu,
                      k_array_cpu,
                      r_array_cpu,
                      input_array_cpu,
                      lora_a_array_cpu,
                      lora_b_array_cpu,
                      lora_buf_array_cpu,
                      output_array_cpu,
                      b);
    } else {
        GroupLoraGemm(m_array_cpu,
                      n_array_cpu,
                      k_array_cpu,
                      r_array_cpu,
                      input_array_cpu,
                      lora_a_array_cpu,
                      lora_b_array_cpu,
                      lora_buf_array_cpu,
                      output_array_cpu,
                      b);
    }
}

template<typename T>
void GemmRunner<T>::GroupLoraGemm(
    int* m, int* n, int* k, int* r, T** input, T** lora_a, T** lora_b, T** lora_buf, T** output, int count) {
    // X = [b, m, k]
    // A = [b, k, r]
    // B = [b, r, n]
    // M = [b, m, r]
    // Y = [b, m, n]
    // M = X * A
    group_gemm_runner_->gemm(input, lora_a, lora_buf, m, r, k, 1.0f, 0.0f, count, stream_);
    // Y = M * B + Y
    group_gemm_runner_->gemm(lora_buf, lora_b, output, m, n, r, 1.0f, 1.0f, count, stream_);
}

template<typename T>
void GemmRunner<T>::ForLoraGemm(
    int* m, int* n, int* k, int* r, T** input, T** lora_a, T** lora_b, T** lora_buf, T** output, int count) {
    // X = [b, m, k]
    // A = [b, k, r]
    // B = [b, r, n]
    // M = [b, m, r]
    // Y = [b, m, n]
    // M = X * A
    // Y = M * B + Y
    for (size_t i = 0; i < count; i++) {
        LoRAGemm(m[i], n[i], k[i], r[i], input[i], lora_a[i], lora_b[i], lora_buf[i], output[i]);
    }
}

template class GemmRunner<float>;
template class GemmRunner<half>;
#ifdef ENABLE_BF16
template class GemmRunner<__nv_bfloat16>;
#endif
}  // namespace fastertransformer
