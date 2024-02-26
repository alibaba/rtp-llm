#include "src/fastertransformer/layers/LoraGemm.h"
#include <fstream>
namespace fastertransformer {

template<typename T>
void LoraGemm<T>::allocateBuffer(size_t s, size_t k, size_t n, size_t r) {
    lora_buf_ = (T*)allocator_->reMalloc(lora_buf_, sizeof(T) * s * r, true);
}

template<typename T>
bool LoraGemm<T>::useLoRA(const int batch_size, const int* lora_ids, const LoRAWeight<T>* lora_weights) {
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
void LoraGemm<T>::freeBuffer() {
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
void LoraGemm<T>::applyLoRA(const int              s,
                              const int            b,
                              const int*           input_lengths,
                              const int            k,
                              const int            n,
                              const int*           lora_ids,
                              const LoRAWeight<T>* lora_weights,
                              const T*             input,
                              T*                   output) {
    if (!useLoRA(b, lora_ids, lora_weights)) {
        return;
    }
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
    T* lora_buf_tmp = lora_buf_;

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
                lora_buf_array_cpu[index] = lora_buf_tmp;
                m_array_cpu[index]        = input_lengths[i];
                n_array_cpu[index]        = n;
                k_array_cpu[index]        = k;
                index                     = index + 1;
            } else {
                extent = extent + 1;
            }

            input           = input + input_lengths[i] * k;
            output          = output + input_lengths[i] * n;
            lora_buf_tmp    = lora_buf_tmp + input_lengths[i] * r;
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
void LoraGemm<T>::LoRAGemm(const int m,
                             const int n,
                             const int k,
                             const int r,
                             const T*  input,
                             const T*  lora_a,
                             const T*  lora_b,
                             T*        lora_buf,
                             T*        output) {
    // X = [m, k]
    // A = [k, r]
    // B = [r, n]
    // M = [m, r]
    // Y = [m, n]
    // M = X * A
    cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, r, m, k, lora_a, r, input, k, lora_buf, r, 1.0f, 0.0f);
    // Y = M * B + Y
    cublas_wrapper_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, r, lora_b, n, lora_buf, r, output, n, 1.0f, 1.0f);
}

template<typename T>
void LoraGemm<T>::BatchLoraGemm(const int b, const int k, const int n) {
    bool same_rank = true;
    int  rank      = r_array_cpu[0];
    int  m         = m_array_cpu[0];
    for (int i = 0; i < b; i++) {
        if ((r_array_cpu[i] != rank) || (m != m_array_cpu[i])) {
            same_rank = false;
        }
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
void LoraGemm<T>::GroupLoraGemm(
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
void LoraGemm<T>::ForLoraGemm(
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

template class LoraGemm<float>;
template class LoraGemm<half>;
#ifdef ENABLE_BF16
template class LoraGemm<__nv_bfloat16>;
#endif
}  // namespace fastertransformer
