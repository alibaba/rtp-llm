#pragma once

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {
class rocmFmhaWrapper {
private:
    /* data */
    DataType dtype_;
    bool     is_causal_;

    size_t head_num_;
    size_t kv_head_num_;
    size_t size_per_head_;
    float  q_scaling_;

    hipStream_t stream_;

public:
    rocmFmhaWrapper(/* args */) = default;
    ~rocmFmhaWrapper()          = default;

    void init(hipStream_t stream) {
        stream_ = stream;
    }
    void setup(DataType dtype,
               bool     is_causal,
               size_t   head_num,
               size_t   kv_head_num,
               size_t   size_per_head,
               float    q_scaling) {
        dtype_         = dtype;
        is_causal_     = is_causal;
        head_num_      = head_num;
        kv_head_num_   = kv_head_num;
        size_per_head_ = size_per_head;
        q_scaling_     = q_scaling;
    }
    uint32_t runCKFmha(void*  q,
                       void*  k,
                       void*  v,
                       void*  output,
                       void*  softmax_lse_,
                       size_t batch_size,
                       size_t seq_len,
                       size_t max_prefix_prompt_length,
                       void*  seqstart_q,
                       void*  seqstart_k,
                       void*  lse_acc_buf,
                       void*  linear_bias_slopes = nullptr,
                       void*  biasBuffer         = nullptr,
                       bool   i_perm_            = false,  // if true, will be batch * nhead * seqlen * hdim
                       bool   o_perm_            = false   // if false, will be batch * seqlen * nhead * hdim
    );
    uint32_t runCKFmhaV2(void*  q,
                         void*  k,
                         void*  v,
                         void*  output,
                         void*  softmax_lse_,
                         size_t batch_size,
                         size_t seq_len,
                         size_t max_prefix_prompt_length,
                         void*  seqstart_q,
                         void*  seqstart_k,
                         void*  lse_acc_buf,
                         void*  linear_bias_slopes = nullptr,
                         void*  biasBuffer         = nullptr,
                         size_t token_num          = 0,
                         bool   i_perm_            = false,  // if true, will be batch * nhead * seqlen * hdim
                         bool   o_perm_            = false   // if false, will be batch * seqlen * nhead * hdim
    );
    uint32_t runCKFmhaMLA(void*  q,
                          void*  k,
                          void*  v,
                          void*  output,
                          void*  softmax_lse_,
                          size_t batch_size,
                          size_t seq_len,
                          float  softmax_extra_scale,
                          void*  seqstart_q,
                          void*  seqstart_k,
                          void*  lse_acc_buf,
                          void*  linear_bias_slopes = nullptr,
                          void*  biasBuffer         = nullptr);
};

}  // namespace rtp_llm
