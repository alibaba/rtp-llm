#pragma once

#include "3rdparty/flash_attention/flash_api.h"
#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttentionSm70/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm{

class cufmha {

public:
    cufmha(DataType          dtype,
           AttentionMaskType mtype,
           size_t            head_num,
           size_t            kv_head_num,
           size_t            size_per_head,
           float             q_scaling,
           bool              use_linear_bias_slopes,
           bool              can_use_trtv1_fmha,
           bool              can_use_trtv2_fmha,
           bool              can_use_trtv2_fmha_paged,
           bool              can_use_open_source_fmha,
           bool              can_use_open_source_fmha_paged,
           cudaStream_t      stream);

    ~cufmha() = default;

    bool trtV1FmhaSupport() {
        return support_trt_v1_fmha_;
    }

    bool trtV2FmhaSupport() {
        return support_trt_v2_fhma_;
    }

    bool trtV2FmhaPagedSupport() {
        return support_trt_v2_paged_fmha_;
    }

    bool openSourceFmhaSupport() {
        return support_open_source_fmha_;
    }

    void runTrtV1Fmha(void* input,
                      void* cu_seqlens,
                      void* output,
                      void* qkv_buf_temp,
                      size_t batch_size,
                      size_t seq_len,
                      size_t token_num);

    void runTrtV2Fmha(void* input,
                      void* cu_seqlens,
                      void* output,
                      uint32_t* tile_counter_ptr,
                      float* attention_output_orig_quant_scale,
                      size_t batch_size,
                      size_t seq_len,
                      size_t token_num,
                      KVBlockArray kv_block_array,
                      bool mFMHAForceFP32Acc    = false,
                      bool mRemovePadding       = false,
                      bool is_alibi             = false,
                      bool is_alibi_with_sacle  = false);

    void runTrtV2FmhaPaged(void*  input,
                           void*  cu_q_seqlens,
                           void*  cu_kv_seqlens,
                           void*  output,
                           uint32_t* tile_counter_ptr,
                           size_t batch_size,
                           size_t input_seq_len,
                           size_t max_past_kv_len,
                           size_t token_num,
                           KVBlockArray kv_block_array,
                           bool mFMHAForceFP32Acc    = false,
                           bool mRemovePadding       = false,
                           bool is_alibi             = false,
                           bool is_alibi_with_sacle  = false);

    void runOpenSourceFmha(void*  q,
                           void*  k,
                           void*  v,
                           void*  output,
                           int*   cu_seqlens,
                           size_t batch_size,
                           size_t seq_len,
                           void   *workspace,
                           float* linear_bias_slopes = nullptr,
                           float softmax_extra_scale = 1.0f);

    void runOpenSourceFmhaPaged(void*  q,
                                void*  k,
                                void*  v,
                                void*  output,
                                int*   cu_seqlens,
                                int*   cu_kv_seqlens,
                                int*   block_table,
                                size_t batch_size,
                                size_t block_table_batch_stride,
                                size_t seq_size_per_block,
                                size_t seq_len,
                                void   *workspace,
                                float* linear_bias_slopes = nullptr,
                                float softmax_extra_scale = 1.0f);

    size_t getOpenSourceWorkSpaceSize(size_t batch_size,
                                      size_t seq_len_q,
                                      size_t max_seq_len_kv = 0,
                                      bool   paged = false);

    bool checkSignature(DataType dtype,
                        AttentionMaskType mtype,
                        size_t head_num,
                        size_t kv_head_num,
                        size_t size_per_head,
                        float  q_scaling,
                        bool use_linear_bias_slopes);
private:
    bool initTrtV1FmhaAndCheckSupport();

    bool initTrtV2FmhaAndCheckSupport();

    bool initTrtV2FmhaPagedAndCheckSupport();

    bool initOpenSourceFmhaAndCheckSupport();

    static int roundMultiple(int x, int m) {
        return (x + m - 1) / m * m;
    }

    int getNumSplits(size_t batch_size,
                     size_t seqlen_q,
                     size_t seqlen_k) const;

    Flash_fwd_params genFlashFwdParams(void* q,
                                       void* k,
                                       void* v,
                                       void* output,
                                       int* cu_seqlens,
                                       int* cu_kv_seqlens,
                                       void* softmax_lse,
                                       size_t batch_size,
                                       size_t seq_len_q,
                                       size_t seq_len_kv,
                                       float* linear_bias_slopes = nullptr,
                                       float softmax_extra_scale = 1.0f) const;
private:

    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> trtv2_fmha_runner_;
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> trtv2_paged_fmha_runner_;
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2Sm70> trtv2_sm70_fmha_runner_;
#ifdef USE_OLD_TRT_FMHA
    std::unique_ptr<FusedMHARunnerFP16v2> trtv1_fmha_runner_;
#endif
    DataType dtype_;
    AttentionMaskType mtype_;

    size_t head_num_;
    size_t kv_head_num_;
    size_t size_per_head_;
    float q_scaling_;
    bool use_linear_bias_slopes_;
    bool support_trt_v1_fmha_;
    bool support_trt_v2_fhma_;
    bool support_trt_v2_paged_fmha_;
    bool support_open_source_fmha_;

    cudaStream_t stream_;
};

} // namespace rtp_llm
