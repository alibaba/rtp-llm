#pragma once

#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "3rdparty/flash_attention2/flash.h"
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"

#include "src/fastertransformer/core/Types.h"


namespace fastertransformer{

class cufmha {

public:
    cufmha() = default;
    ~cufmha() = default;
    
    void init(cudaStream_t stream) {
        stream_ = stream;
    }

    void setup(DataType dtype,
               AttentionMaskType mtype,
               size_t head_num,
               size_t kv_head_num,
               size_t size_per_head,
               float q_scaling)
    {
        dtype_ = dtype;
        mtype_ = mtype;
        head_num_ = head_num;
        kv_head_num_ = kv_head_num;
        size_per_head_ = size_per_head;
        q_scaling_ = q_scaling;
    }


    bool trtV1FmhaSupport();
    
    bool trtV2FmhaSupport();

    bool openSourceFmhaSupport();

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
                      size_t batch_size, 
                      size_t seq_len,
                      size_t token_num,
                      bool mFMHAForceFP32Acc    = false,
                      bool mRemovePadding       = false,
                      bool is_alibi             = false,
                      bool is_alibi_with_sacle  = false);

    void runOpenSourceFmha(void* q,
                           void* k,
                           void* v,
                           void* output,
                           int* cu_seqlens,
                           void* softmax_lse_,
                           size_t token_num,
                           size_t batch_size,
                           size_t seq_len,
                           void* linear_bias_slopes = nullptr);

private:

    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> trtv2_fmha_runner_;
#ifdef USE_OLD_TRT_FMHA
    std::unique_ptr<FusedMHARunnerFP16v2> trtv1_fmha_runner_;
#endif
    DataType dtype_;
    AttentionMaskType mtype_;

    size_t head_num_;
    size_t kv_head_num_;
    size_t size_per_head_;
    float q_scaling_;

    cudaStream_t stream_;
};

} // namespace fastertransformer