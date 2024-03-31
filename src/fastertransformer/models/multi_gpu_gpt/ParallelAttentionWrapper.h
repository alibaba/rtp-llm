#pragma once

#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "3rdparty/flash_attention2/flash.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/GemmRunner.h"
#include "src/fastertransformer/layers/LoraGemm.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class ParallelAttentionWrapper: public BaseAttentionLayer<T> {
private:
    const GptInitParameter& params_;

    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_head_num_kv_;
    const size_t local_hidden_units_;

    bool                           is_qk_buf_float_;
    std::shared_ptr<LoraGemm<T>>   lora_gemm_;
    std::shared_ptr<GemmRunner<T>> gemm_runner_;

    bool multi_block_mode_ = false;
    // for sparse
    const std::vector<int64_t> local_layer_head_num_;
    const std::vector<int64_t> local_layer_head_num_kv_;

    // fmha runner
    int              sm_ = getSMVersion();
    Flash_fwd_params flash_fwd_params_;

    bool                                              use_trt_fmha_         = false;
    bool                                              use_open_source_fmha_ = false;
    std::unique_ptr<tensorrt_llm::kernels::MHARunner> mFMHARunner;
    bool                                              mFMHAForceFP32Acc = false;
    bool                                              mRemovePadding    = true;

    void allocateBuffer() override;
    void allocateBuffer(size_t h_token_num,
                        size_t context_batch_size,
                        size_t generate_batch_size,
                        size_t seq_len,
                        size_t seq_len_with_prefix,
                        bool   allocate_qk_buf,
                        bool   multi_block_mode);
    void freeBuffer() override;

    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    // int8_mode_ == 0 means we don't use any mechanism related to INT8.
    // int8_mode_ == 1 for weight quantized only gemm for GPT
    // int8_mode_ == 2 for SmoothQuant O3 (per tensor scales)
    const float q_scaling_;
    NcclParam   tensor_para_;

protected:
    using BaseAttentionLayer<T>::allocator_;
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::sparse_;
    T*     qkv_buf_              = nullptr;
    T*     q_buf_2_              = nullptr;
    T*     k_buf_2_              = nullptr;
    T*     v_buf_2_              = nullptr;
    T*     qk_buf_               = nullptr;
    T*     partial_out_          = nullptr;
    float* partial_sum_          = nullptr;
    float* partial_max_          = nullptr;
    int*   block_counter_        = nullptr;
    float* softmax_lse_          = nullptr;
    float* qk_buf_float_         = nullptr;
    T*     qkv_buf_2_            = nullptr;
    T*     qkv_buf_3_            = nullptr;
    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;
    char*  int8_gemm_workspace_  = nullptr;
    size_t int8_gemm_ws_bytes_   = 0;
    int    max_seq_len_tile_     = 0;

    struct ContextAttentionParams {
        T const* attention_input;
        int32_t  input_seq_length;  // padded input length
        int32_t  max_past_kv_len;
        // By default, max_kv_cache_length == cyclic_kv_cache_length
        // unless each layer has different cyclic kv cache length.
        // Max cache capacity (used to allocate KV cache)
        // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        int*    cu_seqlens;
        int32_t cyclic_kv_cache_length;
        T*      context_buf;
        void*   block_pointers;
        void*   host_block_pointers;
        int32_t batch_size;
        int32_t num_tokens;
        int32_t max_blocks_per_sequence;
        bool    is_alibi;
        bool    is_alibi_with_sacle = false;
        // optional when relative position
        const T* relative_attention_bias        = nullptr;
        int      relative_attention_bias_stride = 0;
        // optional when cross attention
        T const*       cross_qkv             = nullptr;
        int32_t        cross_qkv_length      = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t        num_encoder_tokens    = 0;
    };

public:
    ParallelAttentionWrapper(const GptInitParameter& gpt_init_parameter,
                             NcclParam               tensor_para,
                             cudaStream_t            stream,
                             cublasMMWrapper*        cublas_wrapper,
                             tc::QuantAlgo           quant_algo,
                             IAllocator*             allocator,
                             bool                    is_free_buffer_after_forward,
                             bool                    is_qk_buf_float,
                             bool                    sparse = false);

    virtual ~ParallelAttentionWrapper();

    void preAllocate() override;
    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;
    void QKVGemm(const int                 h_token_num,
                 const int                 layer_id,
                 const T*                  attention_input,
                 const AttentionWeight<T>* attention_weights,
                 int*                      lora_ids,
                 int                       batch_size,
                 const int*                input_lengths);
    void
    ContextAttention(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights);
    void
    SelfAttention(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights);
    void DenseGemm(const int                 h_token_num,
                   const int                 layer_id,
                   T*                        attention_out,
                   const AttentionWeight<T>* attention_weights,
                   int*                      lora_ids,
                   int                       batch_size,
                   const int*                input_lengths);

    void Attention(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights);
    bool CheckUseFMHA() const;
    bool UseOpenSourceFMHA() const;
    bool UseTRTFMHA() const;
    bool UseMultiBlockMode() const;

    void TRTFMHA(const ContextAttentionParams& params, cudaStream_t stream);
    void OpenSourceFMHA(T*           qkv,
                        int*         cu_seqlens,
                        const int    batch_size,
                        const int    num_heads,
                        const int    num_heads_kv,
                        const int    head_size,
                        const int    max_seqlen,
                        const float  softmax_scale,
                        T*           linear_bias_slopes,
                        T*           out,
                        cudaStream_t stream);
    
    bool UseFMHA();
};

}  // namespace fastertransformer
