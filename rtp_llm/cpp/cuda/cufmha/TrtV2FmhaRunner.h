#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/cuda/cufmha/TRTAttn.h"
#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttentionSm70/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"

namespace tensorrt_llm {
namespace kernels {
struct MHARunnerFixedParams;
struct MHARunnerParams;
class FusedMHARunnerV2;
class FusedMHARunnerV2Sm70;
}  // namespace kernels
}  // namespace tensorrt_llm

namespace rtp_llm {

struct KVBlockArray;

struct TrtV2FmhaRunnerConfig {
    size_t            head_num;
    size_t            kv_head_num;
    size_t            size_per_head;
    size_t            tokens_per_block;
    AttentionMaskType mask_type;
    float             q_scaling;
    float             softmax_extra_scale;

    // 从 AttentionConfigs 创建配置
    static TrtV2FmhaRunnerConfig fromAttentionConfigs(const AttentionConfigs& configs) {
        return TrtV2FmhaRunnerConfig{configs.head_num,
                                     configs.kv_head_num,
                                     configs.size_per_head,
                                     configs.tokens_per_block,
                                     configs.mask_type,
                                     configs.q_scaling,
                                     configs.softmax_extra_scale};
    }
};

class TrtV2FmhaRunner {
public:
    TrtV2FmhaRunner(const TrtV2FmhaRunnerConfig& config, DataType attn_dtype, bool is_s_padded, cudaStream_t stream);

    ~TrtV2FmhaRunner() = default;

    bool trtV2FmhaSupported() const {
        return support_trt_v2_fmha_;
    }

    bool trtV2PagedFmhaSupported() const {
        return support_trt_v2_paged_fmha_;
    }

    void runTrtV2Fmha(void*        input,
                      void*        cu_seqlens,
                      void*        output,
                      uint32_t*    tile_counter_ptr,
                      float*       attention_output_orig_quant_scale,
                      size_t       batch_size,
                      size_t       max_seq_len,
                      size_t       token_num,
                      KVBlockArray kv_block_array,
                      void*        custom_mask = nullptr);

    void runTrtV2FmhaPaged(void*        input,
                           void*        cu_q_seqlens,
                           void*        cu_kv_seqlens,
                           void*        output,
                           uint32_t*    tile_counter_ptr,
                           float*       attention_output_orig_quant_scale,
                           size_t       batch_size,
                           size_t       max_input_seq_len,
                           size_t       max_past_kv_len,
                           size_t       token_num,
                           size_t       token_num_kv,
                           KVBlockArray kv_block_array,
                           void*        custom_mask = nullptr);

private:
    bool initTrtV2FmhaAndCheckSupport();

    bool initTrtV2FmhaPagedAndCheckSupport();

    tensorrt_llm::kernels::MHARunnerFixedParams createMHARunnerFixedParams(bool paged);

    tensorrt_llm::kernels::MHARunnerParams createMHARunnerParams(void*        input,
                                                                 void*        cu_seqlens,
                                                                 void*        cu_kv_seqlens,
                                                                 void*        output,
                                                                 uint32_t*    tile_counter_ptr,
                                                                 float*       attention_output_orig_quant_scale,
                                                                 size_t       batch_size,
                                                                 size_t       max_input_length,
                                                                 size_t       max_kv_length,
                                                                 size_t       total_q_seq_len,
                                                                 size_t       total_kv_seq_len,
                                                                 KVBlockArray kv_block_array,
                                                                 void*        custom_mask = nullptr);

private:
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2>     trtv2_fmha_runner_;
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2>     trtv2_paged_fmha_runner_;
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2Sm70> trtv2_sm70_fmha_runner_;

    TrtV2FmhaRunnerConfig config_;
    DataType              attn_dtype_;
    bool                  is_s_padded_;
    float                 q_scaling_;
    bool                  support_trt_v2_fmha_;
    bool                  support_trt_v2_paged_fmha_;
    cudaStream_t          stream_;
};

std::shared_ptr<TRTAttn> prepareTrtAttnParams(const AttentionConfigs& configs,
                                              const BufferPtr&        kv_cache_block_id,
                                              int                     batch_size,
                                              bool                    use_fp8_fmha,
                                              cudaStream_t            stream,
                                              bool                    enable_paged_trt_v2 = false);

}  // namespace rtp_llm
