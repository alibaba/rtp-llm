#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include <numeric>

using namespace std;

namespace rtp_llm {
AttentionLayerOutput DeviceBase::mlaAttentionLayer(const AttentionLayerParams& params) {
    DevicePerfWrapper wrapper(this, "mla_layer_%d", params.layer_id);
    const auto&       input         = params.input;
    const auto&       input_lengths = *params.common.input_lengths;

    //    const auto& output_weight = params.weights.output_weight;

    const auto generate_batch_size = params.common.decoder_batch_size;
    const auto context_batch_size  = params.common.context_batch_size;
    const auto context_token_num   = params.common.context_token_num;
    const auto h_token_num         = context_token_num + generate_batch_size;

    RUNTIME_ASSERT_OP_ARG(!params.residual, "default attention layer impl does not support residual!");

    const auto& layer_kv_cache = params.common.kv_cache;
    if (layer_kv_cache.has_value()) {
        const auto& kv_cache          = layer_kv_cache.value();
        const auto& kv_cache_block_id = *kv_cache.kv_cache_block_id;
        const auto& shape             = kv_cache.kv_cache_block_id->shape();
        RUNTIME_ASSERT_OP_ARG(((shape.size() == 2) && (shape[0] == input_lengths.shape()[0])),
                              "kv_cache_block_id shape in attention layer should be [batch_size, block_length]"
                              ", but got %s",
                              kv_cache_block_id.debugString().c_str());
        RUNTIME_ASSERT_OP_ARG(kv_cache.k_cache_buffer, "k_cache_buffer should has value when use kv_cache_block_id");
        const auto& layer_cache_shape = kv_cache.k_cache_buffer->shape();
        RUNTIME_ASSERT_OP_ARG(((layer_cache_shape.size() == 3)
                               && (layer_cache_shape[1] == params.configs.tokens_per_block)
                               && (layer_cache_shape[2] == params.configs.kv_lora_rank + params.configs.rope_head_dim)),
                              "mla kv cache buffer check shape failed. layer_cache: %s",
                              kv_cache.k_cache_buffer->debugString().c_str());
    }
    BufferPtr         fused_qkv = nullptr;
    BufferPtr         q         = nullptr;
    int64_t           kv_offset = 0;
    DevicePerfWrapper pre_mla_wrapper(this, "pre_mla_layer");
    if (params.weights.fusedqkrope_weight != nullptr) {
        // auto q_output_size = params.configs.nope_head_dim;
        fused_qkv        = gemm(GemmParams(input, *(params.weights.fusedqkrope_weight->kernel)));
        kv_offset        = params.configs.q_lora_rank;
        auto norm_output = layernormWithStride(LayernormWithStrideParams({fused_qkv,
                                                                          mayGetRef(params.weights.q_a_norm_weight),
                                                                          params.ln_params.eps,
                                                                          params.ln_params.norm_type,
                                                                          0,
                                                                          params.configs.q_lora_rank,
                                                                          QScheme::NoQuantize,
                                                                          false}));
        q                = gemm(GemmParams(*norm_output.output, *(params.weights.q_b_weight->kernel)));
    } else {
        fused_qkv = gemm(GemmParams(input, *(params.weights.fusedqkrope_no_lora_weight->kernel)));
        kv_offset = params.configs.head_num * params.configs.size_per_head;
        q = slice(SliceParams({*fused_qkv, -1, 0, (int64_t)(params.configs.head_num * params.configs.size_per_head)}));
    }
    layernormWithStride(LayernormWithStrideParams({fused_qkv,
                                                   mayGetRef(params.weights.kv_a_norm_weight),
                                                   params.ln_params.eps,
                                                   params.ln_params.norm_type,
                                                   (size_t)kv_offset,
                                                   params.configs.kv_lora_rank,
                                                   QScheme::NoQuantize,
                                                   true}));
    pre_mla_wrapper.stop();
    auto dtype = input.type();
    auto qkv_output =
        allocateBuffer({dtype, {h_token_num, params.configs.head_num * params.configs.v_head_dim}}, {"qkv_output"});
    if (generate_batch_size) {
        RTP_LLM_LOG_DEBUG("absorb decode mla attention");
        RTP_LLM_CHECK_WITH_INFO(layer_kv_cache.has_value(), "kv cache can not be null for mla attention layer");
        auto generate_q          = q->view(0, generate_batch_size);
        auto generate_fused_qkv  = fused_qkv->view(0, generate_batch_size);
        auto generate_qkv_output = qkv_output->slice(0, generate_batch_size);
        mlaAbsorbAttention({params.layer_id,
                            generate_q,
                            generate_fused_qkv,
                            kv_offset,
                            generate_qkv_output,
                            params.common,
                            params.weights,
                            params.configs,
                            params.qscheme});
    }

    if (context_batch_size) {
        bool use_absorb_attention = params.common.max_prefix_length > 0;
        if (use_absorb_attention) {
            RTP_LLM_LOG_DEBUG("absorb context mla attention");
            RTP_LLM_CHECK_WITH_INFO(layer_kv_cache.has_value(), "kv cache can not be null for mla attention layer");
            auto generate_q          = q->view(generate_batch_size, context_token_num);
            auto generate_fused_qkv  = fused_qkv->view(generate_batch_size, context_token_num);
            auto generate_qkv_output = qkv_output->slice(generate_batch_size, context_token_num);
            if (layer_kv_cache) {
                auto layer_kv_cache_block_id = layer_kv_cache->kv_cache_block_id;
                params.common.kv_cache->kv_cache_block_id =
                    layer_kv_cache_block_id->slice(generate_batch_size, context_batch_size);
            }
            mlaAbsorbAttention({params.layer_id,
                                generate_q,
                                generate_fused_qkv,
                                kv_offset,
                                generate_qkv_output,
                                params.common,
                                params.weights,
                                params.configs,
                                params.qscheme,
                                params.compute_type,
                                true});
        } else {
            RTP_LLM_LOG_DEBUG("no absorb context mla attention");
            // slice to get BufferPtr
            auto context_qkv_output = qkv_output->slice(generate_batch_size, context_token_num);
            auto context_fused_qkv  = fused_qkv->slice(generate_batch_size, context_token_num);
            auto context_q          = q->view(generate_batch_size, context_token_num);
            if (layer_kv_cache) {
                auto layer_kv_cache_block_id = layer_kv_cache->kv_cache_block_id;
                params.common.kv_cache->kv_cache_block_id =
                    layer_kv_cache_block_id->slice(generate_batch_size, context_batch_size);
            }
            mlaContextAttention({params.layer_id,
                                 context_q,
                                 *context_fused_qkv,
                                 kv_offset,
                                 context_qkv_output,
                                 params.common,
                                 params.weights,
                                 params.configs,
                                 params.qscheme});
        }
    }

    printBufferData(*qkv_output, "attent_proj_input");
    auto output_gemm_params = GemmParams(*qkv_output,
                                         *(params.weights.output_weight->kernel),
                                         std::nullopt,
                                         nullptr,
                                         DataType::TYPE_INVALID,
                                         params.compute_type);
    auto attention_out =
        loraLinear(LoraLinearParams(output_gemm_params, params.common.lora_input.out_lora_input)).output;
    printBufferData(*attention_out, "attention_out");
    return {std::move(attention_out)};
}

};  // namespace rtp_llm
