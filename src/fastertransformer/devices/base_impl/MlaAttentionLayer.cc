#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/utils/DevicePerfWrapper.h"
#include <numeric>

using namespace std;

namespace fastertransformer {
AttentionLayerOutput DeviceBase::mlaAttentionLayer(const AttentionLayerParams& params) {
    DevicePerfWrapper wrapper(this, "mla_layer_%d", params.layer_id);
    const auto& input            = params.input;
    const auto& input_lengths    = *params.common.input_lengths;

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
        RUNTIME_ASSERT_OP_ARG(kv_cache.k_cache_buffer && kv_cache.v_cache_buffer,
                            "kv cache buffer should has value when use kv_cache_block_id");
        const auto& k_cache_shape = kv_cache.k_cache_buffer->shape();
        const auto& v_cache_shape = kv_cache.v_cache_buffer->shape();
        RUNTIME_ASSERT_OP_ARG(
            ((k_cache_shape.size() == 3) && (v_cache_shape.size() == 3) && (k_cache_shape[0] == v_cache_shape[0])
            && (k_cache_shape[1] == v_cache_shape[1]) && (k_cache_shape[1] == params.configs.tokens_per_block)
            && (k_cache_shape[2] == params.configs.kv_lora_rank + params.configs.rope_head_dim) && (v_cache_shape[2] == 0)),
            "mla kv cache buffer check shape failed. k_cache_buffer: %s, v_cache_buffer: %s",
            kv_cache.k_cache_buffer->debugString().c_str(),
            kv_cache.v_cache_buffer->debugString().c_str());
        if (kv_cache.k_scale_buffer) {
            const auto& k_scale_shape = kv_cache.k_scale_buffer->shape();
            const auto& v_scale_shape = kv_cache.v_scale_buffer->shape();
            RUNTIME_ASSERT_OP_ARG(((k_scale_shape.size() == 2) && (v_scale_shape.size() == 3)
                                && (k_scale_shape[0] == v_scale_shape[0]) && (k_scale_shape[1] == v_scale_shape[1])
                                && (k_cache_shape[0] == k_scale_shape[0])
                                && (k_scale_shape[1] == params.configs.tokens_per_block)),
                                "kv scale check buffer failed. k_scale_buffer: %s, v_scale_buffer: %s",
                                kv_cache.k_scale_buffer->debugString().c_str(),
                                kv_cache.v_scale_buffer->debugString().c_str());
        }
    }
    BufferPtr q = nullptr;
    DevicePerfWrapper pre_mla_wrapper(this, "pre_mla_layer");
    if (params.weights.q_a_weight != nullptr) {
        // auto q_output_size = params.configs.nope_head_dim;
        BufferPtr q_a                = gemm(GemmParams(input, *(params.weights.q_a_weight->kernel)));
        layernorm(LayernormParams(q_a,
                                  q_a,
                                  mayGetRef(params.weights.q_a_norm_weight),
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  1.0f,
                                  params.ln_params.eps,
                                  true,
                                  false,
                                  params.ln_params.norm_type));
        q = gemm(GemmParams(*q_a, *(params.weights.q_b_weight->kernel)));
    } else {
        q = gemm(GemmParams(input, *(params.weights.q_weight->kernel)));
    }
    printBufferData(input, "kv_a_before_layernorm_input");
    if (params.weights.kv_a_weight->kernel->isQBuffer()) {
        printBufferData((reinterpret_cast<const QBuffer&>(*params.weights.kv_a_weight->kernel)).scales(), "params.weights.kv_a_weight_scales");
    }
    FT_LOG_DEBUG("params.weights.kv_a_weight->kernel:%s", params.weights.kv_a_weight->kernel->debugString().c_str());
    auto kv_a   = gemm(GemmParams(input, *(params.weights.kv_a_weight->kernel)));
    printBufferData(*kv_a, "kv_a_before_layernorm");
    layernorm(LayernormParams(kv_a,
                            kv_a,
                            mayGetRef(params.weights.kv_a_norm_weight),
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            1.0f,
                            params.ln_params.eps,
                            true,
                            false,
                            params.ln_params.norm_type));

    auto k_rope = gemm(GemmParams(input, *(params.weights.k_rope_weight->kernel)));
    printBufferData(*q, "q");
    printBufferData(*kv_a, "kv_a_after_layernorm");
    printBufferData(*k_rope, "k_rope");
    mlaRotaryWriteKVCache({*q, *kv_a, *k_rope, params.common, params.weights, params.configs, params.qscheme});
    printBufferData(*q, "q_after_rotary");
    printBufferData(*k_rope, "k_rope_after_rotary");
    pre_mla_wrapper.stop();
    auto      qscheme       = params.qscheme;
    auto      dtype         = input.type();
    BufferPtr attention_out = nullptr;
    if (qscheme == QScheme::Qfp8PerTensor) {
        auto scales   = params.weights.static_quant_weight->kernel;
        attention_out = BufferPtr(new QBuffer(
            allocateBuffer({DataType::TYPE_FP8_E4M3, {h_token_num, params.configs.hidden_size}}, {"attn_output"}),
            BufferPtr(new Buffer(scales->where(), scales->type(), scales->shape(), scales->data())),
            BufferPtr(new Buffer(scales->where(), scales->type(), {0}, nullptr))));
    } else {
        attention_out = allocateBuffer({dtype, {h_token_num, params.configs.hidden_size}}, {"attn_output"});
    }
    auto qkv_output = allocateBuffer({dtype, {h_token_num, params.configs.head_num * params.configs.size_per_head}}, {"qkv_output"});
    if (generate_batch_size) {
        FT_CHECK_WITH_INFO(layer_kv_cache.has_value(), "kv cache can not be null for mla attention layer");
        auto generate_q = q->view(0, generate_batch_size);
        auto generate_qkv_output = qkv_output->slice(0, generate_batch_size);
        if (qscheme == QScheme::Qfp8PerTokenBlock) {
            size_t m = generate_q.shape()[0];
            size_t m_pad = (m + 127) / 128 * 128;
            auto bmm_indices_host = allocateBuffer({DataType::TYPE_INT32, {m_pad * params.configs.head_num}, AllocationType::HOST}, {"bmm_indices_host"});
            int* index = bmm_indices_host->data<int>();
            for (int i = 0;i < m_pad * params.configs.head_num; ++i) {
                index[i] = i / params.configs.head_num;
            }
            auto bmm_indices = clone({*bmm_indices_host, AllocationType::DEVICE});

            mlaDecoderSelfAttention({params.layer_id,
                                    generate_q,
                                    generate_qkv_output,
                                    params.common,
                                    params.weights,
                                    params.configs,
                                    params.qscheme,
                                    bmm_indices});
        } else {
            mlaDecoderSelfAttention({params.layer_id,
                                    generate_q,
                                    generate_qkv_output,
                                    params.common,
                                    params.weights,
                                    params.configs,
                                    params.qscheme});
        }
    }
    if (context_batch_size) {
        // slice to get BufferPtr
        auto context_qkv_output = qkv_output->slice(generate_batch_size, context_token_num);
        auto context_q = q->view(generate_batch_size, context_token_num);
        auto context_kv_a   = kv_a->view(generate_batch_size, context_token_num);
        auto context_k_rope = k_rope->view(generate_batch_size, context_token_num);
        if (layer_kv_cache) {
            auto layer_kv_cache_block_id = layer_kv_cache->kv_cache_block_id;
            params.common.kv_cache->kv_cache_block_id =
                layer_kv_cache_block_id->slice(generate_batch_size, context_batch_size);
        }
        mlaContextAttention({params.layer_id,
                             context_q,
                             context_kv_a,
                             context_k_rope,
                             context_qkv_output,
                             params.common,
                             params.weights,
                             params.configs,
                             params.qscheme});
    }
    printBufferData(*qkv_output, "attent_projt_input");
    auto seq_len = qkv_output->shape()[0];  
    auto qkv_output_reshape = qkv_output->reshape({seq_len, params.configs.head_num, params.configs.nope_head_dim + params.configs.rope_head_dim});
    auto stride_qkv_output = Buffer2torchTensorWithStride(
        qkv_output_reshape,
        {(int64_t)seq_len, (int64_t)params.configs.head_num, (int64_t)params.configs.nope_head_dim}).reshape({(int64_t)seq_len, (int64_t)(params.configs.head_num*params.configs.nope_head_dim)}).contiguous();
    auto strid_qkv_buffer = torchTensor2Buffer(stride_qkv_output);
    auto output_gemm_params = GemmParams(*strid_qkv_buffer, *(params.weights.output_weight->kernel), nullopt, attention_out);
    loraLinear(LoraLinearParams(output_gemm_params, params.common.lora_input.out_lora_input)).output;
    printBufferData(*attention_out, "attention_out");
    return {std::move(attention_out)};
}

};  // namespace fastertransformer
