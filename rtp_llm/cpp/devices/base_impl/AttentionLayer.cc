#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

#include <numeric>

using namespace std;

namespace rtp_llm {

BufferPtr DeviceBase::attentionQKVGemm(const AttentionLayerParams& params) {
    const auto token_num     = params.input.shape()[0];
    const auto pad_token_num = params.enable_sp ? params.pad_token_num : token_num;

    BufferPtr qkv = nullptr;
    if (params.enable_sp && params.layer_id > 0) {
        BufferPtr ag_recv_buffer = nullptr;
        BufferPtr attn_input_ptr = nullptr;
        printBufferData(*attn_input_ptr, "attn_ag_input");

        if (params.qscheme == NoQuantize || params.qscheme == Qfp8PerToken) {
            attn_input_ptr = params.input.slice(0, params.input.shape()[0]);
            ag_recv_buffer = allocateBuffer({attn_input_ptr->type(), {pad_token_num, attn_input_ptr->shape()[1]}},
                                            {"ag_recv_buffer"});
        } else if (params.qscheme == Qint8PerToken) {
            attn_input_ptr   = reinterpret_cast<const QBuffer&>(params.input).qslice(0, params.input.shape()[0]);
            BufferPtr kernel = allocateBuffer({attn_input_ptr->type(), {pad_token_num, attn_input_ptr->shape()[1]}},
                                              {"ag_recv_buffer_kernel"});
            BufferPtr scales = allocateBuffer({DataType::TYPE_FP32, {pad_token_num}, AllocationType::DEVICE},
                                              {"ag_recv_buffer_scale"});
            ag_recv_buffer   = BufferPtr(new QBuffer(
                std::move(kernel),
                std::move(scales),
                std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        } else if (params.qscheme == Qfp8PerTensor) {
            attn_input_ptr = reinterpret_cast<const QBuffer&>(params.input).qslicePerTensor(0, params.input.shape()[0]);
            BufferPtr kernel = allocateBuffer({attn_input_ptr->type(), {pad_token_num, attn_input_ptr->shape()[1]}},
                                              {"ag_recv_buffer_kernel"});
            BufferPtr scales = reinterpret_cast<const QBuffer&>(params.input).scalesPtr();
            ag_recv_buffer   = BufferPtr(new QBuffer(
                std::move(kernel),
                std::move(scales),
                std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
        } else {
            throw OpException({OpErrorType::ERROR_UNIMPLEMENTED, "allGatherloraLinear qscheme type not supported"});
        }
        GemmParams                qkv_gemm_params = GemmParams(*ag_recv_buffer,
                                                *(params.weights.qkv_weight->kernel),
                                                std::nullopt,
                                                nullptr,
                                                DataType::TYPE_INVALID,
                                                params.compute_type);
        AllGatherLoraLinearOutput all_gather_output =
            allGatherloraLinear({LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input),
                                 attn_input_ptr,
                                 ag_recv_buffer,
                                 params.qscheme,
                                 params.output->type()});
        // syncAndCheck();
        ag_recv_buffer = all_gather_output.all_gather_recv_buffer;
        qkv            = all_gather_output.output;
        printBufferData(*ag_recv_buffer, "attn_ag_inter_output");
        printBufferData(*qkv, "attn_ag_final_output");
    } else {
        // NOTE: Cuda implementation fused adding qkv_weight->bias in invokeAddFusedQKVBiasTranspose kernel call.
        // other devices need to be careful about this.
        // maybe add a device property here.
        qkv = params.configs.use_mla ? mlaQKVGemm(params) : mhaQKVGemm(params);
    }
    printBufferData(*qkv, "qkv");
    return qkv;
}

BufferPtr DeviceBase::attentionAttn(const AttentionLayerParams& params) {
    const auto& input_lengths    = *params.common.input_lengths;
    const auto& sequence_lengths = *params.common.sequence_lengths;

    const auto generate_batch_size = sequence_lengths.shape()[0];
    const auto context_batch_size  = input_lengths.shape()[0] - generate_batch_size;
    const auto context_token_num   = params.common.context_token_num;
    const auto pad_token_num = params.enable_sp ? params.pad_token_num : (context_token_num + generate_batch_size);

    const auto& layer_kv_cache = params.common.kv_cache;
    if (layer_kv_cache) {
        const auto& kv_cache          = layer_kv_cache.value();
        const auto& kv_cache_block_id = *kv_cache.kv_cache_block_id;
        const auto& shape             = kv_cache.kv_cache_block_id->shape();
        RUNTIME_ASSERT_OP_ARG(((shape.size() == 2) && (shape[0] == input_lengths.shape()[0])),
                              "kv_cache_block_id shape in attention layer should be [batch_size, block_length]"
                              ", but got %s",
                              kv_cache_block_id.debugString().c_str());
        RUNTIME_ASSERT_OP_ARG(kv_cache.k_cache_buffer, "kv cache buffer should has value when use kv_cache_block_id");
        const auto& layer_cache_shape = kv_cache.k_cache_buffer->shape();
        RUNTIME_ASSERT_OP_ARG(((layer_cache_shape.size() == 5) && (layer_cache_shape[1] == 2)
                               && (layer_cache_shape[2] == params.configs.kv_head_num)
                               && (layer_cache_shape[3] == params.configs.tokens_per_block)
                               && (layer_cache_shape[4] == params.configs.size_per_head)),
                              "kv cache buffer check shape failed. k_cache_buffer: %s",
                              kv_cache.k_cache_buffer->debugString().c_str());
    }

    const auto qkv_hidden_size = params.configs.head_num * params.configs.size_per_head;
    auto       qscheme         = params.qscheme;
    auto&      qkv             = params.input;
    BufferPtr  qkv_output      = nullptr;
    if (qscheme == QScheme::Qfp8PerTensor) {
        auto scales = params.weights.static_quant_weight->kernel;
        qkv_output  = BufferPtr(
            new QBuffer(allocateBuffer({DataType::TYPE_FP8_E4M3, {pad_token_num, qkv_hidden_size}}, {"qkv_output"}),
                        BufferPtr(new Buffer(scales->where(), scales->type(), scales->shape(), scales->data())),
                        BufferPtr(new Buffer(scales->where(), scales->type(), {0}, nullptr))));
    } else {
#if defined(__aarch64__)
        // Arm attention op only support fp32 data type
        qkv_output = allocateBuffer({DataType::TYPE_FP32, {pad_token_num, qkv_hidden_size}}, {"qkv_output"});
#else
        qkv_output = allocateBuffer({qkv.type(), {pad_token_num, qkv_hidden_size}}, {"qkv_output"});
#endif
    }

    auto kv_cache_block_id = layer_kv_cache ? layer_kv_cache->kv_cache_block_id : nullptr;
    if (generate_batch_size) {
        auto generate_qkv    = qkv.view(0, generate_batch_size);
        auto generate_output = qkv_output->view(0, generate_batch_size);
        if (layer_kv_cache) {
            params.common.kv_cache->kv_cache_block_id = kv_cache_block_id->slice(0, generate_batch_size);
        }
        decoderSelfAttention({params.layer_id,
                              generate_qkv,
                              generate_output,
                              params.common,
                              params.weights,
                              params.configs,
                              params.qscheme,
                              params.compute_type});
    }
    if (context_batch_size) {
        auto context_qkv    = qkv.view(generate_batch_size, context_token_num);
        auto context_output = qkv_output->view(generate_batch_size, context_token_num);
        if (layer_kv_cache) {
            params.common.kv_cache->kv_cache_block_id =
                kv_cache_block_id->slice(generate_batch_size, context_batch_size);
        }
        contextAttention({params.layer_id,
                          context_qkv,
                          context_output,
                          params.common,
                          params.weights,
                          params.configs,
                          params.qscheme,
                          params.compute_type});
    }
    if (layer_kv_cache) {
        params.common.kv_cache->kv_cache_block_id = kv_cache_block_id;
    }
    printBufferData(*qkv_output, "qkv_output");
    return qkv_output;
}

BufferPtr DeviceBase::attentionOutGemm(const AttentionLayerParams& params) {
    auto&       qkv_output    = params.input;
    BufferPtr   gemm_output   = nullptr;
    BufferPtr   attn_output   = nullptr;
    const auto& output_weight = params.weights.output_weight;
    if (params.enable_sp) {
        gemm_output = allocateBuffer({qkv_output.type(), {params.pad_token_num, output_weight->kernel->shape()[1]}},
                                     {"attn_layer_out"});
        attn_output = params.output;
    } else if (params.output) {
        gemm_output = params.output;
        attn_output = gemm_output;
    }

    BufferPtr quanted_attn_input = nullptr;
    if (params.qscheme == QScheme::Qint8PerTensor || params.qscheme == QScheme::Qint8PerToken) {
        OptionalConstBufferRef smoother_weight =
            params.weights.smoother_weight ? (OptionalConstBufferRef) * (params.weights.smoother_weight->kernel) :
                                             std::nullopt;

        OptionalConstBufferRef shift_weight = (params.weights.shift_weight == nullptr) ?
                                                  nullopt :
                                                  (OptionalConstBufferRef)*params.weights.shift_weight->kernel;

        OptionalConstBufferRef static_scale_weight =
            params.weights.static_quant_weight ?
                (OptionalConstBufferRef) * (params.weights.static_quant_weight->kernel) :
                std::nullopt;

        OptionalConstBufferRef static_scale_reciprocal_weight =
            params.weights.static_scale_reciprocal_weight ?
                (OptionalConstBufferRef) * (params.weights.static_scale_reciprocal_weight->kernel) :
                std::nullopt;
        auto quant_data_type = DataType::TYPE_INT8;
        auto quant_params    = QuantizeParams(qkv_output,
                                           quant_data_type,
                                           1,
                                           params.qscheme,
                                           smoother_weight,
                                           shift_weight,
                                           static_scale_weight,
                                           static_scale_reciprocal_weight);

        quanted_attn_input = quantize(quant_params);
    }
    auto& output_gemm_input = quanted_attn_input ? *quanted_attn_input : qkv_output;
#if defined(__aarch64__)
    // Arm attention op only support fp32 data type, convert to original dtype
    GemmParams output_gemm_params =
        GemmParams(output_gemm_input, *(output_weight->kernel), nullopt, gemm_output, gemm_output->type());
#else
    GemmParams output_gemm_params = GemmParams(
        output_gemm_input, *(output_weight->kernel), nullopt, gemm_output, DataType::TYPE_INVALID, params.compute_type);
#endif
    if (params.enable_sp) {
        printBufferData(output_gemm_input, "attn_rs_input");
        ReduceScatterLoraLinearOutput reduce_scatter_output =
            loraLinearReduceScatter({LoraLinearParams(output_gemm_params, params.common.lora_input.out_lora_input),
                                     params.output,
                                     params.qscheme,
                                     params.output->type()});
        // syncAndCheck();
        gemm_output = reduce_scatter_output.reduce_scatter_recv_buffer;
        attn_output = reduce_scatter_output.output;
        printBufferData(*gemm_output, "attn_rs_inter_output");
        printBufferData(*attn_output, "attn_rs_final_output");
    } else {
        attn_output = loraLinear(LoraLinearParams(output_gemm_params, params.common.lora_input.out_lora_input)).output;
    }
    return {std::move(attn_output)};
}

AttentionLayerOutput DeviceBase::attentionLayer(const AttentionLayerParams& params) {
    auto      qkv      = attentionQKVGemm(params);
    BufferPtr attn_out = attentionAttn({params.layer_id,
                                        *qkv,
                                        params.output,
                                        params.configs,
                                        params.weights,
                                        params.common,
                                        params.residual,
                                        params.ln_params,
                                        params.qscheme,
                                        params.compute_type,
                                        params.enable_sp,
                                        params.pad_token_num});
    return {attentionOutGemm({params.layer_id,
                              *attn_out,
                              params.output,
                              params.configs,
                              params.weights,
                              params.common,
                              params.residual,
                              params.ln_params,
                              params.qscheme,
                              params.compute_type,
                              params.enable_sp,
                              params.pad_token_num})};
}

};  // namespace rtp_llm
