#include "rtp_llm/cpp/models/FfnDisaggregateModel.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

using namespace rtp_llm;

namespace rtp_llm {

GptModelOutputs FfnDisaggregateModel::forward(const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "forward [tp=%d, dp=%d]", device_props_.tp_size, device_props_.dp_size);
    RTP_LLM_LOG_DEBUG("FfnDisaggregateModel forward");
    if (!device_props_.ffn_as_service) {
        return forwardNormal(inputs);
    } else {

        return forwardFFnService(inputs);
    }
}

// for last rank
GptModelOutputs FfnDisaggregateModel::forwardFFnService(const GptModelInputs& inputs) {
    auto hidden_dtype = description_.data_type;
    RTP_LLM_LOG_DEBUG("FfnDisaggregateModel forwardFFnService");
    // hidden buffer [micro_batch_size[i], hidden_size]
    std::vector<BufferPtr> hidden_buffers;
    // recv buffer: attn output buffer [micro_batch_size[i], q_head * head_size]
    std::vector<BufferPtr> recv_buffers;
    // send buffer: attn qkv buffer [micro_batch_size[i], (q_head + kv_head * 2) * head_size]
    std::vector<BufferPtr> send_buffers;
    // recv buffer should be pre-allocated
    auto batch_split_info = recvBatchSplitInfo();
    for (int i = 0; i < micro_batch_size_; i++) {
        RTP_LLM_CHECK_WITH_INFO(batch_split_info.micro_batch_sizes_list[i].size() == device_props_.dp_size - 1,
                                "batch_split_info.micro_batch_sizes_list[%d].size() != %d",
                                i,
                                device_props_.dp_size - 1);
        recv_buffers.push_back(
            device_->allocateBuffer({hidden_dtype,
                                     {batch_split_info.total_micro_batch_sizes[i],
                                      description_.attention_conf.head_num * description_.attention_conf.size_per_head},
                                     AllocationType::DEVICE}));
        hidden_buffers.push_back(device_->allocateBuffer(
            {hidden_dtype,
             {batch_split_info.total_micro_batch_sizes[i], description_.attention_conf.hidden_size},
             AllocationType::DEVICE}));
    }
    RTP_LLM_LOG_DEBUG("batch_split_info: %s", batch_split_info.debugInfo().c_str());
    recvInputBuffer(batch_split_info, hidden_buffers);
    RTP_LLM_CHECK_WITH_INFO(batch_split_info.micro_batch_sizes_list.size() == micro_batch_size_
                                && batch_split_info.total_micro_batch_sizes.size() == micro_batch_size_,
                            "batch_split_info.micro_batch_sizes_list.size() != %d",
                            micro_batch_size_);

    // handle first layer pre attention operation seperately
    send_buffers.resize(micro_batch_size_, nullptr);
    for (int batch_idx = 0; batch_idx < micro_batch_size_; batch_idx++) {
        send_buffers[batch_idx] = preAttentionOperation(hidden_buffers[batch_idx], 0);
        BatchSendRecvParams send_recv_params;
        for (int i = 0; i < batch_split_info.micro_batch_sizes_list[batch_idx].size(); i++) {
            auto buf = send_buffers[batch_idx]->slice(0, batch_split_info.micro_batch_sizes_list[batch_idx][i]);
            send_recv_params.p2p_params.push_back(P2pSendRecvParams{SendRecvType::kSend, buf, (int)i});
        }
        // pre-recv attention output buffer
        if (batch_idx > 0) {
            auto recv_idx    = batch_idx - 1;
            auto recv_buffer = recv_buffers[recv_idx];
            auto offset      = 0;
            for (int j = 0; j < batch_split_info.micro_batch_sizes_list[recv_idx].size(); j++) {
                auto buf = recv_buffer->slice(offset, batch_split_info.micro_batch_sizes_list[recv_idx][j]);
                send_recv_params.p2p_params.push_back(P2pSendRecvParams{SendRecvType::kRecv, buf, (int)j});
                offset += batch_split_info.micro_batch_sizes_list[recv_idx][j];
            }
        }
        device_->batchSendRecv(send_recv_params, ParallelMode::DP_AND_TP);
    }

    for (int i = 0; i < layer_num_; i++) {
        for (int batch_idx = 0; batch_idx < micro_batch_size_; batch_idx++) {
            // attention out gemm
            const auto& layer                   = weights_.layers[i];
            auto        attention_common_inputs = AttentionCommonInputs();
            auto        attn_out_params         = AttentionLayerParams({i,
                                                                        *recv_buffers[batch_idx],
                                                                        nullptr,
                                                                        description_.attention_conf,
                                                                        weights_.layers[i].self_attention_weights,
                                                                        attention_common_inputs,
                                                                        nullopt,
                                                                        {description_.layernorm_eps, description_.norm_type},
                                                                        description_.act_qscheme,
                                                                        false,
                                                                        0});
            auto        attn_out                = device_->attentionOutGemm(attn_out_params);
            // post layernorm
            auto post_layernorm_params =
                LayernormParams(attn_out,
                                attn_out,
                                rtp_llm::mayGetRef(layer.post_layernorm),
                                *hidden_buffers[batch_idx],
                                nullopt,
                                rtp_llm::mayGetRef(layer.self_attention_weights.output_weight->bias),
                                0.f,
                                description_.layernorm_eps,
                                false,
                                description_.post_layernorm,
                                description_.norm_type,
                                description_.act_qscheme,
                                false,
                                true);
            auto post_layernorm_output = device_->layernorm(post_layernorm_params);
            auto hidden                = post_layernorm_output.output;
            auto residual              = post_layernorm_output.before_norm_output;
            device_->checkError();
            auto ffn_params = FfnLayerParams(
                {*hidden, description_.ffn_conf, layer.ffn_weights, nullopt, description_.act_qscheme, nullptr, false});
            auto ffn_output = device_->ffnLayer(ffn_params);
            device_->checkError();
            printBufferData(*ffn_output.hidden_states, "ffn service ffn layer after ffn");
            printBufferData(*residual, "ffn service ffn residual");
            auto ffn_layernorm_output = device_->layernorm(
                LayernormParams(ffn_output.hidden_states,
                                ffn_output.hidden_states,
                                rtp_llm::mayGetRef(layer.post_ffn_layernorm),
                                *residual,
                                nullopt,
                                rtp_llm::mayGetRef(WEIGHT_MAY_GET_BIAS(layer.ffn_weights.down_weight)),
                                1.0f,
                                description_.layernorm_eps,
                                true,
                                description_.post_layernorm,
                                description_.norm_type,
                                QScheme::NoQuantize));
            device_->checkError();
            hidden_buffers[batch_idx] = ffn_layernorm_output.output;
            printBufferData(*hidden_buffers[batch_idx], "ffn service ffn layer after ffn layernorm");
            if (i != layer_num_ - 1) {
                auto res                = preAttentionOperation(ffn_layernorm_output.output, i + 1);
                send_buffers[batch_idx] = res;
            }
            // send result to attn service
            auto batch_send_recv_params = BatchSendRecvParams();
            auto send_offset            = 0;
            auto recv_offset            = 0;
            // recv
            if (!(i == layer_num_ - 1 && batch_idx > 0)) {
                auto recv_idx = batch_idx == 0 ? micro_batch_size_ - 1 : batch_idx - 1;
                for (int j = 0; j < batch_split_info.micro_batch_sizes_list[recv_idx].size(); j++) {
                    // (int)batch_split_info.micro_batch_sizes_list[batch_idx][j]);
                    auto recv_buffer = recv_buffers[recv_idx]->slice(
                        recv_offset, batch_split_info.micro_batch_sizes_list[recv_idx][j]);
                    batch_send_recv_params.p2p_params.push_back(
                        P2pSendRecvParams{SendRecvType::kRecv, recv_buffer, (int)j});
                    // printBufferData_(*recv_buffer, "forwardffnserverice recv buffer", nullptr, true);
                    recv_offset += batch_split_info.micro_batch_sizes_list[recv_idx][j];
                }
            }
            // send
            {
                auto sendback_buffer = i == layer_num_ - 1 ? hidden_buffers[batch_idx] : send_buffers[batch_idx];
                for (int j = 0; j < batch_split_info.micro_batch_sizes_list[batch_idx].size(); j++) {
                    auto buf =
                        sendback_buffer->slice(send_offset, batch_split_info.micro_batch_sizes_list[batch_idx][j]);
                    batch_send_recv_params.p2p_params.push_back(P2pSendRecvParams{SendRecvType::kSend, buf, (int)j});
                    // printBufferData_(*buf, "forwardffnserverice send buffer", nullptr, true);
                    send_offset += batch_split_info.micro_batch_sizes_list[batch_idx][j];
                }
            }
            device_->batchSendRecv(batch_send_recv_params, ParallelMode::DP_AND_TP);
        }
    }
    device_->checkError();
    // send last micro batch result to others
    return {nullptr, nullptr};
}

BufferPtr FfnDisaggregateModel::preAttentionOperation(BufferPtr input, int layer_id) {
    // here hidden->dtype maybe int8, so use dytpe of embedding lookup result instead
    auto        hidden       = input;
    const auto& layer        = weights_.layers[layer_id];
    auto        attn_out_buf = device_->prepareAllReduce({std::move(input), ReduceOp::Sum}).buffer;
    auto        residual     = hidden;
    if (device_->initParams().use_deepep_moe) {
        // avoid attention o gemm copy
        attn_out_buf.reset();
    }
    printBufferData(*residual, "in residual");
    BufferPtr residual2         = nullptr;
    BufferPtr hidden_to_slice   = nullptr;  // for sp and overlap comm type 2
    BufferPtr last_layer_hidden = nullptr;
    if (layer.pre_layernorm) {
        residual                  = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
        auto pre_layernorm_output = device_->layernorm(LayernormParams(hidden,
                                                                       residual,
                                                                       *layer.pre_layernorm,
                                                                       nullopt,
                                                                       nullopt,
                                                                       std::nullopt,
                                                                       0.f,
                                                                       description_.layernorm_eps,
                                                                       false,
                                                                       false,
                                                                       description_.norm_type,
                                                                       description_.act_qscheme,
                                                                       false,
                                                                       false));

        hidden = std::move(pre_layernorm_output.output);
    }
    printBufferData(*hidden, "pre layer norm hidden");
    auto attention_common_inputs = AttentionCommonInputs();
    auto attn_params             = AttentionLayerParams({0,
                                                         *hidden,
                                                         move(attn_out_buf),
                                                         description_.attention_conf,
                                                         layer.self_attention_weights,
                                                         attention_common_inputs,
                                                         nullopt,
                                                         {description_.layernorm_eps, description_.norm_type},
                                                         description_.act_qscheme,
                                                         false,
                                                         0});
    auto qkv_output              = device_->attentionQKVGemm(attn_params);
    device_->checkError();
    return qkv_output;
}

// for rank 0->dp_size-2
GptModelOutputs FfnDisaggregateModel::forwardNormal(const GptModelInputs& inputs) {
    RTP_LLM_LOG_DEBUG("FfnDisaggregateModel forwardNormal");
    auto model_input              = forwardPreLayers(inputs);
    auto micro_batch_layer_inputs = model_input.micro_batch_inputs;
    RTP_LLM_CHECK_WITH_INFO(micro_batch_layer_inputs.size() == micro_batch_size_,
                            "micro_batch_inputs.size:[%d] != %d",
                            micro_batch_layer_inputs.size(),
                            micro_batch_size_);
    sendBatchSplitInfo(micro_batch_layer_inputs);
    sendInputBuffer(micro_batch_layer_inputs);
    std::vector<BufferPtr> attn_input_buffer;
    auto attn_size = (description_.attention_conf.head_num + description_.attention_conf.kv_head_num * 2)
                     * description_.attention_conf.size_per_head;
    for (int i = 0; i < micro_batch_size_; i++) {
        attn_input_buffer.push_back(
            device_->allocateBuffer({description_.data_type,
                                     {micro_batch_layer_inputs[i].hidden->shape()[0], attn_size},
                                     AllocationType::DEVICE}));
    }
    for (int i = 0; i < layer_num_; i++) {
        for (int batch_idx = 0; batch_idx < micro_batch_size_; batch_idx++) {
            auto& attention_common_inputs = micro_batch_layer_inputs[batch_idx].attention_common_inputs;
            if (attention_common_inputs.kv_cache) {
                attention_common_inputs.kv_cache->k_cache_buffer = k_cache_buffer_->index(i);
                attention_common_inputs.kv_cache->v_cache_buffer = v_cache_buffer_->index(i);
                if (k_scale_buffer_) {
                    attention_common_inputs.kv_cache->k_scale_buffer = k_scale_buffer_->index(i);
                    attention_common_inputs.kv_cache->v_scale_buffer = v_scale_buffer_->index(i);
                }
            }
            if (i == 0 && batch_idx == 0) {
                BatchSendRecvParams recv_params;
                recv_params.p2p_params.push_back(
                    P2pSendRecvParams{SendRecvType::kRecv, attn_input_buffer[i], (int)device_props_.dp_size - 1});
                device_->batchSendRecv(recv_params, ParallelMode::DP_AND_TP);
            }
            device_->checkError();
            auto attn_output = device_->attentionAttn(
                AttentionLayerParams({i,
                                      *attn_input_buffer[batch_idx],
                                      nullptr,
                                      description_.attention_conf,
                                      weights_.layers[i].self_attention_weights,
                                      micro_batch_layer_inputs[batch_idx].attention_common_inputs,
                                      nullopt,
                                      {description_.layernorm_eps, description_.norm_type},
                                      description_.act_qscheme,
                                      false,
                                      0}));
            device_->checkError();
            // send current output and get next circle input
            BatchSendRecvParams send_recv_params;
            send_recv_params.p2p_params.push_back(
                P2pSendRecvParams{SendRecvType::kSend, attn_output, (int)device_props_.dp_size - 1});
            // printBufferData_(*attn_output, "forwardnormal send", nullptr, true);
            if (!(i == layer_num_ - 1 && batch_idx == micro_batch_size_ - 1)) {
                auto recv_idx    = batch_idx == micro_batch_size_ - 1 ? 0 : batch_idx + 1;
                auto recv_buffer = attn_input_buffer[recv_idx];
                send_recv_params.p2p_params.push_back(
                    P2pSendRecvParams{SendRecvType::kRecv, recv_buffer, (int)device_props_.dp_size - 1});
                // printBufferData_(*recv_buffer, "forwardnormal recv", nullptr, true);
            }
            device_->batchSendRecv(send_recv_params, ParallelMode::DP_AND_TP);
        }
    }
    // get final output
    {
        BatchSendRecvParams recv_params;
        for (int i = 0; i < micro_batch_size_; i++) {
            auto recv_buffer = micro_batch_layer_inputs[i].hidden;
            recv_params.p2p_params.push_back(
                P2pSendRecvParams{SendRecvType::kRecv, recv_buffer, (int)device_props_.dp_size - 1});
            printBufferData(*recv_buffer, "micro batch hidden: " + std::to_string(i));
        }
        device_->batchSendRecv(recv_params, ParallelMode::DP_AND_TP);
    }

    auto outputs = forwardPostLayers(model_input.hidden,
                                     inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                                     inputs.need_all_logits,
                                     inputs.lm_output_indexes,
                                     model_input.enable_sp,
                                     model_input.token_num,
                                     inputs,
                                     nullptr);
    // make sure cpu buffers out lives gpu exec
    outputs.captured_values = make_shared<GptLayerInputs>(model_input);
    return outputs;
}

// TODO: some check
void FfnDisaggregateModel::recvInputBuffer(const BatchSplitInfo&   batch_split_info,
                                           std::vector<BufferPtr>& input_batch_buffers) {
    BatchSendRecvParams recv_params;
    for (int i = 0; i < micro_batch_size_; i++) {
        auto offset = 0;
        for (int j = 0; j < batch_split_info.micro_batch_sizes_list[i].size(); j++) {
            auto recv_buffer = input_batch_buffers[i]->slice(offset, batch_split_info.micro_batch_sizes_list[i][j]);
            recv_params.p2p_params.push_back(P2pSendRecvParams{SendRecvType::kRecv, recv_buffer, j});
            offset += batch_split_info.micro_batch_sizes_list[i][j];
        }
    }
    device_->batchSendRecv(recv_params, ParallelMode::DP_AND_TP);
}

FfnDisaggregateModel::BatchSplitInfo FfnDisaggregateModel::recvBatchSplitInfo() {
    RTP_LLM_LOG_DEBUG("FfnDisaggregateModel recvBatchSplitInfo");
    auto miroBatchBuffer = device_->allocateBuffer(
        {DataType::TYPE_INT32, {(device_props_.dp_size - 1) * micro_batch_size_}, AllocationType::DEVICE});
    BatchSendRecvParams params;
    // current asume ffn service as last rank of dp
    for (int i = 0; i < device_props_.dp_size - 1; i++) {
        auto recv_buffer = miroBatchBuffer->slice(i * micro_batch_size_, micro_batch_size_);
        params.p2p_params.push_back(P2pSendRecvParams{SendRecvType::kRecv, recv_buffer, i});
    }
    device_->batchSendRecv(params, ParallelMode::DP_AND_TP);
    auto           hostMicroBatchBuffer = device_->clone({*miroBatchBuffer, AllocationType::HOST});
    BatchSplitInfo info;
    info.micro_batch_sizes_list.resize(micro_batch_size_, {});
    info.total_micro_batch_sizes.resize(micro_batch_size_, 0);
    for (int dp_idx = 0; dp_idx < device_props_.dp_size - 1; dp_idx++) {
        for (int mirco_batch_idx = 0; mirco_batch_idx < micro_batch_size_; mirco_batch_idx++) {
            info.micro_batch_sizes_list[mirco_batch_idx].push_back(
                hostMicroBatchBuffer->data<int32_t>()[dp_idx * micro_batch_size_ + mirco_batch_idx]);
            info.total_micro_batch_sizes[mirco_batch_idx] += info.micro_batch_sizes_list[mirco_batch_idx][dp_idx];
        }
    }
    return info;
}

void FfnDisaggregateModel::sendBatchSplitInfo(const std::vector<LayerMicroBatchInputs>& batch_infos) {
    RTP_LLM_LOG_DEBUG("FfnDisaggregateModel sendBatchSplitInfo");
    RTP_LLM_CHECK_WITH_INFO(batch_infos.size() == micro_batch_size_, "batch_infos.size() != %d", micro_batch_size_);
    auto miroBatchBuffer = device_->allocateBuffer({DataType::TYPE_INT32, {batch_infos.size()}, AllocationType::HOST});
    for (int i = 0; i < batch_infos.size(); i++) {
        RTP_LLM_CHECK_WITH_INFO(batch_infos[i].hidden != nullptr && batch_infos[i].hidden->shape()[0] > 0,
                                "batch_infos[%d].hidden is nullptr or empty",
                                i);
        miroBatchBuffer->data<int32_t>()[i] = batch_infos[i].hidden->shape()[0];
    }
    auto deviceBatchBuffer = device_->clone({*miroBatchBuffer});
    // step1: send micro batch size to last rank
    device_->batchSendRecv(
        BatchSendRecvParams{.p2p_params =
                                {
                                    {SendRecvType::kSend, deviceBatchBuffer, int(device_props_.dp_size - 1)},
                                }},
        ParallelMode::DP_AND_TP);
}

void FfnDisaggregateModel::sendInputBuffer(const std::vector<LayerMicroBatchInputs>& batch_infos) {
    RTP_LLM_LOG_DEBUG("FfnDisaggregateModel sendInputBuffer");
    BatchSendRecvParams send_params;
    for (int i = 0; i < batch_infos.size(); i++) {
        auto send_buffer = batch_infos[i].hidden;
        send_params.p2p_params.push_back(
            P2pSendRecvParams{SendRecvType::kSend, send_buffer, int(device_props_.dp_size - 1)});
    }
    device_->batchSendRecv(send_params, ParallelMode::DP_AND_TP);
    device_->checkError();
}

}  // namespace rtp_llm
