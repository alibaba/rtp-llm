#include "maga_transformer/cpp/model_rpc/QueryConverter.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace ft = fastertransformer;

namespace rtp_llm {
#define TRANS_OPTIONAL(name)                                                                                           \
    if (config_proto->has_##name()) {                                                                                  \
        generate_config->name = config_proto->name().value();                                                          \
    }

std::shared_ptr<GenerateConfig> QueryConverter::transGenerateConfig(const GenerateConfigPB* config_proto) {
    std::shared_ptr<GenerateConfig> generate_config = std::make_shared<GenerateConfig>();
    generate_config->max_new_tokens                 = config_proto->max_new_tokens();
    generate_config->min_new_tokens                 = config_proto->min_new_tokens();
    generate_config->num_beams                      = config_proto->num_beams();
    generate_config->num_return_sequences           = config_proto->num_return_sequences();
    generate_config->return_logits                  = config_proto->return_logits();
    generate_config->return_incremental             = config_proto->return_incremental();
    generate_config->return_hidden_states           = config_proto->return_hidden_states();
    generate_config->calculate_loss                 = config_proto->calculate_loss();
    generate_config->is_streaming                   = config_proto->is_streaming();
    generate_config->timeout_ms                     = config_proto->timeout_ms();
    generate_config->select_tokens_id.resize(config_proto->select_tokens_id_size());
    memcpy(generate_config->select_tokens_id.data(), config_proto->select_tokens_id().data(), config_proto->select_tokens_id_size() * sizeof(int));
    for (const auto& stop_words_proto : config_proto->stop_words_list().rows()) {
        std::vector<int> stop_words;
        for (const int value : stop_words_proto.values()) {
            stop_words.push_back(value);
        }
        generate_config->stop_words_list.push_back(stop_words);
    }

    generate_config->top_k = config_proto->top_k();
    generate_config->top_p = config_proto->top_p();
    generate_config->temperature = config_proto->temperature();
    generate_config->repetition_penalty = config_proto->repetition_penalty();
    TRANS_OPTIONAL(random_seed);
    TRANS_OPTIONAL(top_p_decay);
    TRANS_OPTIONAL(top_p_min);
    TRANS_OPTIONAL(top_p_reset_ids);
    TRANS_OPTIONAL(task_id);
    return generate_config;
}

std::shared_ptr<GenerateInput> QueryConverter::transQuery(const GenerateInputPB* input) {
    std::shared_ptr<GenerateInput> generate_input = std::make_shared<GenerateInput>();
    generate_input->request_id = input->request_id();
    if (input->has_generate_config()) {
        generate_input->generate_config = transGenerateConfig(&(input->generate_config()));
    }
    generate_input->lora_id       = input->lora_id();
    auto device                   = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    generate_input->input_ids     = device->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)input->token_ids_size()}, ft::AllocationType::HOST}, {});
    memcpy(generate_input->input_ids->data(), input->token_ids().data(), generate_input->input_ids->sizeBytes());
    return generate_input;
}

void QueryConverter::transTensor(TensorPB* t, const ft::Buffer* buffer) {
    assert(t);
    auto shape       = t->mutable_shape();
    auto shape_array = buffer->shape();
    shape->Resize(shape_array.size(), 0);
    memcpy(shape->mutable_data(), shape_array.data(), shape_array.size() * sizeof(int64_t));

    TensorPB_DataType data_type;
    switch(buffer->type()) {
        case ft::DataType::TYPE_FP32: 
            data_type = TensorPB_DataType::TensorPB_DataType_FP32;
            t->set_fp32_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case ft::DataType::TYPE_INT32:
            data_type = TensorPB_DataType::TensorPB_DataType_INT32;
            t->set_int32_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case ft::DataType::TYPE_FP16:
            data_type = TensorPB_DataType::TensorPB_DataType_FP16;
            t->set_fp16_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case ft::DataType::TYPE_BF16:
            data_type = TensorPB_DataType::TensorPB_DataType_BF16;
            t->set_bf16_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        default:
            throw std::invalid_argument("unsupport buffer data type: " + std::to_string(buffer->type()));
            break; 
    }
    t->set_data_type(data_type);
}

void QueryConverter::transResponse(GenerateOutputsPB* outputs, const GenerateOutputs* responses) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    outputs->set_request_id(responses->request_id);
    for (size_t i = 0; i < responses->generate_outputs.size(); i++) {
        const auto& response = responses->generate_outputs[i];
        GenerateOutputPB* output = outputs->add_generate_outputs();
        output->set_finished(response.finished);
        auto aux_info = output->mutable_aux_info();
        aux_info->set_cost_time_us(response.aux_info.cost_time_us);
        aux_info->set_iter_count(response.aux_info.iter_count);
        aux_info->set_input_len(response.aux_info.input_len);
        aux_info->set_reuse_len(response.aux_info.reuse_len);
        aux_info->set_prefix_len(response.aux_info.prefix_len);
        aux_info->set_output_len(response.aux_info.output_len);
        if (response.aux_info.cum_log_probs.has_value()) {
            transTensor(aux_info->mutable_cum_log_probs(), response.aux_info.cum_log_probs.value().get());
        }
        transTensor(output->mutable_output_ids(), response.output_ids.get());
        if (response.hidden_states.has_value()) {
            transTensor(output->mutable_hidden_states(), response.hidden_states.value().get());
        }
        if (response.loss.has_value()) {
            transTensor(output->mutable_loss(), response.loss.value().get());
        }
        if (response.logits.has_value()) {
            transTensor(output->mutable_logits(), response.logits.value().get());
        }
    }
    FT_LOG_DEBUG("transResponse done");
}

}  // namespace rtp_llm
