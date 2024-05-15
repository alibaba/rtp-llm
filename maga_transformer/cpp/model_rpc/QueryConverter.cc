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

    TRANS_OPTIONAL(top_k);
    TRANS_OPTIONAL(top_p);
    TRANS_OPTIONAL(temperature);
    TRANS_OPTIONAL(repetition_penalty);
    TRANS_OPTIONAL(random_seed);
    TRANS_OPTIONAL(top_p_decay);
    TRANS_OPTIONAL(top_p_min);
    TRANS_OPTIONAL(top_p_reset_ids);
    TRANS_OPTIONAL(task_id);
    TRANS_OPTIONAL(adapter_name);
    return generate_config;
}

std::shared_ptr<GenerateStream> QueryConverter::transQuery(const ResourceContext& resource_context, const GenerateInputPB* input) {
    std::shared_ptr<GenerateInput> generate_input = std::make_shared<GenerateInput>();

    if (input->has_generate_config()) {
        generate_input->generate_config = transGenerateConfig(&(input->generate_config()));
    }
    generate_input->prefix_length = input->prefix_length();
    auto device                   = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    generate_input->input_ids     = device->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)input->token_ids_size()}, ft::AllocationType::HOST}, {});
    memcpy(generate_input->input_ids->data(), input->token_ids().data(), generate_input->input_ids->sizeBytes());
    // TODO(xinfei.sxf) set max seq len
    std::shared_ptr<GenerateStream> stream = std::make_shared<GenerateStream>(generate_input, resource_context);

    return stream;
}

void QueryConverter::transTensor(TensorPB* t, const ft::Buffer* buffer) {
    assert(t);
    t->set_data_type(buffer->type() == ft::DataType::TYPE_FP32 ? TensorPB_DataType::TensorPB_DataType_FLOAT32 :
                                                                 TensorPB_DataType::TensorPB_DataType_INT32);
    auto shape       = t->mutable_shape();
    auto shape_array = buffer->shape();
    shape->Resize(shape_array.size(), 0);
    memcpy(shape->mutable_data(), shape_array.data(), shape_array.size() * sizeof(int64_t));
    if (buffer->type() == ft::DataType::TYPE_FP32) {
        auto tensor = t->mutable_data_float32();
        tensor->Resize(buffer->size(), 0);
        memcpy(tensor->mutable_data(), buffer->data(), buffer->sizeBytes());
    } else if (buffer->type() == ft::DataType::TYPE_INT32) {
        auto tensor = t->mutable_data_int32();
        tensor->Resize(buffer->size(), 0);
        memcpy(tensor->mutable_data(), buffer->data(), buffer->sizeBytes());
    } else {
        throw std::invalid_argument("unsupport tensor type");
    }
}

void QueryConverter::transResponse(GenerateOutputPB* output, const GenerateOutput* response) {
    output->set_finished(response->finished);
    auto aux_info = output->mutable_aux_info();
    aux_info->set_cost_time_ms(response->aux_info.cost_time_ms);
    aux_info->set_iter_count(response->aux_info.iter_count);
    aux_info->set_input_len(response->aux_info.input_len);
    aux_info->set_output_len(response->aux_info.output_len);
    // aux_info->mutable_cum_log_probs()

    transTensor(output->mutable_output_ids(), response->output_ids.get());
    if (response->hidden_states.has_value()) {
        transTensor(output->mutable_hidden_states(), response->hidden_states.value().get());
    }
    if (response->loss.has_value()) {
        transTensor(output->mutable_loss(), response->loss.value().get());
    }
    if (response->logits.has_value()) {
        transTensor(output->mutable_logits(), response->logits.value().get());
    }
}

}  // namespace rtp_llm
