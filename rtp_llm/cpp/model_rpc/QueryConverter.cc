#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

#include "RPCPool.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
#define TRANS_OPTIONAL(name)                                                                                           \
    if (config_proto->has_##name()) {                                                                                  \
        generate_config->name = config_proto->name().value();                                                          \
    }

std::shared_ptr<GenerateConfig> QueryConverter::transGenerateConfig(const GenerateConfigPB* config_proto) {
    std::shared_ptr<GenerateConfig> generate_config = std::make_shared<GenerateConfig>();
    generate_config->global_request_id              = config_proto->global_request_id();
    generate_config->max_new_tokens                 = config_proto->max_new_tokens();
    generate_config->min_new_tokens                 = config_proto->min_new_tokens();
    generate_config->num_beams                      = config_proto->num_beams();
    generate_config->variable_num_beams.resize(config_proto->variable_num_beams_size());
    memcpy(generate_config->variable_num_beams.data(),
           config_proto->variable_num_beams().data(),
           config_proto->variable_num_beams_size() * sizeof(int));
    generate_config->num_return_sequences     = config_proto->num_return_sequences();
    generate_config->return_logits            = config_proto->return_logits();
    generate_config->return_incremental       = config_proto->return_incremental();
    generate_config->return_hidden_states     = config_proto->return_hidden_states();
    generate_config->return_all_hidden_states = config_proto->return_all_hidden_states();
    generate_config->hidden_states_cut_dim    = config_proto->hidden_states_cut_dim();
    generate_config->normalized_hidden_states = config_proto->normalized_hidden_states();
    generate_config->calculate_loss           = config_proto->calculate_loss();
    generate_config->is_streaming             = config_proto->is_streaming();
    generate_config->timeout_ms               = config_proto->timeout_ms();
    generate_config->sp_edit                  = config_proto->sp_edit();
    generate_config->force_disable_sp_run     = config_proto->force_disable_sp_run();
    generate_config->force_sp_accept          = config_proto->force_sp_accept();
    generate_config->return_cum_log_probs     = config_proto->return_cum_log_probs();
    generate_config->return_all_probs         = ReturnAllProbsMode(config_proto->return_all_probs());
    generate_config->return_softmax_probs     = config_proto->return_softmax_probs();
    generate_config->can_use_pd_separation    = config_proto->can_use_pd_separation();
    generate_config->gen_timeline             = config_proto->gen_timeline();
    generate_config->profile_step             = config_proto->profile_step();
    generate_config->ignore_eos               = config_proto->ignore_eos();
    generate_config->select_tokens_id.resize(config_proto->select_tokens_id_size());
    memcpy(generate_config->select_tokens_id.data(),
           config_proto->select_tokens_id().data(),
           config_proto->select_tokens_id_size() * sizeof(int));
    for (const auto& stop_words_proto : config_proto->stop_words_list().rows()) {
        std::vector<int> stop_words;
        for (const int value : stop_words_proto.values()) {
            stop_words.push_back(value);
        }
        generate_config->stop_words_list.push_back(stop_words);
    }

    for (const auto& token_id : config_proto->sp_advice_prompt_token_ids()) {
        generate_config->sp_advice_prompt_token_ids.push_back(token_id);
    }

    generate_config->top_k              = config_proto->top_k();
    generate_config->top_p              = config_proto->top_p();
    generate_config->temperature        = config_proto->temperature();
    generate_config->repetition_penalty = config_proto->repetition_penalty();
    generate_config->presence_penalty   = config_proto->presence_penalty();
    generate_config->frequency_penalty  = config_proto->frequency_penalty();
    generate_config->do_sample          = config_proto->do_sample();
    TRANS_OPTIONAL(no_repeat_ngram_size);
    TRANS_OPTIONAL(random_seed);
    TRANS_OPTIONAL(top_p_decay);
    TRANS_OPTIONAL(top_p_min);
    TRANS_OPTIONAL(top_p_reset_ids);
    TRANS_OPTIONAL(task_id);
    TRANS_OPTIONAL(adapter_name);
    generate_config->in_think_mode       = config_proto->in_think_mode();
    generate_config->max_thinking_tokens = config_proto->max_thinking_tokens();
    for (const auto& token_id : config_proto->end_think_token_ids()) {
        generate_config->end_think_token_ids.push_back(token_id);
    }

    for (const auto& role_addr : config_proto->role_addrs()) {
        generate_config->role_addrs.emplace_back(
            RoleType(role_addr.role()), role_addr.ip(), role_addr.http_port(), role_addr.grpc_port());
    }

    generate_config->inter_request_id    = config_proto->inter_request_id();
    generate_config->reuse_cache         = config_proto->reuse_cache();
    generate_config->enable_3fs          = config_proto->enable_3fs();
    generate_config->enable_device_cache = config_proto->enable_device_cache();
    generate_config->enable_memory_cache = config_proto->enable_memory_cache();
    TRANS_OPTIONAL(trace_id);

    return generate_config;
}

std::shared_ptr<GenerateInput> QueryConverter::transQuery(const GenerateInputPB* input) {
    std::shared_ptr<GenerateInput> generate_input = std::make_shared<GenerateInput>();
    generate_input->request_id                    = input->request_id();
    generate_input->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
    if (input->has_generate_config()) {
        generate_input->generate_config = transGenerateConfig(&(input->generate_config()));
    }
    auto device               = rtp_llm::DeviceFactory::getDefaultDevice();
    generate_input->input_ids = device->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)input->token_ids_size()}, rtp_llm::AllocationType::HOST}, {});
    memcpy(generate_input->input_ids->data(), input->token_ids().data(), generate_input->input_ids->sizeBytes());
    if (input->multimodal_inputs_size() > 0) {
        std::vector<MultimodalInput> mm_inputs;
        for (int i = 0; i < input->multimodal_inputs_size(); i++) {
            auto               mm_input             = &input->multimodal_inputs(i);
            auto               mm_preprocess_config = &mm_input->mm_preprocess_config();
            std::vector<float> crop_positions;
            for (const auto& crop_position : mm_preprocess_config->crop_positions()) {
                crop_positions.push_back(crop_position);
            }
            mm_inputs.emplace_back(mm_input->multimodal_url(),
                                   torch::empty(1),
                                   mm_input->multimodal_type(),
                                   mm_preprocess_config->width(),
                                   mm_preprocess_config->height(),
                                   mm_preprocess_config->min_pixels(),
                                   mm_preprocess_config->max_pixels(),
                                   mm_preprocess_config->fps(),
                                   mm_preprocess_config->min_frames(),
                                   mm_preprocess_config->max_frames(),
                                   crop_positions,
                                   mm_preprocess_config->mm_timeout_ms());
        }
        generate_input->multimodal_inputs = std::move(mm_inputs);
    }

    return generate_input;
}

std::vector<RoleAddr> QueryConverter::getRoleAddrs(const GenerateConfigPB* config_proto) {
    std::vector<RoleAddr> role_addrs;
    for (const auto& role_addr : config_proto->role_addrs()) {
        role_addrs.emplace_back(
            RoleType(role_addr.role()), role_addr.ip(), role_addr.http_port(), role_addr.grpc_port());
    }
    return role_addrs;
}

std::vector<MultimodalInput> QueryConverter::transMMInput(const MultimodalInputsPB* mm_inputs) {
    std::vector<MultimodalInput> inputs_vec;
    for (int i = 0; i < mm_inputs->multimodal_inputs_size(); i++) {
        auto mm_input             = &mm_inputs->multimodal_inputs(i);
        auto mm_preprocess_config = &mm_input->mm_preprocess_config();

        std::vector<float> crop_positions;
        for (const auto& crop_position : mm_preprocess_config->crop_positions()) {
            crop_positions.push_back(crop_position);
        }

        // tensor should also converted from input pb, however it is only used in some embedding model, so just empty
        // for now
        inputs_vec.emplace_back(mm_input->multimodal_url(),
                                torch::empty(1),
                                mm_input->multimodal_type(),
                                mm_preprocess_config->width(),
                                mm_preprocess_config->height(),
                                mm_preprocess_config->min_pixels(),
                                mm_preprocess_config->max_pixels(),
                                mm_preprocess_config->fps(),
                                mm_preprocess_config->min_frames(),
                                mm_preprocess_config->max_frames(),
                                crop_positions,
                                mm_preprocess_config->mm_timeout_ms());
    }
    return inputs_vec;
}

MultimodalInputsPB QueryConverter::transMMInputsPB(const std::vector<MultimodalInput> mm_inputs) {
    MultimodalInputsPB mm_inputs_pb;
    for (auto& mm_input : mm_inputs) {
        auto now_input = mm_inputs_pb.add_multimodal_inputs();
        now_input->set_multimodal_url(mm_input.url);
        now_input->set_multimodal_type(mm_input.mm_type);
        transTensorPB(now_input->mutable_multimodal_tensor(), rtp_llm::torchTensor2Buffer(mm_input.tensor).get());
        transMMPreprocessConfig(now_input->mutable_mm_preprocess_config(), mm_input.mm_preprocess_config);
    }
    return mm_inputs_pb;
}

void QueryConverter::transMMPreprocessConfig(MMPreprocessConfigPB* config_pb, const MMPreprocessConfig config) {
    config_pb->set_width(config.width);
    config_pb->set_height(config.height);
    config_pb->set_min_pixels(config.min_pixels);
    config_pb->set_max_pixels(config.max_pixels);
    config_pb->set_fps(config.fps);
    config_pb->set_min_frames(config.min_frames);
    config_pb->set_max_frames(config.max_frames);
    config_pb->set_mm_timeout_ms(config.mm_timeout_ms);
    for (const float& crop_position : config.crop_positions) {
        config_pb->add_crop_positions(crop_position);
    }
}

MultimodalOutput QueryConverter::transMMOutput(const MultimodalOutputPB* output_pb) {
    torch::Tensor mm_embedding = transTensor(output_pb->multimodal_embedding()), mm_position_id, mm_deepstack_embeds;
    bool          contain_pos  = output_pb->has_multimodal_pos_id();
    bool          contain_deepstack = output_pb->has_multimodal_deepstack_embeds();
    if (contain_pos) {
        mm_position_id = transTensor(output_pb->multimodal_pos_id());
    }
    if (contain_deepstack) {
        mm_deepstack_embeds = transTensor(output_pb->multimodal_deepstack_embeds());
    }
    MultimodalOutput     mm_output;
    std::vector<int64_t> split_sizes;
    for (auto split_size : output_pb->split_size()) {
        split_sizes.push_back(split_size);
    }
    mm_output.mm_features = mm_embedding.split(split_sizes, 0);
    if (contain_pos) {
        mm_output.mm_position_ids = mm_position_id.split(split_sizes, 0);
    }

    if (contain_deepstack) {
        mm_output.mm_deepstack_embeds = mm_deepstack_embeds.split(split_sizes, 1);
    }
    return mm_output;
}

torch::Tensor QueryConverter::transTensor(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    void*                data_ptr = nullptr;
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            data_ptr     = const_cast<char*>(tensor_pb.fp32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::INT32: {
            data_ptr     = const_cast<char*>(tensor_pb.int32_data().data());
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::FP16: {
            data_ptr     = const_cast<char*>(tensor_pb.fp16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        case TensorPB::BF16: {
            data_ptr     = const_cast<char*>(tensor_pb.bf16_data().data());
            auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            return torch::from_blob(data_ptr, shape, options).clone();
        }
        default:
            throw std::runtime_error("Unsupported data type.");
    }
}

void QueryConverter::transTensorPB(TensorPB* tensor_pb, const torch::Tensor& tensor) {

    // 设置数据类型
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32:
            tensor_pb->set_data_type(TensorPB::FP32);
            break;
        case torch::kInt32:
            tensor_pb->set_data_type(TensorPB::INT32);
            break;
        case torch::kFloat16:
            tensor_pb->set_data_type(TensorPB::FP16);
            break;
        case torch::kBFloat16:
            tensor_pb->set_data_type(TensorPB::BF16);
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    auto shape = tensor.sizes();
    for (auto dim : shape) {
        tensor_pb->add_shape(dim);
    }
    torch::Tensor contiguous_tensor = tensor.contiguous();
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(float);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_fp32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kInt32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(int32_t);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_int32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::Half);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_fp16_data(data_ptr, num_bytes);
            break;
        }
        case torch::kBFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::BFloat16);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            tensor_pb->set_bf16_data(data_ptr, num_bytes);
            break;
        }
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
}

void QueryConverter::transTensorPB(TensorPB* t, const rtp_llm::Buffer* buffer) {
    RTP_LLM_CHECK(t != nullptr);
    RTP_LLM_CHECK_WITH_INFO(buffer->where() != rtp_llm::MemoryType::MEMORY_GPU,
                            "buffer is on gpu, not supported transfer to tensorpb");
    auto shape       = t->mutable_shape();
    auto shape_array = buffer->shape();
    shape->Resize(shape_array.size(), 0);
    memcpy(shape->mutable_data(), shape_array.data(), shape_array.size() * sizeof(int64_t));

    TensorPB_DataType data_type;
    switch (buffer->type()) {
        case rtp_llm::DataType::TYPE_FP32:
            data_type = TensorPB_DataType::TensorPB_DataType_FP32;
            t->set_fp32_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case rtp_llm::DataType::TYPE_INT32:
            data_type = TensorPB_DataType::TensorPB_DataType_INT32;
            t->set_int32_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case rtp_llm::DataType::TYPE_FP16:
            data_type = TensorPB_DataType::TensorPB_DataType_FP16;
            t->set_fp16_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        case rtp_llm::DataType::TYPE_BF16:
            data_type = TensorPB_DataType::TensorPB_DataType_BF16;
            t->set_bf16_data(reinterpret_cast<const char*>(buffer->data()), buffer->sizeBytes());
            break;
        default:
            throw std::invalid_argument("unsupport buffer data type: " + std::to_string(buffer->type()));
            break;
    }
    t->set_data_type(data_type);
}

template<typename T>
void QueryConverter::mergeAndPadBuffersToTensorPB(TensorPB*                                   target_pb,
                                                  const std::vector<rtp_llm::ConstBufferPtr>& buffers,
                                                  T                                           pad_value) {
    if (buffers.empty()) {
        return;
    }

    size_t max_len = 0;
    for (const auto& buffer : buffers) {
        RTP_LLM_CHECK(buffer->dim() == 2 && buffer->shape()[0] == 1);
        if (buffer->shape()[1] > max_len) {
            max_len = buffer->shape()[1];
        }
    }

    const size_t        batch_size  = buffers.size();
    std::vector<size_t> final_shape = {batch_size, 1, max_len};

    const auto   mem_type       = buffers[0]->where();
    const auto   data_type      = buffers[0]->type();
    const size_t total_elements = batch_size * max_len;
    T*           new_data       = new T[total_elements];
    std::fill(new_data, new_data + total_elements, pad_value);

    for (size_t i = 0; i < batch_size; ++i) {
        const auto&  src_buffer = buffers[i];
        T*           dst_ptr    = new_data + i * max_len;
        const T*     src_ptr    = src_buffer->data<T>();
        const size_t src_len    = src_buffer->shape()[1];
        memcpy(dst_ptr, src_ptr, src_len * sizeof(T));
    }

    auto deleter       = [=](rtp_llm::Buffer* b) { delete[] b->data<T>(); };
    auto merged_buffer = std::make_shared<rtp_llm::Buffer>(mem_type, data_type, final_shape, new_data, deleter);
    transTensorPB(target_pb, merged_buffer.get());
}

template<typename Container, typename Accessor>
void QueryConverter::stackBuffersToTensorPB(TensorPB*        target_pb,
                                            const Container& source_container,
                                            Accessor         tensor_accessor) {
    rtp_llm::ConstBufferPtr ref_buffer = nullptr;
    for (const auto& item : source_container) {
        auto buffer_opt = std::invoke(tensor_accessor, item);
        if (buffer_opt.has_value()) {
            ref_buffer = *buffer_opt;
            break;
        }
    }

    if (!ref_buffer) {
        return;
    }

    const auto&  ref_shape                = ref_buffer->shape();
    const size_t single_buffer_size_bytes = ref_buffer->sizeBytes();
    const size_t batch_size               = source_container.size();

    std::vector<size_t> final_shape = {batch_size};
    final_shape.insert(final_shape.end(), ref_shape.begin(), ref_shape.end());

    const auto   mem_type    = ref_buffer->where();
    const auto   data_type   = ref_buffer->type();
    const size_t total_bytes = batch_size * single_buffer_size_bytes;
    char*        new_data    = new char[total_bytes];

    char* current_dst_ptr = new_data;
    for (const auto& item : source_container) {
        auto buffer_opt = std::invoke(tensor_accessor, item);
        RTP_LLM_CHECK_WITH_INFO(buffer_opt.has_value(), "Inconsistent tensor presence in a batch for stacking.");
        auto current_buffer = *buffer_opt;

        RTP_LLM_CHECK_WITH_INFO(current_buffer->shape() == ref_shape,
                                "All buffers must have the same shape for stacking.");

        memcpy(current_dst_ptr, current_buffer->data(), single_buffer_size_bytes);
        current_dst_ptr += single_buffer_size_bytes;
    }

    auto deleter        = [=](rtp_llm::Buffer* b) { delete[] static_cast<char*>(b->data()); };
    auto stacked_buffer = std::make_shared<rtp_llm::Buffer>(mem_type, data_type, final_shape, new_data, deleter);
    QueryConverter::transTensorPB(target_pb, stacked_buffer.get());
}

void QueryConverter::transResponse(GenerateOutputsPB*     outputs,
                                   const GenerateOutputs* responses,
                                   bool                   dump_aux_info,
                                   const std::string&     aux_string,
                                   const int32_t          eos_token_id) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    outputs->set_request_id(responses->request_id);
    const auto& source_outputs = responses->generate_outputs;
    if (source_outputs.empty()) {
        return;
    }
    FlattenOutputPB* flatten_output = outputs->mutable_flatten_output();
    for (const auto& response : source_outputs) {
        flatten_output->add_finished(response.finished);
        if (dump_aux_info) {
            auto* aux_info = flatten_output->add_aux_info();
            aux_info->set_cost_time_us(response.aux_info.cost_time_us);
            aux_info->set_first_token_cost_time_us(response.aux_info.first_token_cost_time_us);
            aux_info->set_wait_time_us(response.aux_info.wait_time_us);
            aux_info->set_iter_count(response.aux_info.iter_count);
            aux_info->set_input_len(response.aux_info.input_len);
            aux_info->set_prefix_len(response.aux_info.prefix_len);
            aux_info->set_output_len(response.aux_info.output_len);
            aux_info->set_step_output_len(response.aux_info.step_output_len);
            aux_info->set_pd_sep(response.aux_info.pd_sep);
            aux_info->set_total_reuse_len(response.aux_info.reuse_len);
            aux_info->set_local_reuse_len(response.aux_info.local_reuse_len);
            aux_info->set_remote_reuse_len(response.aux_info.remote_reuse_len);
            aux_info->set_aux_string(aux_string);
            if (response.aux_info.cum_log_probs.has_value()) {
                transTensorPB(aux_info->mutable_cum_log_probs(), response.aux_info.cum_log_probs.value().get());
            }
            if (response.aux_info.softmax_probs.has_value()) {
                transTensorPB(aux_info->mutable_softmax_probs(), response.aux_info.softmax_probs.value().get());
            }
        }
    }

    std::vector<rtp_llm::ConstBufferPtr> output_id_buffers;
    output_id_buffers.reserve(source_outputs.size());
    for (const auto& resp : source_outputs) {
        if (resp.output_ids) {
            output_id_buffers.push_back(resp.output_ids);
        }
    }
    if (!output_id_buffers.empty()) {
        mergeAndPadBuffersToTensorPB<int32_t>(flatten_output->mutable_output_ids(), output_id_buffers, eos_token_id);
    }

    stackBuffersToTensorPB(
        flatten_output->mutable_all_probs(), source_outputs, [](const auto& r) { return r.aux_info.all_probs; });
    stackBuffersToTensorPB(
        flatten_output->mutable_hidden_states(), source_outputs, [](const auto& r) { return r.hidden_states; });

    stackBuffersToTensorPB(flatten_output->mutable_loss(), source_outputs, [](const auto& r) { return r.loss; });

    stackBuffersToTensorPB(flatten_output->mutable_logits(), source_outputs, [](const auto& r) { return r.logits; });

    stackBuffersToTensorPB(
        flatten_output->mutable_all_hidden_states(), source_outputs, [](const auto& r) { return r.all_hidden_states; });

    RTP_LLM_LOG_DEBUG("transResponse done");
}

}  // namespace rtp_llm
