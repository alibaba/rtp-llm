#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

#include "RPCPool.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

#include <algorithm>

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
    generate_config->return_all_probs         = config_proto->return_all_probs();
    generate_config->return_logprobs          = config_proto->return_logprobs();
    generate_config->top_logprobs             = static_cast<int>(config_proto->top_logprobs());
    generate_config->return_softmax_probs     = config_proto->return_softmax_probs();
    generate_config->can_use_pd_separation    = config_proto->can_use_pd_separation();
    generate_config->gen_timeline             = config_proto->gen_timeline();
    generate_config->profile_step             = config_proto->profile_step();
    generate_config->profile_trace_name       = config_proto->profile_trace_name();
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
    TRANS_OPTIONAL(json_schema);
    TRANS_OPTIONAL(regex);
    TRANS_OPTIONAL(ebnf);
    TRANS_OPTIONAL(structural_tag);
    TRANS_OPTIONAL(response_format);
    TRANS_OPTIONAL(adapter_name);
    generate_config->in_think_mode       = config_proto->in_think_mode();
    generate_config->max_thinking_tokens = config_proto->max_thinking_tokens();
    for (const auto& token_id : config_proto->begin_think_token_ids()) {
        generate_config->begin_think_token_ids.push_back(token_id);
    }
    for (const auto& token_id : config_proto->end_think_token_ids()) {
        generate_config->end_think_token_ids.push_back(token_id);
    }

    for (const auto& role_addr : config_proto->role_addrs()) {
        generate_config->role_addrs.emplace_back(
            RoleType(role_addr.role()), role_addr.ip(), role_addr.http_port(), role_addr.grpc_port());
    }

    generate_config->reuse_cache         = config_proto->reuse_cache();
    generate_config->enable_device_cache = config_proto->enable_device_cache();
    generate_config->enable_memory_cache = config_proto->enable_memory_cache();
    generate_config->enable_remote_cache = config_proto->enable_remote_cache();
    TRANS_OPTIONAL(trace_id);
    TRANS_OPTIONAL(batch_group_timeout);
    TRANS_OPTIONAL(force_batch);

    return generate_config;
}

RequestInfo QueryConverter::transRequestInfo(const RequestInfoPB& request_info_pb) {
    RequestInfo request_info;
    request_info.frontend_ip = request_info_pb.frontend_ip();
    request_info.dash_ip     = request_info_pb.dash_ip();
    request_info.trace_id    = request_info_pb.trace_id();
    request_info.request_id  = request_info_pb.request_id();
    request_info.source_role = request_info_pb.source_role();
    return request_info;
}

std::shared_ptr<GenerateInput> QueryConverter::transQuery(const GenerateInputPB* input) {
    std::shared_ptr<GenerateInput> generate_input = std::make_shared<GenerateInput>();
    generate_input->request_id                    = input->request_id();
    generate_input->request_info                  = transRequestInfo(input->request_info());
    generate_input->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
    if (input->has_generate_config()) {
        generate_input->generate_config = transGenerateConfig(&(input->generate_config()));
    }
    if (generate_input->request_info.trace_id.empty() && generate_input->generate_config) {
        generate_input->request_info.trace_id = generate_input->generate_config->trace_id;
    }
    if (generate_input->request_info.request_id.empty()) {
        generate_input->request_info.request_id = std::to_string(input->request_id());
    }
    if (generate_input->generate_config && generate_input->generate_config->trace_id.empty()) {
        generate_input->generate_config->trace_id = generate_input->request_info.trace_id;
    }
    generate_input->input_ids =
        torch::from_blob(const_cast<int*>(input->token_ids().data()), {(int64_t)input->token_ids_size()}, torch::kInt32)
            .clone();
    if (input->multimodal_inputs_size() > 0) {
        std::vector<MultimodalInput> mm_inputs;
        for (int i = 0; i < input->multimodal_inputs_size(); i++) {
            auto mm_input             = &input->multimodal_inputs(i);
            auto mm_preprocess_config = &mm_input->mm_preprocess_config();
            mm_inputs.emplace_back(mm_input->multimodal_url(),
                                   torch::empty(1),
                                   mm_input->multimodal_type(),
                                   mm_preprocess_config->width(),
                                   mm_preprocess_config->height(),
                                   mm_preprocess_config->min_pixels(),
                                   mm_preprocess_config->max_pixels(),
                                   mm_preprocess_config->fps(),
                                   mm_preprocess_config->min_frames(),
                                   mm_preprocess_config->max_frames());
        }
        generate_input->multimodal_inputs = std::move(mm_inputs);
    }
    generate_input->batch_group_size = input->batch_group_size() > 0 ? input->batch_group_size() : 1;
    if (input->has_batch_group_id()) {
        generate_input->batch_group_id = input->batch_group_id().value();
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
        // tensor should also converted from input pb, however it is only used in some embedding model, so just empty
        // for now
        inputs_vec.emplace_back(mm_input->multimodal_url(),
                                torch::empty(1),
                                mm_input->multimodal_type(),
                                mm_preprocess_config->width(),
                                mm_preprocess_config->height(),
                                mm_preprocess_config->min_pixels(),
                                mm_preprocess_config->max_pixels(),
                                mm_preprocess_config->fps());
    }
    return inputs_vec;
}

MultimodalInputsPB QueryConverter::transMMInputsPB(const std::vector<MultimodalInput> mm_inputs) {
    MultimodalInputsPB mm_inputs_pb;
    for (auto& mm_input : mm_inputs) {
        auto now_input = mm_inputs_pb.add_multimodal_inputs();
        now_input->set_multimodal_url(mm_input.url);
        now_input->set_multimodal_type(mm_input.mm_type);
        transTensorPB(now_input->mutable_multimodal_tensor(), mm_input.tensor);
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
}

MultimodalOutput QueryConverter::transMMOutput(const MultimodalOutputsPB* outputs_pb) {
    MultimodalOutput mm_output;
    for (int i = 0; i < outputs_pb->multimodal_outputs_size(); i++) {
        auto output_pb = outputs_pb->multimodal_outputs(i);
        mm_output.mm_features.emplace_back(transTensor(output_pb.multimodal_embedding()));
        if (output_pb.has_multimodal_pos_id()) {
            if (mm_output.mm_position_ids == std::nullopt) {
                mm_output.mm_position_ids = std::vector<torch::Tensor>();
            }
            mm_output.mm_position_ids.value().emplace_back(transTensor(output_pb.multimodal_pos_id()));
        }
    }
    return mm_output;
}

torch::Tensor QueryConverter::transTensor(const TensorPB& tensor_pb) {
    return TensorPbConvert::pbToTorch(tensor_pb);
}

void QueryConverter::transTensorPB(TensorPB* tensor_pb, const torch::Tensor& tensor) {
    TensorPbConvert::torchToPb(tensor_pb, tensor);
}

template<typename T>
void QueryConverter::mergeAndPadTensorsToTensorPB(TensorPB*                         target_pb,
                                                  const std::vector<torch::Tensor>& tensors,
                                                  T                                 pad_value) {
    if (tensors.empty()) {
        return;
    }

    int64_t max_len = 0;
    for (const auto& t : tensors) {
        RTP_LLM_CHECK(t.dim() == 2 && t.size(0) == 1);
        if (t.size(1) > max_len) {
            max_len = t.size(1);
        }
    }

    const int64_t batch_size = tensors.size();
    // Create padded tensor [batch_size, 1, max_len]
    auto merged = torch::full({batch_size, 1, max_len}, pad_value, tensors[0].options());
    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t src_len = tensors[i].size(1);
        merged[i][0].slice(0, 0, src_len).copy_(tensors[i][0].slice(0, 0, src_len));
    }
    transTensorPB(target_pb, merged.contiguous());
}

template<typename Container, typename Accessor>
void QueryConverter::stackBuffersToTensorPB(TensorPB*        target_pb,
                                            const Container& source_container,
                                            Accessor         tensor_accessor) {
    torch::Tensor ref_tensor;
    for (const auto& item : source_container) {
        auto tensor_opt = std::invoke(tensor_accessor, item);
        if (tensor_opt.has_value()) {
            ref_tensor = *tensor_opt;
            break;
        }
    }

    if (!ref_tensor.defined()) {
        return;
    }

    std::vector<torch::Tensor> tensors;
    tensors.reserve(source_container.size());
    for (const auto& item : source_container) {
        auto tensor_opt = std::invoke(tensor_accessor, item);
        RTP_LLM_CHECK_WITH_INFO(tensor_opt.has_value(), "Inconsistent tensor presence in a batch for stacking.");
        tensors.push_back(tensor_opt->contiguous());
    }

    auto stacked = torch::stack(tensors, 0);
    QueryConverter::transTensorPB(target_pb, stacked);
}

void QueryConverter::padAndStackLogprobTensorsToTensorPB(TensorPB*                         target_pb,
                                                         const std::vector<torch::Tensor>& tensors) {
    if (tensors.empty()) {
        return;
    }

    const auto& reference = tensors.front();
    RTP_LLM_CHECK_WITH_INFO(reference.defined() && reference.dim() >= 1,
                            "logprob tensor must be defined and have a token-row dimension");

    int64_t max_rows = 0;
    for (const auto& tensor : tensors) {
        RTP_LLM_CHECK_WITH_INFO(tensor.defined() && tensor.dim() == reference.dim(),
                                "logprob tensors must have a consistent rank for padding");
        RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == reference.scalar_type()
                                    && tensor.device() == reference.device(),
                                "logprob tensors must have a consistent dtype and device for padding");
        for (int64_t dim = 1; dim < reference.dim(); ++dim) {
            RTP_LLM_CHECK_WITH_INFO(tensor.size(dim) == reference.size(dim),
                                    "logprob tensor trailing dimensions must match for padding");
        }
        max_rows = std::max(max_rows, tensor.size(0));
    }

    std::vector<int64_t> merged_shape;
    merged_shape.reserve(static_cast<size_t>(reference.dim()) + 1);
    merged_shape.push_back(static_cast<int64_t>(tensors.size()));
    merged_shape.push_back(max_rows);
    for (int64_t dim = 1; dim < reference.dim(); ++dim) {
        merged_shape.push_back(reference.size(dim));
    }

    auto merged = torch::zeros(merged_shape, reference.options());
    for (size_t index = 0; index < tensors.size(); ++index) {
        const int64_t rows = tensors[index].size(0);
        if (rows > 0) {
            merged[static_cast<int64_t>(index)].narrow(0, 0, rows).copy_(tensors[index]);
        }
    }
    transTensorPB(target_pb, merged.contiguous());
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
            aux_info->set_memory_reuse_len(response.aux_info.memory_reuse_len);
            aux_info->set_prefill_total_reuse_len(response.aux_info.prefill_total_reuse_len);
            aux_info->set_prefill_local_reuse_len(response.aux_info.prefill_local_reuse_len);
            aux_info->set_prefill_remote_reuse_len(response.aux_info.prefill_remote_reuse_len);
            aux_info->set_prefill_memory_reuse_len(response.aux_info.prefill_memory_reuse_len);
            aux_info->set_decode_total_reuse_len(response.aux_info.decode_total_reuse_len);
            aux_info->set_decode_local_reuse_len(response.aux_info.decode_local_reuse_len);
            aux_info->set_decode_remote_reuse_len(response.aux_info.decode_remote_reuse_len);
            aux_info->set_decode_memory_reuse_len(response.aux_info.decode_memory_reuse_len);
            aux_info->set_aux_string(aux_string);
            if (response.aux_info.cum_log_probs.has_value()) {
                transTensorPB(aux_info->mutable_cum_log_probs(), response.aux_info.cum_log_probs.value());
            }
            if (response.aux_info.softmax_probs.has_value()) {
                transTensorPB(aux_info->mutable_softmax_probs(), response.aux_info.softmax_probs.value());
            }
        }
    }

    {
        std::vector<torch::Tensor> output_id_tensors;
        output_id_tensors.reserve(source_outputs.size());
        for (const auto& resp : source_outputs) {
            if (resp.output_ids.defined()) {
                output_id_tensors.push_back(resp.output_ids.contiguous());
            }
        }
        if (!output_id_tensors.empty()) {
            mergeAndPadTensorsToTensorPB<int32_t>(
                flatten_output->mutable_output_ids(), output_id_tensors, eos_token_id);
        }
    }

    stackBuffersToTensorPB(
        flatten_output->mutable_hidden_states(), source_outputs, [](const auto& r) { return r.hidden_states; });

    stackBuffersToTensorPB(flatten_output->mutable_loss(), source_outputs, [](const auto& r) { return r.loss; });

    stackBuffersToTensorPB(flatten_output->mutable_logits(), source_outputs, [](const auto& r) { return r.logits; });

    stackBuffersToTensorPB(
        flatten_output->mutable_all_hidden_states(), source_outputs, [](const auto& r) { return r.all_hidden_states; });

    // Logprob outputs are optional and absent for the common disabled path.
    // When enabled, placement metadata is emitted for every output. Compact
    // row counts may differ, so zero-count outputs get a zero-width tensor and
    // transport padding is removed by the RPC client using logprobs_counts.
    std::vector<torch::Tensor> token_logprobs;
    std::vector<torch::Tensor> top_logprob_token_ids;
    std::vector<torch::Tensor> top_logprobs;

    bool          logprobs_enabled = false;
    torch::Tensor token_reference;
    torch::Tensor top_token_reference;
    torch::Tensor top_logprobs_reference;
    for (const auto& response : source_outputs) {
        const bool has_token_logprobs = response.token_logprobs.has_value() && response.token_logprobs->defined();
        const bool has_top_token_ids =
            response.top_logprob_token_ids.has_value() && response.top_logprob_token_ids->defined();
        const bool has_top_logprobs = response.top_logprobs.has_value() && response.top_logprobs->defined();
        RTP_LLM_CHECK_WITH_INFO(has_token_logprobs == has_top_token_ids && has_token_logprobs == has_top_logprobs,
                                "Logprob tensors must be present or absent together.");
        logprobs_enabled =
            logprobs_enabled || has_token_logprobs || response.logprobs_offset != 0 || response.logprobs_count != 0;
        if (has_token_logprobs && !token_reference.defined()) {
            token_reference        = *response.token_logprobs;
            top_token_reference    = *response.top_logprob_token_ids;
            top_logprobs_reference = *response.top_logprobs;
        }
    }

    if (logprobs_enabled) {
        token_logprobs.reserve(source_outputs.size());
        top_logprob_token_ids.reserve(source_outputs.size());
        top_logprobs.reserve(source_outputs.size());
    }
    for (const auto& response : source_outputs) {
        if (!logprobs_enabled) {
            break;
        }

        const bool has_tensors = response.token_logprobs.has_value() && response.token_logprobs->defined();
        int64_t    offset      = response.logprobs_offset;
        int64_t    count       = response.logprobs_count;

        // Compatibility for callers constructed before placement metadata was
        // introduced: an aligned non-empty tensor still means offset=0.
        if (has_tensors && offset == 0 && count == 0 && response.token_logprobs->size(0) > 0) {
            count = response.token_logprobs->size(0);
        }
        RTP_LLM_CHECK_WITH_INFO(offset >= 0 && count >= 0, "logprobs offset/count must be non-negative");
        if (response.output_ids.defined()) {
            const int64_t output_token_count = response.output_ids.numel();
            RTP_LLM_CHECK_WITH_INFO(offset + count == output_token_count,
                                    "compact logprobs must cover one suffix of output_ids");
        }
        flatten_output->add_logprobs_offsets(static_cast<int32_t>(offset));
        flatten_output->add_logprobs_counts(static_cast<int32_t>(count));

        if (!has_tensors) {
            RTP_LLM_CHECK_WITH_INFO(count == 0, "positive logprobs_count requires compact tensors");
            if (token_reference.defined()) {
                token_logprobs.push_back(torch::empty({0}, token_reference.options()));
                top_logprob_token_ids.push_back(
                    torch::empty({0, top_token_reference.size(1)}, top_token_reference.options()));
                top_logprobs.push_back(
                    torch::empty({0, top_logprobs_reference.size(1)}, top_logprobs_reference.options()));
            }
            continue;
        }

        RTP_LLM_CHECK(response.token_logprobs->dim() == 1);
        RTP_LLM_CHECK(response.top_logprob_token_ids->dim() == 2);
        RTP_LLM_CHECK(response.top_logprobs->dim() == 2);
        RTP_LLM_CHECK(response.top_logprob_token_ids->size(0) == response.token_logprobs->size(0));
        RTP_LLM_CHECK(response.top_logprobs->sizes() == response.top_logprob_token_ids->sizes());

        RTP_LLM_CHECK_WITH_INFO(count == response.token_logprobs->size(0),
                                "logprobs_count must equal the compact tensor row count");
        token_logprobs.push_back(response.token_logprobs->contiguous());
        top_logprob_token_ids.push_back(response.top_logprob_token_ids->contiguous());
        top_logprobs.push_back(response.top_logprobs->contiguous());
    }

    auto serialize_optional_tensors =
        [&](const std::vector<torch::Tensor>& tensors, auto mutable_tensor_pb, const char* field_name) {
            if (tensors.empty()) {
                return;
            }
            RTP_LLM_CHECK_WITH_INFO(tensors.size() == source_outputs.size(),
                                    "Inconsistent %s tensor presence in a batch for stacking.",
                                    field_name);
            padAndStackLogprobTensorsToTensorPB(mutable_tensor_pb(), tensors);
        };
    serialize_optional_tensors(
        token_logprobs, [&]() { return flatten_output->mutable_token_logprobs(); }, "token_logprobs");
    serialize_optional_tensors(
        top_logprob_token_ids,
        [&]() { return flatten_output->mutable_top_logprob_token_ids(); },
        "top_logprob_token_ids");
    serialize_optional_tensors(top_logprobs, [&]() { return flatten_output->mutable_top_logprobs(); }, "top_logprobs");

    RTP_LLM_LOG_DEBUG("transResponse done");
}

}  // namespace rtp_llm
