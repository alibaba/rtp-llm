#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

#include <numeric>

#include "RPCPool.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
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
    generate_config->num_return_sequences = config_proto->num_return_sequences();
    generate_config->return_logits        = config_proto->return_logits();
    generate_config->return_prompt_logits = config_proto->return_prompt_logits();
    if (generate_config->return_prompt_logits) {
        generate_config->prompt_logits_top_k =
            config_proto->prompt_logits_top_k() > 0 ? config_proto->prompt_logits_top_k() : 64;
        generate_config->prompt_logits_start = config_proto->prompt_logits_start();
        generate_config->prompt_logits_end   = config_proto->prompt_logits_end();
        // Python client (GenerateConfig.return_target_logprob defaults true) always sets this
        // explicitly. proto3 unset bool = false, which is acceptable for raw gRPC clients
        // that explicitly opt out of target_logprobs.
        generate_config->return_target_logprob = config_proto->return_target_logprob();
    }
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
    if (config_proto->return_all_probs_mode() != 0) {
        // new client: explicit mode (offset 1). Clamp out-of-range values to NONE
        // so a malformed client can't synthesize an undefined ReturnAllProbsMode.
        int mode = config_proto->return_all_probs_mode() - 1;
        if (mode < static_cast<int>(ReturnAllProbsMode::NONE)
            || mode > static_cast<int>(ReturnAllProbsMode::ORIGINAL)) {
            mode = static_cast<int>(ReturnAllProbsMode::NONE);
        }
        generate_config->return_all_probs = static_cast<ReturnAllProbsMode>(mode);
    } else {
        // legacy client: only bool field set
        generate_config->return_all_probs =
            config_proto->return_all_probs() ? ReturnAllProbsMode::DEFAULT : ReturnAllProbsMode::NONE;
    }
    generate_config->return_softmax_probs  = config_proto->return_softmax_probs();
    generate_config->can_use_pd_separation = config_proto->can_use_pd_separation();
    generate_config->gen_timeline          = config_proto->gen_timeline();
    generate_config->profile_step          = config_proto->profile_step();
    generate_config->profile_trace_name    = config_proto->profile_trace_name();
    generate_config->ignore_eos            = config_proto->ignore_eos();
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

    // 生成式推荐：组合 token 约束
    generate_config->combo_token_size = config_proto->combo_token_size();
    generate_config->enable_cross_sequence_ban = config_proto->enable_cross_sequence_ban();
    generate_config->cross_seq_diverge_start_combo = config_proto->cross_seq_diverge_start_combo();
    for (const auto& combo_proto : config_proto->banned_combo_token_ids().rows()) {
        std::vector<int> combo;
        combo.reserve(combo_proto.values_size());
        for (const int value : combo_proto.values()) {
            combo.push_back(value);
        }
        generate_config->banned_combo_token_ids.push_back(std::move(combo));
    }

    return generate_config;
}

std::shared_ptr<GenerateInput> QueryConverter::transQuery(const GenerateInputPB* input) {
    std::shared_ptr<GenerateInput> generate_input = std::make_shared<GenerateInput>();
    generate_input->request_id                    = input->request_id();
    generate_input->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
    if (input->has_generate_config()) {
        generate_input->generate_config = transGenerateConfig(&(input->generate_config()));
    }
    generate_input->input_ids =
        torch::from_blob(const_cast<int*>(input->token_ids().data()), {(int64_t)input->token_ids_size()}, torch::kInt32)
            .clone();
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
    generate_input->batch_group_size = input->batch_group_size() > 0 ? input->batch_group_size() : 1;
    if (input->has_batch_group_id()) {
        generate_input->batch_group_id = input->batch_group_id().value();
    }
    if (input->has_request_info()) {
        const auto& info_pb                  = input->request_info();
        generate_input->request_info.frontend_ip = info_pb.frontend_ip();
        generate_input->request_info.dash_ip     = info_pb.dash_ip();
        generate_input->request_info.trace_id    = info_pb.trace_id();
        generate_input->request_info.request_id  = info_pb.request_id();
        generate_input->request_info.source_role = info_pb.source_role();
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
        transTensorPB(now_input->mutable_multimodal_tensor(), mm_input.tensor);
        transMMPreprocessConfig(now_input->mutable_mm_preprocess_config(), mm_input.mm_preprocess_config);
    }
    return mm_inputs_pb;
}

void QueryConverter::transMMPreprocessConfig(MMPreprocessConfigPB* config_pb, const MMPreprocessConfig& config) {
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
    torch::Tensor mm_embedding        = transTensor(output_pb->multimodal_embedding()), mm_position_id;
    bool          contain_pos         = output_pb->has_multimodal_pos_id();
    bool          contain_extra_input = output_pb->multimodal_extra_input_size() > 0;
    if (contain_pos) {
        mm_position_id = transTensor(output_pb->multimodal_pos_id());
    }
    MultimodalOutput     mm_output;
    std::vector<int64_t> split_sizes;
    for (auto split_size : output_pb->split_size()) {
        split_sizes.push_back(split_size);
    }
    const int64_t split_total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
    RTP_LLM_CHECK_WITH_INFO(!split_sizes.empty() && split_total == mm_embedding.size(0),
                            "split_sizes sum=%ld does not match mm_embedding.size(0)=%ld",
                            split_total,
                            mm_embedding.size(0));
    mm_output.mm_features = mm_embedding.split(split_sizes, 0);
    if (contain_pos) {
        RTP_LLM_CHECK_WITH_INFO(split_total == mm_position_id.size(0),
                                "split_sizes sum=%ld does not match mm_position_id.size(0)=%ld",
                                split_total,
                                mm_position_id.size(0));
        mm_output.mm_position_ids = mm_position_id.split(split_sizes, 0);
    }

    if (contain_extra_input) {
        // Each extra-input is an opaque flat 1-D tensor (one per image), reshaped by the
        // model-specific consumer; no split needed here.
        std::vector<torch::Tensor> extra_inputs;
        extra_inputs.reserve(output_pb->multimodal_extra_input_size());
        for (const auto& extra_input_pb : output_pb->multimodal_extra_input()) {
            extra_inputs.emplace_back(transTensor(extra_input_pb));
        }
        mm_output.mm_extra_input = std::move(extra_inputs);
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
            auto* mm_map = aux_info->mutable_multimodal_lengths();
            for (const auto& [key, value] : response.aux_info.multimodal_lengths) {
                (*mm_map)[key] = value;
            }
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
        flatten_output->mutable_all_probs(), source_outputs, [](const auto& r) { return r.aux_info.all_probs; });
    stackBuffersToTensorPB(
        flatten_output->mutable_hidden_states(), source_outputs, [](const auto& r) { return r.hidden_states; });

    stackBuffersToTensorPB(flatten_output->mutable_loss(), source_outputs, [](const auto& r) { return r.loss; });

    stackBuffersToTensorPB(flatten_output->mutable_logits(), source_outputs, [](const auto& r) { return r.logits; });

    stackBuffersToTensorPB(
        flatten_output->mutable_all_hidden_states(), source_outputs, [](const auto& r) { return r.all_hidden_states; });

    if (!source_outputs.empty() && source_outputs[0].prompt_logits.has_value()) {
        auto*       pb = flatten_output->mutable_prompt_logits();
        const auto& pl = source_outputs[0].prompt_logits.value();
        transTensorPB(pb->mutable_topk_logprobs(), pl.topk_logprobs);
        transTensorPB(pb->mutable_topk_token_ids(), pl.topk_token_ids);
        if (pl.target_logprobs.defined()) {
            transTensorPB(pb->mutable_target_logprobs(), pl.target_logprobs);
        }
        pb->set_start_pos(pl.start_pos);
        pb->set_end_pos(pl.end_pos);
    }

    RTP_LLM_LOG_DEBUG("transResponse done");
}

}  // namespace rtp_llm
