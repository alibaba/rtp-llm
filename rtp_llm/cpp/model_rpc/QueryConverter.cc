#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

#include <optional>
#include <stdexcept>

#include "RPCPool.h"
#include "autil/legacy/json.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
namespace {

using JsonMap = autil::legacy::json::JsonMap;

std::optional<std::string> extractSchemaFromJsonSchemaPayload(const std::string& payload) {
    try {
        const auto any_obj   = autil::legacy::json::ParseJson(payload);
        const auto obj_map   = autil::legacy::AnyCast<JsonMap>(any_obj);
        const auto schema_it = obj_map.find("schema");
        if (schema_it == obj_map.end()) {
            return std::nullopt;
        }
        try {
            const auto schema_str = autil::legacy::AnyCast<std::string>(schema_it->second);
            if (!schema_str.empty()) {
                return schema_str;
            }
        } catch (...) {}
        return autil::legacy::ToJsonString(schema_it->second, true);
    } catch (...) {
        return std::nullopt;
    }
}

// Mirrors rtp_llm/cpp/model_rpc/model_rpc_client.py::_normalize_grammar_fields.
// When a request arrives with response_format set but no concrete grammar
// field, we populate the matching grammar field so the engine can compile the
// constraint. Kept in sync with the Python-side normalizer for defense in
// depth — a non-Python client (or a Python path that bypasses the normalizer)
// still gets the same behavior as the OpenAI endpoint.
struct NormalizedGrammar {
    std::optional<std::string> json_schema;
    std::optional<std::string> regex;
    std::optional<std::string> ebnf;
    std::optional<std::string> structural_tag;

    bool empty() const {
        return !json_schema.has_value() && !regex.has_value() && !ebnf.has_value()
               && !structural_tag.has_value();
    }
};

// Read a field as a string, or serialize it back to JSON if it's a nested
// object. Empty strings are treated as absent so callers can `if (auto s = ...)`.
std::optional<std::string> extractJsonField(const JsonMap& map, const std::string& key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        return std::nullopt;
    }
    try {
        const auto str = autil::legacy::AnyCast<std::string>(it->second);
        if (!str.empty()) {
            return str;
        }
        return std::nullopt;
    } catch (...) {}
    try {
        return autil::legacy::ToJsonString(it->second, true);
    } catch (...) {
        return std::nullopt;
    }
}

// Parse the OpenAI-style response_format envelope. Per-type dispatch: one
// shorthand line per known type, no branch-by-branch error catalog. The single
// backstop at the end rejects anything that didn't yield a usable grammar
// field AND wasn't the explicit "text" sentinel, so unknown / malformed types
// produce INVALID_PARAMS instead of silently falling through as unconstrained.
NormalizedGrammar normalizeResponseFormat(const std::string& response_format_str) {
    NormalizedGrammar out;

    JsonMap response_map;
    try {
        const auto response_any = autil::legacy::json::ParseJson(response_format_str);
        response_map            = autil::legacy::AnyCast<JsonMap>(response_any);
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("response_format is not a valid JSON object: ") + e.what());
    }

    std::string format_type;
    if (const auto type_it = response_map.find("type"); type_it != response_map.end()) {
        try {
            format_type = autil::legacy::AnyCast<std::string>(type_it->second);
        } catch (...) {}
    }

    if (format_type == "text") {
        return out;  // OpenAI explicit "no constraint" sentinel.
    }

    if (format_type == "json_schema") {
        const auto json_schema_it = response_map.find("json_schema");
        if (json_schema_it != response_map.end()) {
            try {
                const auto json_schema_str = autil::legacy::AnyCast<std::string>(json_schema_it->second);
                if (!json_schema_str.empty()) {
                    out.json_schema = json_schema_str;
                }
            } catch (...) {}
            if (!out.json_schema.has_value()) {
                try {
                    const auto json_schema_map = autil::legacy::AnyCast<JsonMap>(json_schema_it->second);
                    out.json_schema            = extractJsonField(json_schema_map, "schema");
                } catch (...) {}
            }
        }
    } else if (format_type == "json_object") {
        // OpenAI "any valid JSON object" shorthand — mirrors the Python
        // client: config.json_schema = json.dumps({"type": "object"}).
        out.json_schema = R"({"type":"object"})";
    } else if (format_type == "regex") {
        out.regex = extractJsonField(response_map, "pattern");
    } else if (format_type == "ebnf") {
        out.ebnf = extractJsonField(response_map, "grammar");
    } else if (format_type == "structural_tag") {
        out.structural_tag = extractJsonField(response_map, "structural_tag");
    }

    if (out.empty()) {
        throw std::invalid_argument(
            "response_format did not yield a usable grammar field "
            "(check 'type' and the matching payload field)");
    }
    return out;
}

}  // namespace
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
    TRANS_OPTIONAL(json_schema);
    TRANS_OPTIONAL(regex);
    TRANS_OPTIONAL(ebnf);
    TRANS_OPTIONAL(structural_tag);
    TRANS_OPTIONAL(response_format);
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

    generate_config->reuse_cache         = config_proto->reuse_cache();
    generate_config->enable_device_cache = config_proto->enable_device_cache();
    generate_config->enable_memory_cache = config_proto->enable_memory_cache();
    generate_config->enable_remote_cache = config_proto->enable_remote_cache();
    TRANS_OPTIONAL(trace_id);
    TRANS_OPTIONAL(batch_group_timeout);
    TRANS_OPTIONAL(force_batch);

    // 生成式推荐：组合 token 约束
    generate_config->combo_token_size = config_proto->combo_token_size();
    for (const auto& combo_proto : config_proto->banned_combo_token_ids().rows()) {
        std::vector<int> combo;
        combo.reserve(combo_proto.values_size());
        for (const int value : combo_proto.values()) {
            combo.push_back(value);
        }
        generate_config->banned_combo_token_ids.push_back(std::move(combo));
    }

    if (generate_config->json_schema.has_value()) {
        auto normalized_schema = extractSchemaFromJsonSchemaPayload(generate_config->json_schema.value());
        if (normalized_schema.has_value()) {
            generate_config->json_schema = normalized_schema.value();
        }
    }
    // Mirror the Python normalizer: only fall back to response_format if NO
    // concrete grammar field was supplied. If any one was set, respect the
    // client's explicit choice.
    const bool any_grammar_set = generate_config->json_schema.has_value()
                                 || generate_config->regex.has_value()
                                 || generate_config->ebnf.has_value()
                                 || generate_config->structural_tag.has_value();
    if (!any_grammar_set && config_proto->has_response_format()) {
        auto norm = normalizeResponseFormat(config_proto->response_format().value());
        if (norm.json_schema.has_value()) {
            generate_config->json_schema = std::move(*norm.json_schema);
        }
        if (norm.regex.has_value()) {
            generate_config->regex = std::move(*norm.regex);
        }
        if (norm.ebnf.has_value()) {
            generate_config->ebnf = std::move(*norm.ebnf);
        }
        if (norm.structural_tag.has_value()) {
            generate_config->structural_tag = std::move(*norm.structural_tag);
        }
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

    RTP_LLM_LOG_DEBUG("transResponse done");
}

}  // namespace rtp_llm
