#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/CompletionBoundaryLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using JsonMap   = autil::legacy::json::JsonMap;
using JsonArray = autil::legacy::json::JsonArray;

std::mutex            g_grammar_backend_mutex;
XGrammarBackendCppPtr g_grammar_backend;

struct PretokenizedChatConstraints {
    std::vector<int32_t> reasoning_prompt_tail_token_ids;
    std::vector<int32_t> response_prompt_tail_token_ids;
    std::string          reasoning_structural_tag;
    std::string          response_structural_tag;
    std::vector<int32_t> reasoning_completion_boundary_token_ids;
    std::vector<int32_t> response_completion_boundary_token_ids;

    bool configured() const {
        return !reasoning_structural_tag.empty() || !response_structural_tag.empty()
               || !reasoning_completion_boundary_token_ids.empty()
               || !response_completion_boundary_token_ids.empty();
    }
};

struct PretokenizedConstraintSelection {
    std::string          source = "none";
    std::vector<int32_t> completion_boundary_token_ids;
};

PretokenizedChatConstraints g_pretokenized_chat_constraints;

bool inputEndsWith(const torch::Tensor& input_ids, const std::vector<int32_t>& suffix) {
    if (suffix.empty() || !input_ids.defined() || !input_ids.device().is_cpu()
        || input_ids.scalar_type() != torch::kInt32 || input_ids.numel() < static_cast<int64_t>(suffix.size())) {
        return false;
    }
    const auto  contiguous = input_ids.contiguous();
    const auto* data       = contiguous.data_ptr<int32_t>();
    const auto  offset     = contiguous.numel() - static_cast<int64_t>(suffix.size());
    return std::equal(suffix.begin(), suffix.end(), data + offset);
}

std::string formatInputTail(const torch::Tensor& input_ids, size_t max_tokens) {
    if (!input_ids.defined() || !input_ids.device().is_cpu() || input_ids.scalar_type() != torch::kInt32) {
        return "unavailable";
    }
    const auto         contiguous = input_ids.contiguous();
    const auto*        data       = contiguous.data_ptr<int32_t>();
    const auto         count      = static_cast<size_t>(contiguous.numel());
    const auto         begin      = count > max_tokens ? count - max_tokens : 0;
    std::ostringstream oss;
    oss << "[";
    for (size_t i = begin; i < count; ++i) {
        if (i != begin) {
            oss << ",";
        }
        oss << data[i];
    }
    oss << "]";
    return oss.str();
}

PretokenizedConstraintSelection
applyPretokenizedChatConstraint(std::shared_ptr<GenerateInput> generate_input,
                                const PretokenizedChatConstraints& defaults) {
    auto                        config = generate_input->generate_config;
    const std::string*          structural_tag = nullptr;
    const std::vector<int32_t>* completion_boundary_token_ids = nullptr;
    PretokenizedConstraintSelection selection;
    if (inputEndsWith(generate_input->input_ids, defaults.reasoning_prompt_tail_token_ids)) {
        structural_tag                = &defaults.reasoning_structural_tag;
        completion_boundary_token_ids = &defaults.reasoning_completion_boundary_token_ids;
        selection.source              = "prompt_tail_reasoning";
    } else if (inputEndsWith(generate_input->input_ids, defaults.response_prompt_tail_token_ids)) {
        structural_tag                = &defaults.response_structural_tag;
        completion_boundary_token_ids = &defaults.response_completion_boundary_token_ids;
        selection.source              = "prompt_tail_response";
    }
    if (structural_tag == nullptr) {
        return selection;
    }

    if (!structural_tag->empty()) {
        config->structural_tag = *structural_tag;
    } else if (completion_boundary_token_ids != nullptr && !completion_boundary_token_ids->empty()) {
        selection.completion_boundary_token_ids = *completion_boundary_token_ids;
        selection.source += "_completion_guard";
    } else {
        selection.source = "none";
        return selection;
    }
    config->in_think_mode       = false;
    config->max_thinking_tokens = 0;
    config->begin_think_token_ids.clear();
    config->end_think_token_ids.clear();
    return selection;
}

std::string anyToString(const autil::legacy::Any& any) {
    if (auto str = autil::legacy::AnyCast<std::string>(&any)) {
        return *str;
    }
    return autil::legacy::json::ToString(any, true);
}

std::optional<std::string> getFieldAsString(const JsonMap& map, const std::string& name) {
    auto it = map.find(name);
    if (it == map.end()) {
        return std::nullopt;
    }
    return anyToString(it->second);
}

std::optional<std::string> getType(const JsonMap& map) {
    auto it = map.find("type");
    if (it == map.end()) {
        return std::nullopt;
    }
    auto str = autil::legacy::AnyCast<std::string>(&it->second);
    if (!str) {
        return std::nullopt;
    }
    return *str;
}

std::optional<std::string> extractJsonSchemaFromEnvelope(const JsonMap& response_map) {
    auto schema_it = response_map.find("json_schema");
    if (schema_it == response_map.end()) {
        return std::nullopt;
    }
    if (auto schema_str = autil::legacy::AnyCast<std::string>(&schema_it->second)) {
        return *schema_str;
    }
    auto schema_map = autil::legacy::AnyCast<JsonMap>(&schema_it->second);
    if (!schema_map) {
        return anyToString(schema_it->second);
    }
    auto schema = getFieldAsString(*schema_map, "schema");
    return schema.has_value() ? schema : std::make_optional(anyToString(schema_it->second));
}

GrammarKeyCpp keyFromResponseFormat(const std::string& response_format) {
    autil::legacy::Any any;
    autil::legacy::json::ParseJson(response_format, any);
    auto* response_map = autil::legacy::AnyCast<JsonMap>(&any);
    if (!response_map) {
        auto* response_array = autil::legacy::AnyCast<JsonArray>(&any);
        if (response_array) {
            if (response_array->empty()) {
                return {};
            }
            if (response_array->size() != 1) {
                throw std::invalid_argument("response_format array must contain exactly one JSON object");
            }
            response_map = autil::legacy::AnyCast<JsonMap>(&(*response_array)[0]);
        }
    }
    if (!response_map) {
        throw std::invalid_argument("response_format must be a JSON object");
    }

    auto type = getType(*response_map);
    if (!type.has_value() || *type == "text") {
        return {};
    }
    if (*type == "json_object") {
        return {"json", R"({"type":"object"})"};
    }
    if (*type == "json_schema") {
        auto schema = extractJsonSchemaFromEnvelope(*response_map);
        return schema.has_value() ? GrammarKeyCpp{"json", *schema} : GrammarKeyCpp{};
    }
    if (*type == "regex") {
        auto pattern = getFieldAsString(*response_map, "pattern");
        return pattern.has_value() ? GrammarKeyCpp{"regex", *pattern} : GrammarKeyCpp{};
    }
    if (*type == "ebnf") {
        auto grammar = getFieldAsString(*response_map, "grammar");
        return grammar.has_value() ? GrammarKeyCpp{"ebnf", *grammar} : GrammarKeyCpp{};
    }
    if (*type == "structural_tag") {
        auto tag = getFieldAsString(*response_map, "structural_tag");
        return tag.has_value() ? GrammarKeyCpp{"structural_tag", *tag} : GrammarKeyCpp{};
    }
    throw std::invalid_argument("unknown response_format.type: " + *type);
}

GrammarKeyCpp keyFromGenerateConfig(const GenerateConfig& config) {
    if (config.json_schema.has_value()) {
        return {"json", config.json_schema.value()};
    }
    if (config.regex.has_value()) {
        return {"regex", config.regex.value()};
    }
    if (config.ebnf.has_value()) {
        return {"ebnf", config.ebnf.value()};
    }
    if (config.structural_tag.has_value()) {
        return {"structural_tag", config.structural_tag.value()};
    }
    if (config.response_format.has_value()) {
        return keyFromResponseFormat(config.response_format.value());
    }
    return {};
}

BaseLogitsProcessorPtr createGrammarProcessor(std::shared_ptr<GenerateInput>        generate_input,
                                              int64_t                               eos_token_id,
                                              const GrammarKeyCpp&                  key,
                                              LogitsProcessorFactory::ErrorReporter error_reporter) {
    auto                  config = generate_input->generate_config;
    XGrammarBackendCppPtr backend;
    {
        std::lock_guard<std::mutex> lock(g_grammar_backend_mutex);
        backend = g_grammar_backend;
    }
    if (!backend) {
        if (error_reporter) {
            error_reporter(
                ErrorCode::INVALID_PARAMS, "grammar request rejected: xgrammar backend is not initialized", false);
        }
        return nullptr;
    }

    auto invalid = backend->getCachedInvalid(key);
    if (!invalid.empty()) {
        if (error_reporter) {
            error_reporter(ErrorCode::INVALID_PARAMS, "failed to compile grammar: " + invalid, false);
        }
        return nullptr;
    }

    auto compiled = backend->getCached(key);
    if (!compiled) {
        auto result = backend->compileNow(key);
        if (!result.compiled) {
            backend->setCacheInvalid(key, result.error_message);
            if (error_reporter) {
                error_reporter(ErrorCode::INVALID_PARAMS, "failed to compile grammar: " + result.error_message, false);
            }
            return nullptr;
        }
        compiled = result.compiled;
        backend->setCache(key, compiled);
    }

    const bool       terminate_without_stop_token = key.key_type == "json";
    std::vector<int> request_stop_tokens;
    for (const auto& stop_word : config->stop_words_list) {
        if (stop_word.size() == 1) {
            request_stop_tokens.push_back(stop_word[0]);
        }
    }
    const auto request_stops = std::optional<std::vector<int>>(std::move(request_stop_tokens));
    if (config->in_think_mode) {
        auto matcher = backend->createMatcher(
            compiled, /*require_reasoning=*/false, std::nullopt, terminate_without_stop_token, request_stops);
        return std::make_shared<ReasoningGrammarLogitsProcessor>(std::move(matcher),
                                                                 eos_token_id,
                                                                 config->max_thinking_tokens,
                                                                 config->begin_think_token_ids,
                                                                 config->end_think_token_ids,
                                                                 generate_input->inputLength(),
                                                                 std::move(error_reporter));
    }

    auto matcher = backend->createMatcher(
        compiled, /*require_reasoning=*/false, std::nullopt, terminate_without_stop_token, request_stops);
    return std::make_shared<GrammarLogitsProcessor>(std::move(matcher), eos_token_id, std::move(error_reporter));
}

void appendThinkProcessor(std::vector<BaseLogitsProcessorPtr>& result,
                          std::shared_ptr<GenerateInput>       generate_input,
                          int32_t                              max_batch_size) {
    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(generate_input, max_batch_size);
    if (think_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }
}

void appendCompletionBoundaryProcessor(std::vector<BaseLogitsProcessorPtr>& result,
                                       std::shared_ptr<GenerateInput>       generate_input,
                                       int32_t                              max_batch_size,
                                       int64_t                              eos_token_id,
                                       const std::vector<int32_t>&          completion_boundary_token_ids) {
    if (completion_boundary_token_ids.empty()) {
        return;
    }
    std::vector<int32_t> guarded_stop_token_ids;
    if (eos_token_id >= 0 && eos_token_id <= std::numeric_limits<int32_t>::max()) {
        guarded_stop_token_ids.push_back(static_cast<int32_t>(eos_token_id));
    }
    for (const auto& stop_word : generate_input->generate_config->stop_words_list) {
        if (stop_word.size() == 1) {
            guarded_stop_token_ids.push_back(stop_word.front());
        }
    }

    const bool beam_search = generate_input->generate_config->hasNumBeams()
                             || generate_input->generate_config->num_return_sequences > 1;
    std::vector<CompletionBoundaryState> states;
    states.reserve(max_batch_size);
    for (int32_t i = 0; i < max_batch_size; ++i) {
        states.emplace_back(completion_boundary_token_ids, generate_input->inputLength(), beam_search);
    }
    result.push_back(std::make_shared<CompletionBoundaryLogitsProcessor>(std::move(states),
                                                                         guarded_stop_token_ids,
                                                                         generate_input->request_id,
                                                                         generate_input->request_info.trace_id));
    RTP_LLM_LOG_INFO(
        "completion boundary guard created: request_id=%ld trace_id=%s boundary_tokens=%zu guarded_stops=%zu",
        generate_input->request_id,
        generate_input->request_info.trace_id.c_str(),
        completion_boundary_token_ids.size(),
        guarded_stop_token_ids.size());
}

void appendGrammarProcessor(std::vector<BaseLogitsProcessorPtr>&  result,
                            std::shared_ptr<GenerateInput>        generate_input,
                            int64_t                               eos_token_id,
                            const GrammarKeyCpp&                  grammar_key,
                            LogitsProcessorFactory::ErrorReporter error_reporter) {
    auto grammar_processor = createGrammarProcessor(generate_input, eos_token_id, grammar_key, error_reporter);
    if (grammar_processor != nullptr) {
        RTP_LLM_LOG_INFO("grammar processor created: request_id=%ld trace_id=%s key_type=%s input_len=%d",
                         generate_input->request_id,
                         generate_input->request_info.trace_id.c_str(),
                         grammar_key.key_type.c_str(),
                         generate_input->inputLength());
        result.push_back(std::move(grammar_processor));
    }
}

void appendTreeAndMultiSeqProcessors(std::vector<BaseLogitsProcessorPtr>& result,
                                     std::shared_ptr<GenerateInput>       generate_input,
                                     int32_t                              init_batch_size,
                                     int64_t                              eos_token_id) {
    auto tree_processor = TreeLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (tree_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_processor));
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(generate_input, eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_processor));
    }
}

}  // namespace

bool LogitsProcessorFactory::hasGrammarConstraint(const GenerateConfig& config) {
    try {
        return !keyFromGenerateConfig(config).empty();
    } catch (const std::exception&) {
        return true;
    }
}

void LogitsProcessorFactory::init(const std::string&   ckpt_path,
                                  const std::string&   tree_decode_config,
                                  const GrammarConfig& grammar_config) {
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(ckpt_path, tree_decode_config);

    std::lock_guard<std::mutex> lock(g_grammar_backend_mutex);
    g_grammar_backend.reset();
    g_pretokenized_chat_constraints = {
        grammar_config.reasoning_prompt_tail_token_ids,
        grammar_config.response_prompt_tail_token_ids,
        grammar_config.reasoning_structural_tag,
        grammar_config.response_structural_tag,
        grammar_config.reasoning_completion_boundary_token_ids,
        grammar_config.response_completion_boundary_token_ids,
    };

    auto backend_name = grammar_config.grammar_backend;
    std::transform(backend_name.begin(), backend_name.end(), backend_name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (backend_name.empty() || backend_name == "none") {
        RTP_LLM_LOG_INFO("xgrammar backend disabled by grammar_backend=%s", grammar_config.grammar_backend.c_str());
        return;
    }
    if (backend_name != "xgrammar") {
        RTP_LLM_LOG_WARNING("unknown grammar_backend=%s; grammar disabled", grammar_config.grammar_backend.c_str());
        return;
    }
    if (grammar_config.tokenizer_info_json.empty()) {
        RTP_LLM_LOG_WARNING("xgrammar backend disabled: tokenizer_info_json is empty");
        return;
    }

    XGrammarBackendOptions options;
    options.any_whitespace       = !grammar_config.constrained_json_disable_any_whitespace;
    options.max_compiler_threads = std::max(1, grammar_config.num_workers);
    if (!grammar_config.override_stop_tokens.empty()) {
        options.override_stop_tokens =
            std::vector<int>(grammar_config.override_stop_tokens.begin(), grammar_config.override_stop_tokens.end());
    }
    try {
        g_grammar_backend = std::make_shared<XGrammarBackendCpp>(grammar_config.tokenizer_info_json, options);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("failed to initialize xgrammar backend: %s", e.what());
    }
}

std::vector<BaseLogitsProcessorPtr>
LogitsProcessorFactory::createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                               int32_t                        init_batch_size,
                                               int32_t                        max_batch_size,
                                               int64_t                        eos_token_id,
                                               ErrorReporter                  error_reporter) {
    std::vector<BaseLogitsProcessorPtr> result;
    auto                                config = generate_input->generate_config;

    const bool   in_think_mode_before        = config->in_think_mode;
    const size_t begin_think_ids_size_before = config->begin_think_token_ids.size();
    const size_t end_think_ids_size_before   = config->end_think_token_ids.size();

    GrammarKeyCpp grammar_key;
    std::string          constraint_source = "request";
    std::vector<int32_t> completion_boundary_token_ids;
    try {
        grammar_key = keyFromGenerateConfig(*config);
        if (grammar_key.empty()) {
            PretokenizedChatConstraints defaults;
            {
                std::lock_guard<std::mutex> lock(g_grammar_backend_mutex);
                defaults = g_pretokenized_chat_constraints;
            }
            auto selection = applyPretokenizedChatConstraint(generate_input, defaults);
            constraint_source = selection.source;
            completion_boundary_token_ids = std::move(selection.completion_boundary_token_ids);
            if (constraint_source != "none") {
                if (config->structural_tag.has_value()) {
                    grammar_key = {"structural_tag", config->structural_tag.value()};
                }
            } else if (in_think_mode_before) {
                constraint_source = "legacy_think";
            }
            if (defaults.configured() || in_think_mode_before) {
                const size_t configured_tail_size = std::max(defaults.reasoning_prompt_tail_token_ids.size(),
                                                             defaults.response_prompt_tail_token_ids.size());
                RTP_LLM_LOG_INFO(
                    "constraint selection: request_id=%ld trace_id=%s source=%s key_type=%s "
                    "in_think_mode_before=%d in_think_mode_after=%d begin_think_ids=%zu end_think_ids=%zu "
                    "completion_boundary_ids=%zu input_len=%d input_tail_ids=%s",
                    generate_input->request_id,
                    generate_input->request_info.trace_id.c_str(),
                    constraint_source.c_str(),
                    grammar_key.key_type.empty() ? "none" : grammar_key.key_type.c_str(),
                    in_think_mode_before,
                    config->in_think_mode,
                    begin_think_ids_size_before,
                    end_think_ids_size_before,
                    completion_boundary_token_ids.size(),
                    generate_input->inputLength(),
                    formatInputTail(generate_input->input_ids, std::max<size_t>(8, configured_tail_size)).c_str());
            }
        } else {
            RTP_LLM_LOG_INFO("constraint selection: request_id=%ld trace_id=%s source=request key_type=%s "
                             "in_think_mode=%d begin_think_ids=%zu end_think_ids=%zu input_len=%d",
                             generate_input->request_id,
                             generate_input->request_info.trace_id.c_str(),
                             grammar_key.key_type.c_str(),
                             config->in_think_mode,
                             config->begin_think_token_ids.size(),
                             config->end_think_token_ids.size(),
                             generate_input->inputLength());
        }
    } catch (const std::exception& e) {
        if (error_reporter) {
            error_reporter(
                ErrorCode::INVALID_PARAMS, std::string("invalid grammar response_format: ") + e.what(), false);
        }
        appendTreeAndMultiSeqProcessors(result, generate_input, init_batch_size, eos_token_id);
        return result;
    }

    if (!completion_boundary_token_ids.empty()) {
        appendCompletionBoundaryProcessor(
            result, generate_input, max_batch_size, eos_token_id, completion_boundary_token_ids);
    } else if (grammar_key.empty()) {
        appendThinkProcessor(result, generate_input, max_batch_size);
    } else if (config->in_think_mode) {
        if (config->hasNumBeams() || config->num_return_sequences > 1) {
            if (error_reporter) {
                error_reporter(ErrorCode::INVALID_PARAMS,
                               "grammar-constrained decoding does not support beam search or num_return_sequences > 1",
                               false);
            }
        } else if (config->end_think_token_ids.empty()) {
            if (error_reporter) {
                error_reporter(ErrorCode::INVALID_PARAMS,
                               "grammar-constrained thinking requires non-empty end_think_token_ids",
                               false);
            }
        } else {
            appendGrammarProcessor(result, generate_input, eos_token_id, grammar_key, error_reporter);
        }
    } else if (config->hasNumBeams() || config->num_return_sequences > 1) {
        if (error_reporter) {
            error_reporter(ErrorCode::INVALID_PARAMS,
                           "grammar-constrained decoding does not support beam search or num_return_sequences > 1",
                           false);
        }
    } else {
        appendGrammarProcessor(result, generate_input, eos_token_id, grammar_key, error_reporter);
    }

    appendTreeAndMultiSeqProcessors(result, generate_input, init_batch_size, eos_token_id);
    return result;
}

}  // namespace rtp_llm
