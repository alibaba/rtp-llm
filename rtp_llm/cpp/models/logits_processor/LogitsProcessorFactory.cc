#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include <algorithm>
#include <cctype>
#include <mutex>
#include <optional>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
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

    const bool terminate_without_stop_token = key.key_type == "json";
    if (config->in_think_mode) {
        const int max_thinking_tokens =
            config->max_thinking_tokens < 0 ? config->max_new_tokens : config->max_thinking_tokens;
        auto matcher =
            backend->createMatcher(compiled, /*require_reasoning=*/false, std::nullopt, terminate_without_stop_token);
        return std::make_shared<ReasoningGrammarLogitsProcessor>(std::move(matcher),
                                                                 eos_token_id,
                                                                 max_thinking_tokens,
                                                                 config->begin_think_token_ids,
                                                                 config->end_think_token_ids,
                                                                 generate_input->inputLength(),
                                                                 std::move(error_reporter));
    }

    auto matcher =
        backend->createMatcher(compiled, /*require_reasoning=*/false, std::nullopt, terminate_without_stop_token);
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

void appendGrammarProcessor(std::vector<BaseLogitsProcessorPtr>&  result,
                            std::shared_ptr<GenerateInput>        generate_input,
                            int64_t                               eos_token_id,
                            const GrammarKeyCpp&                  grammar_key,
                            LogitsProcessorFactory::ErrorReporter error_reporter) {
    auto grammar_processor = createGrammarProcessor(generate_input, eos_token_id, grammar_key, error_reporter);
    if (grammar_processor != nullptr) {
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

    GrammarKeyCpp grammar_key;
    try {
        grammar_key = keyFromGenerateConfig(*config);
    } catch (const std::exception& e) {
        if (error_reporter) {
            error_reporter(
                ErrorCode::INVALID_PARAMS, std::string("invalid grammar response_format: ") + e.what(), false);
        }
        appendTreeAndMultiSeqProcessors(result, generate_input, init_batch_size, eos_token_id);
        return result;
    }

    if (grammar_key.empty()) {
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
