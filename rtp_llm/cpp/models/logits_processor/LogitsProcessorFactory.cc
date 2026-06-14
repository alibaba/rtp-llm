#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/GrammarSchemaValidator.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using JsonMap   = autil::legacy::json::JsonMap;
using JsonArray = autil::legacy::json::JsonArray;

std::mutex                       g_grammar_backend_mutex;
std::shared_ptr<XGrammarBackend> g_grammar_backend;

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
    if (response_format.empty()) {
        return {};
    }
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
    // Fixed priority json_schema > regex > ebnf > structural_tag silently drops the
    // others when a caller sets multiple. Warn loudly so we can audit the line and
    // eventually harden to invalid_argument once no live caller relies on this.
    const int grammar_field_count = static_cast<int>(config.json_schema.has_value())
                                    + static_cast<int>(config.regex.has_value())
                                    + static_cast<int>(config.ebnf.has_value())
                                    + static_cast<int>(config.structural_tag.has_value());
    if (grammar_field_count > 1) {
        RTP_LLM_LOG_WARNING(
            "GenerateConfig sets %d grammar constraints simultaneously "
            "(json_schema=%d, regex=%d, ebnf=%d, structural_tag=%d); only the highest-priority "
            "one (json_schema>regex>ebnf>structural_tag) is applied — drop the rest at the client",
            grammar_field_count,
            static_cast<int>(config.json_schema.has_value()),
            static_cast<int>(config.regex.has_value()),
            static_cast<int>(config.ebnf.has_value()),
            static_cast<int>(config.structural_tag.has_value()));
    }
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

// Compile + matcher creation, given an already-resolved GrammarKeyCpp. Works
// off the pre-parsed key so the factory can validate response_format once for
// both the in-think and plain-grammar paths.
std::shared_ptr<RtpGrammarMatcher>
compileMatcherFromKey(XGrammarBackend&                            backend,
                      const GrammarKeyCpp&                        key,
                      bool                                        require_reasoning,
                      std::optional<std::vector<int>>             think_end_token_ids,
                      const LogitsProcessorFactory::ErrorReporter& error_reporter) {
    auto validate = validateGrammarKey(key);
    if (validate.status != GrammarValidateStatus::Ok) {
        backend.setCacheInvalid(key, validate.detail);
        reportInvalidParams(error_reporter, "Failed to compile " + key.key_type + " grammar: " + validate.detail);
        return nullptr;
    }

    CompileResult result;
    try {
        result = backend.getOrCompile(key);
    } catch (const std::exception& e) {
        reportInvalidParams(error_reporter, std::string("grammar compile error: ") + e.what());
        return nullptr;
    }
    if (result.timed_out) {
        // Request-side gave up waiting; the background compile keeps running and
        // the next request for this key may benefit. Surface as GENERATE_TIMEOUT
        // (not INVALID_PARAMS) since the schema may be valid — we just refused to wait.
        reportGenerateTimeout(error_reporter, "grammar compile timeout: " + result.error_message);
        return nullptr;
    }
    if (!result.compiled) {
        const std::string err = result.error_message.empty() ? "unknown compile error" : result.error_message;
        reportInvalidParams(error_reporter, "Failed to compile " + key.key_type + " grammar: " + err);
        return nullptr;
    }

    const bool                         terminate_without_stop_token = key.key_type == "json";
    std::shared_ptr<RtpGrammarMatcher> matcher                      = backend.createMatcher(
        result.compiled, require_reasoning, std::move(think_end_token_ids), terminate_without_stop_token);
    if (!matcher) {
        reportInvalidParams(error_reporter, "grammar matcher install failed");
        return nullptr;
    }
    return matcher;
}

BaseLogitsProcessorPtr createGrammarProcessor(const std::shared_ptr<XGrammarBackend>&     backend,
                                              const std::shared_ptr<GenerateInput>&       input,
                                              const GrammarKeyCpp&                        key,
                                              int64_t                                     eos_token_id,
                                              const LogitsProcessorFactory::ErrorReporter& error_reporter) {
    if (!input || !input->generate_config || key.empty()) {
        return nullptr;
    }
    auto& config = *input->generate_config;
    if (!backend) {
        reportInvalidParams(error_reporter,
                            "structured output requested but constraint backend is disabled "
                            "(check engine startup logs: tokenizer info empty or backend init failed).");
        return nullptr;
    }

    if (config.in_think_mode) {
        if (config.end_think_token_ids.empty()) {
            reportInvalidParams(error_reporter,
                                "structured output with in_think_mode requires non-empty end_think_token_ids");
            return nullptr;
        }
        auto matcher = compileMatcherFromKey(*backend,
                                             key,
                                             /*require_reasoning=*/true,
                                             std::optional<std::vector<int>>(config.end_think_token_ids),
                                             error_reporter);
        if (!matcher) {
            return nullptr;
        }
        return std::make_shared<ReasoningGrammarLogitsProcessor>(std::move(matcher),
                                                                 eos_token_id,
                                                                 config.max_thinking_tokens,
                                                                 config.begin_think_token_ids,
                                                                 config.end_think_token_ids,
                                                                 input->inputLength(),
                                                                 error_reporter);
    }

    auto matcher = compileMatcherFromKey(*backend,
                                         key,
                                         /*require_reasoning=*/false,
                                         /*think_end_token_ids=*/std::nullopt,
                                         error_reporter);
    if (!matcher) {
        return nullptr;
    }
    return std::make_shared<GrammarLogitsProcessor>(std::move(matcher), eos_token_id, error_reporter);
}

}  // namespace

bool LogitsProcessorFactory::hasGrammarConstraint(const GenerateConfig& config) {
    try {
        return !keyFromGenerateConfig(config).empty();
    } catch (const std::exception&) {
        // Malformed response_format counts as a grammar request — defer to the
        // createLogitsProcessors path which surfaces the parse error properly.
        return true;
    }
}

void LogitsProcessorFactory::init(const std::string&   ckpt_path,
                                  const std::string&   tree_decode_config,
                                  const GrammarConfig& grammar_config) {
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(ckpt_path, tree_decode_config);

    std::lock_guard<std::mutex> lock(g_grammar_backend_mutex);
    g_grammar_backend.reset();
    g_grammar_backend = XGrammarBackend::fromConfig(grammar_config);
}

std::vector<BaseLogitsProcessorPtr>
LogitsProcessorFactory::createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                               int32_t                        init_batch_size,
                                               int32_t                        max_batch_size,
                                               int64_t                        eos_token_id,
                                               const ErrorReporter&           error_reporter) {
    std::vector<BaseLogitsProcessorPtr> result;

    auto& config = *generate_input->generate_config;

    GrammarKeyCpp grammar_key;
    bool          grammar_key_invalid = false;
    try {
        grammar_key = keyFromGenerateConfig(config);
    } catch (const std::exception& e) {
        reportInvalidParams(error_reporter, std::string("invalid grammar response_format: ") + e.what());
        grammar_key_invalid = true;
    }

    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(generate_input, max_batch_size);
    if (think_processor != nullptr && grammar_key.empty()) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }

    if (!grammar_key.empty() && !grammar_key_invalid) {
        if (config.hasNumBeams() || config.num_return_sequences > 1) {
            reportInvalidParams(error_reporter,
                                "grammar-constrained decoding does not support beam search or "
                                "num_return_sequences > 1");
        } else {
            std::shared_ptr<XGrammarBackend> backend;
            {
                std::lock_guard<std::mutex> lock(g_grammar_backend_mutex);
                backend = g_grammar_backend;
            }
            if (auto grammar_processor =
                    createGrammarProcessor(backend, generate_input, grammar_key, eos_token_id, error_reporter)) {
                result.push_back(std::move(grammar_processor));
            }
        }
    }

    auto tree_processor = TreeLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (tree_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_processor));
    }

    auto rec_processor = RecommendationLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (rec_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(rec_processor));
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(generate_input, eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_processor));
    }

    return result;
}

}  // namespace rtp_llm
