#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorException.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

std::mutex                       g_grammar_backend_mutex;
std::shared_ptr<XGrammarBackend> g_grammar_backend;

GrammarKeyCpp keyFromGenerateConfig(const GenerateConfig& config) {
    // Fixed priority json_schema > regex > ebnf > structural_tag silently drops the
    // others when a caller sets multiple. Warn so we can audit the call site.
    const bool multi_grammar = (config.json_schema.has_value() + config.regex.has_value() + config.ebnf.has_value()
                                + config.structural_tag.has_value())
                               > 1;
    if (multi_grammar) {
        RTP_LLM_LOG_WARNING("GenerateConfig sets multiple grammar fields simultaneously; "
                            "applying priority json_schema>regex>ebnf>structural_tag");
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
    // response_format envelope is projected to typed fields above in
    // GenerateConfig.validate (Python). The C++ engine only consumes typed fields.
    return {};
}

// Compile + matcher creation, given an already-resolved GrammarKeyCpp. On any
// failure throws LogitsProcessorException; on success returns the matcher.
std::shared_ptr<RtpGrammarMatcher> compileMatcherFromKey(XGrammarBackend&                backend,
                                                         const GrammarKeyCpp&            key,
                                                         bool                            require_reasoning,
                                                         std::optional<std::vector<int>> think_end_token_ids) {
    // Lightweight size caps (xgrammar already validates schema/regex content).
    constexpr size_t kMaxJsonSchemaSize = 1024 * 1024;  // 1 MiB
    constexpr size_t kMaxRegexEbnfSize  = 64 * 1024;    // 64 KiB
    const size_t     limit              = (key.key_type == "json") ? kMaxJsonSchemaSize : kMaxRegexEbnfSize;
    if (key.key_string.size() > limit) {
        const std::string detail =
            key.key_type + " grammar exceeds maximum size limit (" + std::to_string(limit) + " bytes)";
        // Don't push oversize keys into invalid_cache_ — backend.getOrCompile
        // rejects them too, and caching the rejection per-key would let N distinct
        // huge blobs inflate memory before LRU evicts them. The user already gets
        // INVALID_PARAMS; on retry this size check rejects in O(1) without cache.
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                       "Failed to compile " + key.key_type + " grammar: " + detail);
    }

    CompileResult result;
    try {
        result = backend.getOrCompile(key);
    } catch (const std::exception& e) {
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                       std::string("grammar compile error: ") + e.what());
    }
    if (!result.compiled) {
        const std::string err = result.error_message.empty() ? "unknown compile error" : result.error_message;
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                       "Failed to compile " + key.key_type + " grammar: " + err);
    }

    const bool                         terminate_without_stop_token = key.key_type == "json";
    std::shared_ptr<RtpGrammarMatcher> matcher                      = backend.createMatcher(
        result.compiled, require_reasoning, std::move(think_end_token_ids), terminate_without_stop_token);
    if (!matcher) {
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS, "grammar matcher install failed");
    }
    return matcher;
}

BaseLogitsProcessorPtr createGrammarProcessor(const std::shared_ptr<XGrammarBackend>& backend,
                                              const std::shared_ptr<GenerateInput>&   input,
                                              const GrammarKeyCpp&                    key,
                                              int64_t                                 eos_token_id) {
    if (!input || !input->generate_config || key.empty()) {
        return nullptr;
    }
    auto& config = *input->generate_config;
    if (!backend) {
        throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                       "structured output requested but constraint backend is disabled "
                                       "(check engine startup logs: tokenizer info empty or backend init failed).");
    }

    const bool                      require_reasoning = config.in_think_mode;
    std::optional<std::vector<int>> think_end_token_ids;
    if (require_reasoning) {
        if (config.end_think_token_ids.empty()) {
            throw LogitsProcessorException(
                ErrorCode::INVALID_PARAMS,
                "structured output with in_think_mode requires non-empty end_think_token_ids");
        }
        think_end_token_ids = config.end_think_token_ids;
    }

    auto matcher = compileMatcherFromKey(*backend, key, require_reasoning, std::move(think_end_token_ids));

    if (require_reasoning) {
        return std::make_shared<ReasoningGrammarLogitsProcessor>(std::move(matcher),
                                                                 eos_token_id,
                                                                 config.max_thinking_tokens,
                                                                 config.begin_think_token_ids,
                                                                 config.end_think_token_ids,
                                                                 input->inputLength());
    }
    return std::make_shared<GrammarLogitsProcessor>(std::move(matcher), eos_token_id);
}

}  // namespace

bool LogitsProcessorFactory::hasGrammarConstraint(const GenerateConfig& config) {
    return !keyFromGenerateConfig(config).empty();
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
                                               int64_t                        eos_token_id) {
    std::vector<BaseLogitsProcessorPtr> result;

    auto& config = *generate_input->generate_config;

    GrammarKeyCpp grammar_key = keyFromGenerateConfig(config);

    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(generate_input, max_batch_size);
    if (think_processor != nullptr && grammar_key.empty()) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }

    if (!grammar_key.empty()) {
        if (config.hasNumBeams() || config.num_return_sequences > 1) {
            throw LogitsProcessorException(ErrorCode::INVALID_PARAMS,
                                           "grammar-constrained decoding does not support beam search or "
                                           "num_return_sequences > 1");
        }
        std::shared_ptr<XGrammarBackend> backend;
        {
            std::lock_guard<std::mutex> lock(g_grammar_backend_mutex);
            backend = g_grammar_backend;
        }
        if (auto grammar_processor = createGrammarProcessor(backend, generate_input, grammar_key, eos_token_id)) {
            result.push_back(std::move(grammar_processor));
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
