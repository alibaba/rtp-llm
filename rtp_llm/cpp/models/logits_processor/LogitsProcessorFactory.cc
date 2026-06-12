#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

ErrorResult<GrammarKeyCpp> keyFromGenerateConfig(const GenerateConfig& config) {
    std::vector<std::string> constraint_names;
    if (config.json_schema.has_value()) {
        constraint_names.emplace_back("json_schema");
    }
    if (config.regex.has_value()) {
        constraint_names.emplace_back("regex");
    }
    if (config.ebnf.has_value()) {
        constraint_names.emplace_back("ebnf");
    }
    if (config.structural_tag.has_value()) {
        constraint_names.emplace_back("structural_tag");
    }
    if (constraint_names.size() > 1) {
        std::string received = constraint_names.front();
        for (size_t i = 1; i < constraint_names.size(); ++i) {
            received += ", " + constraint_names[i];
        }
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "only one grammar constraint may be set per request; received: " + received);
    }
    if (config.json_schema.has_value()) {
        return GrammarKeyCpp{"json", config.json_schema.value()};
    }
    if (config.regex.has_value()) {
        return GrammarKeyCpp{"regex", config.regex.value()};
    }
    if (config.ebnf.has_value()) {
        return GrammarKeyCpp{"ebnf", config.ebnf.value()};
    }
    if (config.structural_tag.has_value()) {
        return GrammarKeyCpp{"structural_tag", config.structural_tag.value()};
    }
    // response_format envelope is projected to typed fields above in Python ResponseFormatBuilder.
    // The C++ engine only consumes typed fields.
    return GrammarKeyCpp{};
}

}  // namespace

std::shared_ptr<XGrammarBackend>& LogitsProcessorFactory::grammarBackend() {
    // Process-wide by design: each rank runs in its own process, and RTP-LLM
    // constructs exactly one engine/executor per process. The backend is thus
    // initialized once from that engine's tokenizer and shared by its streams.
    // Do not use this factory for multiple engines with different tokenizers in
    // one process without first moving this state to the engine/executor.
    static std::shared_ptr<XGrammarBackend> backend;
    return backend;
}

void LogitsProcessorFactory::init(const ModelConfig&   model_config,
                                  const GrammarConfig& grammar_config,
                                  const std::string&   tree_decode_config) {
    grammarBackend() = XGrammarBackend::create(model_config.tokenizer_info_json, grammar_config);
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(model_config.ckpt_path, tree_decode_config);
}

ErrorResult<std::vector<BaseLogitsProcessorPtr>>
LogitsProcessorFactory::createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                               int32_t                        init_batch_size,
                                               int32_t                        max_batch_size,
                                               int64_t                        eos_token_id) {
    std::vector<BaseLogitsProcessorPtr> result;

    auto& config = *generate_input->generate_config;

    auto grammar_key_result = keyFromGenerateConfig(config);
    if (!grammar_key_result.ok()) {
        return grammar_key_result.status();
    }
    GrammarKeyCpp grammar_key = std::move(grammar_key_result.value());

    // Thinking constraints reach the execution layer as normalized grammar.
    // Thinking-only configs are not supported here; do not add a legacy processor fallback.
    if (!grammar_key.empty()) {
        if (config.hasNumBeams() || config.num_return_sequences > 1) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS,
                             "grammar-constrained decoding does not support beam search or "
                             "num_return_sequences > 1");
        }
        auto& backend = grammarBackend();
        if (!backend) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS,
                             "structured output requested but constraint backend is disabled "
                             "(check engine startup logs: tokenizer info empty or backend init failed).");
        }

        const bool terminate_without_stop_token =
            config.grammar_terminate_without_stop_token || grammar_key.key_type == "json";
        RTP_LLM_LOG_DEBUG("grammar matcher install: type=%s, len=%zu, in_think_mode=%d, "
                          "terminate_without_stop_token=%d",
                          grammar_key.key_type.c_str(),
                          grammar_key.key_string.size(),
                          static_cast<int>(config.in_think_mode),
                          static_cast<int>(terminate_without_stop_token));

        auto matcher_or = backend->createMatcherFromKey(grammar_key, terminate_without_stop_token);
        if (!matcher_or.ok()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, std::string(matcher_or.status().message()));
        }
        auto grammar_processor =
            std::make_shared<GrammarLogitsProcessor>(std::move(matcher_or.value()), eos_token_id);
        result.push_back(std::move(grammar_processor));
    }

    auto tree_processor = TreeLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (tree_processor != nullptr) {
        result.push_back(std::move(tree_processor));
    }

    auto rec_processor = RecommendationLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (rec_processor != nullptr) {
        result.push_back(std::move(rec_processor));
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(generate_input, eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::move(multi_seq_processor));
    }

    return std::move(result);
}

}  // namespace rtp_llm
