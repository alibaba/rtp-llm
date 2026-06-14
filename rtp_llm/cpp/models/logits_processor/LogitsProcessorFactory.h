#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorErrorReporter.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class GenerateConfig;
struct StructuredOutputConfig;
using GrammarConfig = StructuredOutputConfig;

class LogitsProcessorFactory {
public:
    using ErrorReporter = LogitsProcessorErrorReporter;

    static void init(const std::string&   ckpt_path,
                     const std::string&   tree_decode_config,
                     const GrammarConfig& grammar_config);

    // True iff `config` carries any structured-output request (direct field or
    // OpenAI response_format envelope). Malformed envelopes report true so the
    // createLogitsProcessors path can surface the parse error.
    static bool hasGrammarConstraint(const GenerateConfig& config);

    static std::vector<BaseLogitsProcessorPtr> createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        init_batch_size,
                                                                      int32_t                        max_batch_size,
                                                                      int64_t                        eos_token_id,
                                                                      const ErrorReporter& error_reporter = {});
};

}  // namespace rtp_llm
