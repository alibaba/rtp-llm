#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class GenerateConfig;
struct GrammarConfig;

class LogitsProcessorFactory {
public:
    using ErrorReporter = LogitsProcessorErrorReporter;

    // grammar_config.tokenizer_info_json must already be populated by the
    // caller (RtpLLMOp does the bootstrap at the pybind boundary). The factory
    // is pure-C++ and stays out of the Python ABI.
    static void init(const std::string&   ckpt_path,
                     const std::string&   tree_decode_config,
                     const GrammarConfig& grammar_config);

    // True iff `config` carries any structured-output request. Only typed
    // grammar fields (json_schema/regex/ebnf/structural_tag) are inspected;
    // OpenAI response_format envelopes are projected to typed fields by
    // GenerateConfig.validate (Python) before reaching the engine.
    static bool hasGrammarConstraint(const GenerateConfig& config);

    static std::vector<BaseLogitsProcessorPtr> createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        init_batch_size,
                                                                      int32_t                        max_batch_size,
                                                                      int64_t                        eos_token_id,
                                                                      const ErrorReporter& error_reporter = {});
};

}  // namespace rtp_llm
