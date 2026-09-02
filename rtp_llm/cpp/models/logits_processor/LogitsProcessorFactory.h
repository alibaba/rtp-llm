#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class GenerateConfig;
class GenerateInput;
class ModelConfig;
struct GrammarConfig;

class LogitsProcessorFactory {
public:
    using ErrorReporter = std::function<void(ErrorCode, const std::string&, bool)>;

    static void init(const ModelConfig&   model_config,
                     const std::string&   tree_decode_config,
                     const GrammarConfig& grammar_config);

    static bool hasGrammarConstraint(const GenerateConfig& config);

    static std::vector<BaseLogitsProcessorPtr> createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        init_batch_size,
                                                                      int32_t                        max_batch_size,
                                                                      int64_t                        eos_token_id,
                                                                      const std::string&             model_type,
                                                                      ErrorReporter error_reporter = nullptr);
};

}  // namespace rtp_llm
