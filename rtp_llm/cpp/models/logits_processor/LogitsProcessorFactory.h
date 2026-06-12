#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class ModelConfig;
class XGrammarBackend;
struct GrammarConfig;

class LogitsProcessorFactory {
public:
    static void init(const ModelConfig&   model_config,
                     const GrammarConfig& grammar_config,
                     const std::string&   tree_decode_config);

    static ErrorResult<std::vector<BaseLogitsProcessorPtr>>
    createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                           int32_t                        init_batch_size,
                           int32_t                        max_batch_size,
                           int64_t                        eos_token_id);

private:
    static std::shared_ptr<XGrammarBackend>& grammarBackend();
};

}  // namespace rtp_llm
