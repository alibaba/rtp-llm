#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class XGrammarBackend;

class LogitsProcessorFactory {
public:
    // Process-wide one-time init: loads tree-decode prefix dictionary.
    static void init(const std::string& ckpt_path, const std::string& tree_decode_config);

    // Build-time errors come back as non-ok ErrorResult; caller surfaces on the stream.
    // `grammar_backend` is the engine-scoped backend (from ResourceContext); may be
    // null when grammar is disabled.
    static ErrorResult<std::vector<BaseLogitsProcessorPtr>>
    createLogitsProcessors(const std::shared_ptr<XGrammarBackend>& grammar_backend,
                           std::shared_ptr<GenerateInput>          generate_input,
                           int32_t                                 init_batch_size,
                           int32_t                                 max_batch_size,
                           int64_t                                 eos_token_id);
};

}  // namespace rtp_llm
