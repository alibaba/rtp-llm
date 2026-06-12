#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class LogitsProcessorFactory {
public:
    static void init(const std::string& ckpt_path, const std::string& tree_decode_config);

    static std::vector<BaseLogitsProcessorPtr> createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        init_batch_size,
                                                                      int32_t                        max_batch_size,
                                                                      int64_t                        eos_token_id);

    // Injection seam for processors that live in modules this lib cannot depend
    // on directly (a direct reference would form a build cycle:
    // models:logits_processor -> xgrammar -> engine_base/stream ->
    // models:logits_processor). The owning module registers its factory once at
    // engine init (see NormalEngine); createLogitsProcessors appends the
    // produced processor (when non-null) so the grammar processor is created by
    // the same factory path as the built-in ones. The processor may start in a
    // not-yet-ready state and resolve via BaseLogitsProcessor::prepare().
    using ExtraProcessorFactory = std::function<BaseLogitsProcessorPtr(const std::shared_ptr<GenerateInput>&)>;
    static void registerExtraFactory(ExtraProcessorFactory factory);
};

}  // namespace rtp_llm
