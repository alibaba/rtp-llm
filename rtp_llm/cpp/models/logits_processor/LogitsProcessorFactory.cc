#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"

namespace rtp_llm {

namespace {
// Set once during engine init (single thread, before any request is enqueued)
// and only read afterwards, so no additional synchronization is required.
LogitsProcessorFactory::ExtraProcessorFactory& extraFactoryRef() {
    static LogitsProcessorFactory::ExtraProcessorFactory factory;
    return factory;
}
}  // namespace

void LogitsProcessorFactory::init(const std::string& ckpt_path, const std::string& tree_decode_config) {
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(ckpt_path, tree_decode_config);
}

void LogitsProcessorFactory::registerExtraFactory(ExtraProcessorFactory factory) {
    extraFactoryRef() = std::move(factory);
}

std::vector<BaseLogitsProcessorPtr>
LogitsProcessorFactory::createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                               int32_t                        init_batch_size,
                                               int32_t                        max_batch_size,
                                               int64_t                        eos_token_id) {
    std::vector<BaseLogitsProcessorPtr> result;

    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(generate_input, max_batch_size);
    if (think_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }

    auto tree_processor = TreeLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (tree_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_processor));
    }

    // 生成式推荐：combo 粒度去重 + 曝光过滤
    auto rec_processor = RecommendationLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
    if (rec_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(rec_processor));
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(generate_input, eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_processor));
    }

    // Module-injected processors (e.g. structured output / grammar). Created here
    // so the stream stays free of any upward dependency; the processor may be
    // not-yet-ready and resolve later through prepare().
    if (auto& extra_factory = extraFactoryRef()) {
        if (auto extra = extra_factory(generate_input)) {
            result.push_back(std::move(extra));
        }
    }

    return result;
}

}  // namespace rtp_llm