#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"

namespace rtp_llm {

void LogitsProcessorFactory::init(const std::string& ckpt_path, const std::string& tree_decode_config) {
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(ckpt_path, tree_decode_config);
}

std::vector<BaseLogitsProcessorPtr>
LogitsProcessorFactory::createLogitsProcessors(rtp_llm::DeviceBase*           device,
                                               std::shared_ptr<GenerateInput> generate_input,
                                               int32_t                        init_batch_size,
                                               int32_t                        max_batch_size,
                                               int64_t                        eos_token_id) {
    std::vector<BaseLogitsProcessorPtr> result;

    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(device, generate_input, max_batch_size);
    if (think_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }

    auto tree_processor = TreeLogitsProcessor::fromGenerateInput(device, generate_input, init_batch_size);
    if (tree_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_processor));
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(device, generate_input, eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_processor));
    }

    return result;
}

}  // namespace rtp_llm