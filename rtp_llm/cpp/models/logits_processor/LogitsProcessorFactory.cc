#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessorCSR.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"

namespace rtp_llm {

void LogitsProcessorFactory::init(const std::string& ckpt_path, const std::string& tree_decode_config) {
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(ckpt_path, tree_decode_config);
}

std::vector<BaseLogitsProcessorPtr>
LogitsProcessorFactory::createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                                               int32_t                        init_batch_size,
                                               int32_t                        max_batch_size,
                                               int64_t                        eos_token_id,
                                               int32_t                        vocab_size) {
    std::vector<BaseLogitsProcessorPtr> result;

    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(generate_input, max_batch_size);
    if (think_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }

    // 基于 CSR 前缀树的限制性解码：ele_rq_ids 非空时启用。
    // 否则回退到原有基于 DFA 状态机的 TreeLogitsProcessor。
    if (!generate_input->generate_config->ele_rq_ids.empty()) {
        auto csr_processor =
            TreeLogitsProcessorCSR::fromGenerateInput(generate_input, init_batch_size, vocab_size);
        if (csr_processor != nullptr) {
            result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(csr_processor));
        }
    } else {
        auto tree_processor = TreeLogitsProcessor::fromGenerateInput(generate_input, init_batch_size);
        if (tree_processor != nullptr) {
            result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_processor));
        }
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(generate_input, eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_processor));
    }

    return result;
}

}  // namespace rtp_llm
