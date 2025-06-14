#include "rtp_llm/cpp/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;

namespace rtp_llm {

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device) : BaseLogitsProcessor(device) {};

TreeLogitsProcessor::TreeLogitsProcessor(
    rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos)
    : BaseLogitsProcessor(device), 
      tree_infos_(tree_infos) {}
    
void TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) continue;
        
        const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
        if (candidate_token_ids.empty()) continue;

        auto logits = inputs.logits->index(i + start_idx);
        auto vocab_mask = generateVocabMask(logits->shape(), candidate_token_ids);
        maskLogits(logits, vocab_mask);
    }
}

void TreeLogitsProcessor::beamSearchLogitProcessorUpdate(const std::vector<int>& beam_idx_vec) {
    RTP_LLM_CHECK(size() == beam_idx_vec.size());
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto beam_idx: beam_idx_vec) {
        new_tree_infos.push_back(tree_infos_[beam_idx].copy());
    }
    tree_infos_ = new_tree_infos;
}

void TreeLogitsProcessor::updateLogitProcessorStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens->shape().size());
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);
    
    for (size_t i = 0; i < size(); i++) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens->shape()[1]);
        }
        
        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
            info.dfa_ptr->next(current_token_id);
        }
        
        info.current_output_length += num_new_tokens;
    }
}


TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(
    rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int32_t num)
{
    if (PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>(rtp_llm::DeviceFactory::getDefaultDevice());
    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(), 0, 
                                 generate_input->generate_config->num_beams > 1 || generate_input->generate_config->num_return_sequences > 1,
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
        std::vector<StreamTreeInfo> tree_infos = { tree_info };
        auto single_processor = std::make_shared<TreeLogitsProcessor>(device, tree_infos);

        processor_ptr->insert(single_processor, 1);
    }

    return processor_ptr;
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info: tree_infos_) {
        status_list.push_back(tree_info.dfa_ptr->status());
    }
    return status_list;
}


}