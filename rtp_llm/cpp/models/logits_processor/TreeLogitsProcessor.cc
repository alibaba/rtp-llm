#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;

namespace rtp_llm {

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {};

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos):
    BaseLogitsProcessor(device), tree_infos_(tree_infos) {}

void TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    bool                             need_process        = false;
    bool                             need_weight_process = false;
    bool                             is_sparse_mask      = false;
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);
    std::vector<const TokenWeights*> batch_candidate_token_weights(batch_size, nullptr);
    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }
        if (info.dfa_ptr->hasWeightDict()) {
            batch_candidate_token_weights[i] = info.dfa_ptr->getCandidateTokenWeights();
            if (batch_candidate_token_weights[i] != nullptr && batch_candidate_token_weights[i]->token_ids.size() > 0) {
                need_weight_process = true;
            }
        } else {
            const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
            batch_candidate_token_ids[i]    = candidate_token_ids;
            if (candidate_token_ids.size() > 0) {
                need_process   = true;
                is_sparse_mask = info.dfa_ptr->isSparseMask();
            }
        }
    }
    // If no beams need processing, return early
    if (!need_process && !need_weight_process) {
        return;
    }

    auto   batch_logits = inputs.logits->slice(start_idx, batch_size);
    size_t vocab_size   = batch_logits->shape()[1];

    if (need_weight_process) {
        auto weight_logits_params =
            generateVocabWeight(batch_size, vocab_size, batch_candidate_token_weights, batch_logits);
        weightLogits(weight_logits_params);
        if (inputs.sampler_mask_params != nullptr) {
            inputs.sampler_mask_params->addWeightParam(weight_logits_params);
        }
    } else {
        if (is_sparse_mask) {
            auto sparse_mask_logits_params =
                generateSparseVocabMask(batch_size, vocab_size, batch_candidate_token_ids, batch_logits);
            sparseMaskLogits(sparse_mask_logits_params);
        } else {
            auto batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
            maskLogits(batch_logits, batch_vocab_mask);
        }
    }
}

void TreeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_tree_infos.push_back(tree_infos_[src_batch_idx].copy());
    }
    tree_infos_ = std::move(new_tree_infos);
}

void TreeLogitsProcessor::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens->shape().size());
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);

    for (size_t i = 0; i < size(); i++) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode)
            continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens->shape()[1]);
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto        current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
            std::string next_status      = info.dfa_ptr->next(current_token_id);
            if (next_status == "ERROR") {
                std::stringstream ss;
                ss << "Generated invalid status, status[" << info.dfa_ptr->status() << "], input_id["
                   << current_token_id << "]";
                ss << ",new_tokens:" << new_tokens->debugStringWithData<int32_t>()
                   << ", num_new_tokens:" << num_new_tokens;
                ss << "J:" << j << ",i:" << i << ", info.is_beam_search:" << info.is_beam_search
                   << ", info.current_output_length:" << info.current_output_length
                   << ", info.input_length:" << info.input_length;
                throw std::runtime_error(ss.str());
            }
        }

        info.current_output_length += num_new_tokens;
    }
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(rtp_llm::DeviceBase*           device,
                                                              std::shared_ptr<GenerateInput> generate_input,
                                                              int32_t                        num) {
    if (!PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>(rtp_llm::DeviceFactory::getDefaultDevice());
    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo              tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(),
                                 0,
                                 generate_input->generate_config->hasNumBeams()
                                     || generate_input->generate_config->num_return_sequences > 1,
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
        std::vector<StreamTreeInfo> tree_infos       = {tree_info};
        auto                        single_processor = std::make_shared<TreeLogitsProcessor>(device, tree_infos);

        processor_ptr->insert(single_processor, 1);
    }

    return processor_ptr;
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info : tree_infos_) {
        status_list.push_back(tree_info.dfa_ptr->status());
    }
    return status_list;
}

}  // namespace rtp_llm
