#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

#include <algorithm>

using namespace std;

namespace rtp_llm {

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(think_infos) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode || info.state == ThinkModeState::ANSWERING) {
            continue;
        }

        bool enforce = info.state == ThinkModeState::FORCING_CLOSE;
        if (!enforce && info.max_thinking_tokens > 0 && info.current_output_length >= info.max_thinking_tokens) {
            info.state = ThinkModeState::FORCING_CLOSE;
            enforce    = true;
            RTP_LLM_LOG_INFO("think mode budget reached, force close thinking");
        }
        setVocabMask(info.dfa_ptr,
                     inputs.logits[i + start_idx],
                     1,
                     info.end_think_token_ids,
                     inputs.vocab_size,
                     enforce);
    }
}

void ThinkModeLogitsProcessor::setVocabMask(std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr,
                                            const torch::Tensor&                           new_tokens_logits,
                                            int                                            num_new_tokens,
                                            std::vector<int>                               template_token_ids,
                                            size_t                                         vocab_size,
                                            bool                                           enforce) {
    if (!dfa_ptr->isFinished() && enforce) {
        RTP_LLM_LOG_INFO("sampler enforce transfer status");
        memFill(new_tokens_logits, vocab_size, (size_t)template_token_ids[dfa_ptr->status()]);
    }
}

bool ThinkModeLogitsProcessor::isAbortThinkToken(const StreamThinkInfo& info, int token_id) const {
    return std::find(info.abort_think_token_ids.begin(), info.abort_think_token_ids.end(), token_id)
           != info.abort_think_token_ids.end();
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamThinkInfo> new_think_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = new_think_infos;
}

bool ThinkModeLogitsProcessor::postProcessSampledTokens(torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    bool modified = false;
    for (size_t i = 0; i < size(); i++) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode || info.state == ThinkModeState::ANSWERING || info.end_think_token_ids.empty()) {
            continue;
        }
        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;
        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        size_t local_close_status = info.dfa_ptr->status();
        bool   forcing            = info.state == ThinkModeState::FORCING_CLOSE;
        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto token_ptr = new_tokens.data_ptr<int>() + i * new_tokens.size(1) + j + offset;
            if (forcing) {
                if (local_close_status < info.end_think_token_ids.size()) {
                    *token_ptr = info.end_think_token_ids[local_close_status++];
                    modified   = true;
                }
                continue;
            }
            if (isAbortThinkToken(info, *token_ptr)) {
                *token_ptr = info.end_think_token_ids[local_close_status++];
                info.state = ThinkModeState::FORCING_CLOSE;
                forcing    = true;
                modified   = true;
                RTP_LLM_LOG_INFO("think mode abort token sampled, rewrite to close thinking");
            }
        }
    }
    return modified;
}

void ThinkModeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < size(); i++) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode || info.state == ThinkModeState::ANSWERING) {
            continue;
        }

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            info.dfa_ptr->next(current_token_id);
            if (info.dfa_ptr->isFinished()) {
                info.state = ThinkModeState::ANSWERING;
                break;
            }
            if (info.dfa_ptr->status() > 0) {
                info.state = ThinkModeState::FORCING_CLOSE;
            }
        }

        info.current_output_length += num_new_tokens;
    }
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    if (!generate_input->generate_config->in_think_mode || generate_input->generate_config->max_thinking_tokens == 0
        || generate_input->generate_config->end_think_token_ids.empty()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<ThinkModeLogitsProcessor>();
    for (size_t i = 0; i < num; i++) {
        StreamThinkInfo think_info(
            generate_input->generate_config->in_think_mode,
            generate_input->generate_config->max_thinking_tokens,
            generate_input->generate_config->end_think_token_ids,
            generate_input->inputLength(),
            0,
            generate_input->generate_config->hasNumBeams() || generate_input->generate_config->num_return_sequences > 1,
            std::make_shared<StringContainDFA<size_t, int>>(generate_input->generate_config->end_think_token_ids),
            generate_input->generate_config->abort_think_token_ids);
        std::vector<StreamThinkInfo> think_infos = {think_info};
        auto                         ptr         = std::make_shared<ThinkModeLogitsProcessor>(think_infos);

        processor_ptr->insert(ptr, 1);
    }
    return processor_ptr;
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::vector<size_t> status;
    for (auto think_info : think_infos_) {
        auto dfa = think_info.dfa_ptr;
        status.push_back(dfa->status());
    }
    return status;
}

}  // namespace rtp_llm
