#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;

namespace rtp_llm {

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {};

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(rtp_llm::DeviceBase*         device,
                                                   std::vector<StreamThinkInfo> think_infos):
    BaseLogitsProcessor(device), think_infos_(think_infos) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode)
            continue;

        int* input_lengths    = inputs.input_lengths->data<int32_t>();
        int* sequence_lengths = inputs.sequence_lengths->data<int32_t>();
        int  num_new_tokens   = 1;
        bool enforce          = (sequence_lengths[i + start_idx] + num_new_tokens
                        >= info.max_thinking_tokens + input_lengths[i + start_idx]);
        setVocabMask(info.dfa_ptr,
                     inputs.logits->index(i + start_idx),
                     num_new_tokens,
                     info.end_think_token_ids,
                     inputs.vocab_size,
                     enforce);
    }
}

void ThinkModeLogitsProcessor::setVocabMask(std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr,
                                            rtp_llm::BufferPtr                             new_tokens_logits,
                                            int                                            num_new_tokens,
                                            std::vector<int>                               template_token_ids,
                                            size_t                                         vocab_size,
                                            bool                                           enforce) {
    if (!dfa_ptr->isFinished() && enforce) {
        RTP_LLM_LOG_INFO("sampler enforce transfer status");
        memFill(new_tokens_logits, vocab_size, (size_t)template_token_ids[dfa_ptr->status()]);
    }
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamThinkInfo> new_think_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = new_think_infos;
}

void ThinkModeLogitsProcessor::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens->shape().size());
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);

    for (size_t i = 0; i < size(); i++) {
        auto& info = think_infos_[i];
        if (!info.in_think_mode)
            continue;

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

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(rtp_llm::DeviceBase*           device,
                                                                        std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    if (!generate_input->generate_config->in_think_mode || generate_input->generate_config->max_thinking_tokens == 0) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<ThinkModeLogitsProcessor>(rtp_llm::DeviceFactory::getDefaultDevice());
    for (size_t i = 0; i < num; i++) {
        StreamThinkInfo think_info(
            generate_input->generate_config->in_think_mode,
            generate_input->generate_config->max_thinking_tokens,
            generate_input->generate_config->end_think_token_ids,
            generate_input->inputLength(),
            0,
            generate_input->generate_config->hasNumBeams() || generate_input->generate_config->num_return_sequences > 1,
            std::make_shared<StringContainDFA<size_t, int>>(generate_input->generate_config->end_think_token_ids));
        std::vector<StreamThinkInfo> think_infos = {think_info};
        auto                         ptr         = std::make_shared<ThinkModeLogitsProcessor>(device, think_infos);

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