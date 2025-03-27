#pragma once

#include "maga_transformer/cpp/models/BaseLogitsProcessor.h"
#include "maga_transformer/cpp/utils/DFAUtil.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class ThinkModeLogitsProcessor: public BaseLogitsProcessor {
public:
    ThinkModeLogitsProcessor(ft::DeviceBase* device, std::deque<bool> think_modes, std::vector<int> max_thinking_tokens, 
        std::vector<std::vector<int>> end_think_token_ids, std::vector<std::shared_ptr<StringContainDFA<size_t, int>>> think_status_dfa_ptrs);
    virtual ~ThinkModeLogitsProcessor() {}

public:
    void process(const SamplerInputs& inputs);
    void updateStatus(const SamplerInputs& inputs);

private:
    void setVocabMask(std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr, 
        ft::BufferPtr new_tokens_logits, int num_new_tokens, 
        std::vector<int> template_token_ids, size_t vocab_size, bool enforce);

public:
    std::vector<size_t> thinkEndTokensStatus();

private:
    std::deque<bool> think_modes_;
    std::vector<int> max_thinking_tokens_;
    std::vector<std::vector<int>> end_think_token_ids_;
    std::vector<std::shared_ptr<StringContainDFA<size_t, int>>> think_status_dfa_ptrs_;
};
typedef std::shared_ptr<ThinkModeLogitsProcessor> ThinkModeLogitsProcessorPtr;

} // namespace rtp_llm