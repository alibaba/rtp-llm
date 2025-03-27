#pragma once

#include "maga_transformer/cpp/models/BaseLogitsProcessor.h"
#include "maga_transformer/cpp/utils/DFAUtil.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class StreamThinkInfo;

class ThinkModeLogitsProcessor: public BaseLogitsProcessor {
public:
    ThinkModeLogitsProcessor(ft::DeviceBase* device, std::vector<StreamThinkInfo> think_infos);
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
    std::vector<StreamThinkInfo> think_infos_;
};
typedef std::shared_ptr<ThinkModeLogitsProcessor> ThinkModeLogitsProcessorPtr;

} // namespace rtp_llm