#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"

namespace rtp_llm {

struct StreamThinkInfo {
    bool                                           in_think_mode;
    int                                            max_thinking_tokens;
    std::vector<int>                               end_think_token_ids;
    int32_t                                        input_length;
    int32_t                                        current_output_length;
    bool                                           is_beam_search;
    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;

    StreamThinkInfo() = default;

    StreamThinkInfo(bool                                           think_mode,
                    int                                            max_thinking_tokens,
                    std::vector<int>                               end_think_token_ids,
                    int32_t                                        input_length,
                    int32_t                                        output_length,
                    bool                                           is_beam_search,
                    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr):
        in_think_mode(think_mode),
        max_thinking_tokens(max_thinking_tokens),
        end_think_token_ids(end_think_token_ids),
        input_length(input_length),
        current_output_length(output_length),
        is_beam_search(is_beam_search),
        dfa_ptr(dfa_ptr) {}

    StreamThinkInfo copy() {
        StreamThinkInfo think_info;
        think_info.in_think_mode         = in_think_mode;
        think_info.max_thinking_tokens   = max_thinking_tokens;
        think_info.end_think_token_ids   = end_think_token_ids;
        think_info.input_length          = input_length;
        think_info.current_output_length = current_output_length;
        think_info.is_beam_search        = is_beam_search;
        if (dfa_ptr) {
            think_info.dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(*dfa_ptr);
        }
        return think_info;
    }
};

class ThinkModeLogitsProcessor: public BaseLogitsProcessor {
public:
    ThinkModeLogitsProcessor(rtp_llm::DeviceBase* device);
    ThinkModeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamThinkInfo> think_infos);
    virtual ~ThinkModeLogitsProcessor() {}

public:
    static std::shared_ptr<ThinkModeLogitsProcessor>
    fromGenerateInput(rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int32_t num);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) override;

private:
    void setVocabMask(std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr,
                      rtp_llm::BufferPtr                             new_tokens_logits,
                      int                                            num_new_tokens,
                      std::vector<int>                               template_token_ids,
                      size_t                                         vocab_size,
                      bool                                           enforce);

public:
    std::vector<size_t> thinkEndTokensStatus();
    size_t              size() {
        return think_infos_.size();
    }
    void insert(std::shared_ptr<ThinkModeLogitsProcessor> others, size_t num) {
        if (others != nullptr) {
            think_infos_.insert(think_infos_.end(), others->think_infos_.begin(), others->think_infos_.end());
        }
    }

private:
    std::vector<StreamThinkInfo> think_infos_;
};
typedef std::shared_ptr<ThinkModeLogitsProcessor> ThinkModeLogitsProcessorPtr;

}  // namespace rtp_llm