#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rtp_llm {

enum class ThinkProcessState {
    NO_THINK,
    IN_THINK,
    CLOSING_THINK,
    AFTER_THINK,
};

inline int64_t thinkBodyTokenBudget(int max_thinking_tokens, size_t begin_tag_tokens, size_t end_tag_tokens) {
    if (max_thinking_tokens <= 0) {
        return 0;
    }
    return std::max<int64_t>(0,
                             static_cast<int64_t>(max_thinking_tokens) - static_cast<int64_t>(begin_tag_tokens)
                                 - static_cast<int64_t>(end_tag_tokens));
}

inline int64_t thinkGeneratedTokenBudget(int max_thinking_tokens, size_t begin_tag_tokens, size_t end_tag_tokens) {
    if (max_thinking_tokens <= 0) {
        return 0;
    }
    const int64_t generated_budget = static_cast<int64_t>(max_thinking_tokens) - static_cast<int64_t>(begin_tag_tokens);
    return std::max<int64_t>(static_cast<int64_t>(end_tag_tokens), generated_budget);
}

struct StreamThinkInfo {
    bool                                           in_think_mode;
    int                                            max_thinking_tokens;
    std::vector<int>                               begin_think_token_ids;
    std::vector<int>                               end_think_token_ids;
    int32_t                                        input_length;
    int32_t                                        current_output_length;
    int32_t                                        think_output_length = -1;
    bool                                           is_beam_search;
    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
    std::vector<int>                               pending_forced_think_end_token_ids;
    ThinkProcessState                              process_state = ThinkProcessState::NO_THINK;

    StreamThinkInfo() = default;

    StreamThinkInfo(bool                                           think_mode,
                    int                                            max_thinking_tokens,
                    std::vector<int>                               begin_think_token_ids,
                    std::vector<int>                               end_think_token_ids,
                    int32_t                                        input_length,
                    int32_t                                        output_length,
                    bool                                           is_beam_search,
                    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr):
        in_think_mode(think_mode),
        max_thinking_tokens(max_thinking_tokens),
        begin_think_token_ids(begin_think_token_ids),
        end_think_token_ids(end_think_token_ids),
        input_length(input_length),
        current_output_length(output_length),
        is_beam_search(is_beam_search),
        dfa_ptr(dfa_ptr) {
        if (think_mode && max_thinking_tokens > 0 && dfa_ptr) {
            process_state = ThinkProcessState::IN_THINK;
        }
    }

    int64_t bodyTokenBudget() const {
        return thinkBodyTokenBudget(max_thinking_tokens, begin_think_token_ids.size(), end_think_token_ids.size());
    }

    void markAfterThink() {
        process_state = ThinkProcessState::AFTER_THINK;
        if (think_output_length < 0) {
            think_output_length = current_output_length;
        }
    }

    int64_t finishedThinkOutputLen() const {
        if (process_state != ThinkProcessState::AFTER_THINK) {
            return -1;
        }
        return think_output_length >= 0 ? think_output_length : current_output_length;
    }

    StreamThinkInfo copy() const {
        StreamThinkInfo think_info;
        think_info.in_think_mode                      = in_think_mode;
        think_info.max_thinking_tokens                = max_thinking_tokens;
        think_info.begin_think_token_ids              = begin_think_token_ids;
        think_info.end_think_token_ids                = end_think_token_ids;
        think_info.input_length                       = input_length;
        think_info.current_output_length              = current_output_length;
        think_info.think_output_length                = think_output_length;
        think_info.is_beam_search                     = is_beam_search;
        think_info.pending_forced_think_end_token_ids = pending_forced_think_end_token_ids;
        think_info.process_state                      = process_state;
        if (dfa_ptr) {
            think_info.dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(*dfa_ptr);
        }
        return think_info;
    }
};

struct ThinkModeSpecSnapshot {
    bool            eligible = false;
    StreamThinkInfo info;
    uint64_t        version = 0;
};

class ThinkModeLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    ThinkModeLogitsProcessor() = default;
    ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos);
    virtual ~ThinkModeLogitsProcessor() {}

public:
    static std::shared_ptr<ThinkModeLogitsProcessor> fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                       int32_t                        num);

public:
    void    process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void    updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void    updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    bool    isSpecVerifyEligible() const override;
    int     tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool    isStateful() const override;
    int64_t acceptedTokenLen() const override;
    int64_t finishedThinkOutputLen() const override;

private:
    bool forceThinkEndToken(const torch::Tensor& new_tokens_logits, StreamThinkInfo& info, size_t vocab_size);

public:
    std::vector<size_t> thinkEndTokensStatus();
    size_t              size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return think_infos_.size();
    }
    void insert(std::shared_ptr<ThinkModeLogitsProcessor> others, size_t num) {
        if (others != nullptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            think_infos_.insert(think_infos_.end(), others->think_infos_.begin(), others->think_infos_.end());
            publishSpecSnapshotLocked();
        }
    }

private:
    void publishSpecSnapshotLocked();

private:
    std::vector<StreamThinkInfo>                 think_infos_;
    mutable std::mutex                           mutex_;
    std::shared_ptr<const ThinkModeSpecSnapshot> spec_snapshot_;
    uint64_t                                     spec_snapshot_version_ = 0;
};
typedef std::shared_ptr<ThinkModeLogitsProcessor> ThinkModeLogitsProcessorPtr;

}  // namespace rtp_llm
