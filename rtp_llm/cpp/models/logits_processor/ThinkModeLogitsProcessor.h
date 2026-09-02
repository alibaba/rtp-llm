#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace rtp_llm {

enum class ThinkProcessState {
    NO_THINK,
    IN_THINK,
    CLOSING_THINK,
    AFTER_THINK,
};

std::vector<int> makeMaskedStopTokenIds(const std::vector<std::vector<int>>& stop_words_list,
                                        int64_t                              eos_token_id,
                                        const std::vector<int>&              end_think_token_ids);

struct StreamThinkInfo {
    int              max_thinking_tokens;
    std::vector<int> begin_think_token_ids;
    std::vector<int> end_think_token_ids;
    std::vector<int> masked_stop_token_ids;
    int32_t          input_length;
    int32_t          current_output_length;
    bool             is_beam_search;
    int32_t          last_token_id = -1;
    std::string      model_type;

    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
    ThinkProcessState                              process_state = ThinkProcessState::NO_THINK;
    std::vector<int>                               pending_forced_think_end_token_ids;

    StreamThinkInfo() = default;

    StreamThinkInfo(bool                                           think_mode,
                    int                                            max_thinking_tokens,
                    std::vector<int>                               begin_think_token_ids,
                    std::vector<int>                               end_think_token_ids,
                    int32_t                                        input_length,
                    int32_t                                        output_length,
                    bool                                           is_beam_search,
                    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr,
                    std::vector<int>                               masked_stop_token_ids = {},
                    std::string                                    model_type = {}):
        max_thinking_tokens(max_thinking_tokens),
        begin_think_token_ids(std::move(begin_think_token_ids)),
        end_think_token_ids(std::move(end_think_token_ids)),
        masked_stop_token_ids(std::move(masked_stop_token_ids)),
        input_length(input_length),
        current_output_length(output_length),
        is_beam_search(is_beam_search),
        model_type(std::move(model_type)),
        dfa_ptr(std::move(dfa_ptr)) {
        if (think_mode && max_thinking_tokens > 0 && this->dfa_ptr) {
            process_state = ThinkProcessState::IN_THINK;
        }
    }

    StreamThinkInfo copy() const {
        StreamThinkInfo think_info;
        think_info.max_thinking_tokens                = max_thinking_tokens;
        think_info.begin_think_token_ids              = begin_think_token_ids;
        think_info.end_think_token_ids                = end_think_token_ids;
        think_info.masked_stop_token_ids              = masked_stop_token_ids;
        think_info.input_length                       = input_length;
        think_info.current_output_length              = current_output_length;
        think_info.is_beam_search                     = is_beam_search;
        think_info.pending_forced_think_end_token_ids = pending_forced_think_end_token_ids;
        think_info.last_token_id                      = last_token_id;
        think_info.model_type                         = model_type;
        think_info.process_state                      = process_state;
        if (dfa_ptr) {
            think_info.dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(*dfa_ptr);
        }
        return think_info;
    }

    bool    isActiveThinkState() const;
    bool    thinkEndCloseInProgress() const;
    int32_t nextThinkEndToken() const;
    int32_t thinkBoundaryToMask(const std::vector<int>& boundary) const;
    bool    updateState(int32_t token_id);
    void    precommitThinkEndToken(int32_t token_id);
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
    static std::shared_ptr<ThinkModeLogitsProcessor>
    fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                      int32_t                        num,
                      int64_t                        eos_token_id = -1,
                      const std::string&             model_type = {});

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool isStateful() const override;
    int64_t acceptedTokenLen() const override;

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
