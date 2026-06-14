#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace rtp_llm {

enum class ThinkProcessState {
    NO_THINK,
    IN_THINK,
    CLOSING_THINK,
    AFTER_THINK,
};

struct StreamThinkInfo {
    bool                                           in_think_mode;
    int                                            max_thinking_tokens;
    std::vector<int>                               begin_think_token_ids;
    std::vector<int>                               end_think_token_ids;
    int32_t                                        input_length;
    int32_t                                        current_output_length;
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

    StreamThinkInfo copy() const {
        StreamThinkInfo think_info;
        think_info.in_think_mode                      = in_think_mode;
        think_info.max_thinking_tokens                = max_thinking_tokens;
        think_info.begin_think_token_ids              = begin_think_token_ids;
        think_info.end_think_token_ids                = end_think_token_ids;
        think_info.input_length                       = input_length;
        think_info.current_output_length              = current_output_length;
        think_info.is_beam_search                     = is_beam_search;
        think_info.pending_forced_think_end_token_ids = pending_forced_think_end_token_ids;
        think_info.process_state                      = process_state;
        if (dfa_ptr) {
            think_info.dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(*dfa_ptr);
        }
        return think_info;
    }
};

class ThinkModeLogitsProcessorTestPeer;

// Snapshot of think state published lock-free for spec verifiers. Combines the
// eligibility flag, the cloned info, and a monotonic version counter so the spec
// path observes a consistent triple via a single atomic_load.
struct ThinkModeSpecSnapshot {
    bool            eligible = false;
    StreamThinkInfo info;
    uint64_t        version = 0;
};

class ThinkModeLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    ThinkModeLogitsProcessor() = default;
    explicit ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos);
    ~ThinkModeLogitsProcessor() override = default;

    static std::shared_ptr<ThinkModeLogitsProcessor> fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                       int32_t                        num);

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    bool    isSpecVerifyEligible() const override;
    int     tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool    isStateful() const override;
    int64_t acceptedTokenLen() const override;

private:
    friend class ThinkModeLogitsProcessorTestPeer;

    void publishSpecSnapshotLocked();

    std::vector<StreamThinkInfo> think_infos_;
    mutable std::mutex           mutex_;
    // `spec_eligible_` is fixed at construction time: only updateMultiSeqStatus
    // could resize think_infos_, and that path is only taken for beam search,
    // which is itself ineligible for spec. So the flag never needs to flip.
    bool                                          spec_eligible_ = false;
    std::shared_ptr<const ThinkModeSpecSnapshot>  spec_snapshot_;
    uint64_t                                      spec_snapshot_version_ = 0;
};

using ThinkModeLogitsProcessorPtr = std::shared_ptr<ThinkModeLogitsProcessor>;

// Test-only peer that exposes internal DFA state. Definition lives in the .cc;
// linkage from tests is fine because the .o ends up in the same archive.
class ThinkModeLogitsProcessorTestPeer {
public:
    static std::vector<size_t> thinkEndTokensStatus(ThinkModeLogitsProcessor& proc);
};

}  // namespace rtp_llm
