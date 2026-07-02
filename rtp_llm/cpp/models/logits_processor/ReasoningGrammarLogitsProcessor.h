#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarMaskCore.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

namespace rtp_llm {

class RtpGrammarMatcher;
class GenerateInput;
struct SamplerInputs;

// Reasoning-grammar concrete action set. OutsideThink routes to the grammar matcher;
// inside/closing think masks the {begin, eos} boundaries or forces the end-think token.
enum class ReasoningGrammarAction {
    GrammarRoute,
    AllowAll,
    ForceEndToken,
    MaskBoundaries,
};

struct ReasoningGrammarDecision {
    ReasoningGrammarAction action       = ReasoningGrammarAction::GrammarRoute;
    int32_t                forced_token = -1;
    int32_t                begin_token  = -1;
    int32_t                eos_token    = -1;
};

// Owns the think-phase state machine for reasoning-grammar decoding (a pure think concern):
// per step it decides whether to route to the grammar matcher or to mask think boundaries /
// force-close the think block, and advances think state on commit. Grammar masking/acceptance
// lives in ReasoningGrammarLogitsProcessor, which *composes* a ThinkRouter with a GrammarMaskCore
// rather than interleaving raw think_info_ field mutations with grammar logic.
class ThinkRouter {
public:
    enum class CommitRoute {
        ConsumedByThink,  // token handled by the think state machine (pending-drain / DFA advance)
        RouteToGrammar,   // think output length already advanced; caller commits token to grammar
    };

    // Verify path cursor: owns a throwaway think-state snapshot and applies the
    // same routing/advance rules without mutating the live ThinkRouter.
    class VerifyCursor {
    public:
        VerifyCursor(StreamThinkInfo state, int64_t eos_token_id);

        ReasoningGrammarDecision decideForMask();
        bool                     tokenBelongsToGrammar() const;
        void                     commitThinkToken(int32_t token_id);
        void                     commitGrammarToken();

        const std::vector<int>& beginThinkTokenIds() const {
            return state_.begin_think_token_ids;
        }
        const std::vector<int>& endThinkTokenIds() const {
            return state_.end_think_token_ids;
        }

    private:
        StreamThinkInfo state_;
        int64_t         eos_token_id_;
    };

    ThinkRouter(int              max_thinking_tokens,
                std::vector<int> begin_think_token_ids,
                std::vector<int> end_think_token_ids,
                int32_t          input_length,
                int64_t          eos_token_id);

    int64_t committedOutputLen() const {
        return info_.current_output_length;
    }
    const std::vector<int>& beginThinkTokenIds() const {
        return info_.begin_think_token_ids;
    }
    const std::vector<int>& endThinkTokenIds() const {
        return info_.end_think_token_ids;
    }

    // process() path: decide using the live think state at the given emitted-token count.
    ReasoningGrammarDecision decideForMask(int tokens_emitted) {
        return decide(info_, eos_token_id_, tokens_emitted);
    }
    // Two-phase force-close protocol phase 1 (paired with commitToken's pending-drain).
    void commitForcedEnd(int32_t forced_token);

    // updateStatus() path: advance think state for one committed token.
    CommitRoute commitToken(int32_t token_id);

    VerifyCursor verifyCursor() const;

    // Static verify helpers operating on the snapshot copy (no live state mutated).
    static ReasoningGrammarDecision decide(StreamThinkInfo& state, int64_t eos_token_id, int tokens_emitted);
    static bool                     tokenBelongsToGrammar(const StreamThinkInfo& state);
    static void                     advanceForVerify(StreamThinkInfo& state, int32_t token_id);

private:
    StreamThinkInfo info_;
    int64_t         eos_token_id_;
};

// Composes a ThinkRouter (think-phase state machine) with a GrammarMaskCore (grammar matcher):
// routes each step to one or the other and layers think force-close on top of grammar masking.
class ReasoningGrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    ReasoningGrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                    int64_t                            eos_token_id,
                                    int                                max_thinking_tokens,
                                    std::vector<int>                   begin_think_token_ids,
                                    std::vector<int>                   end_think_token_ids,
                                    int32_t                            input_length);
    ~ReasoningGrammarLogitsProcessor() override = default;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& /*src_batch_indices*/) override {}
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

    bool isStateful() const override {
        return true;
    }

    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;

    int64_t committedOutputLen() const override;

    bool hasError() const override {
        return has_error_.load(std::memory_order_acquire);
    }
    ErrorInfo error() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_info_;
    }

private:
    void applyMaskLocked(const SamplerInputs& inputs, size_t batch_idx, ErrorInfo& out_err);
    void applyGrammarMaskLocked(const torch::Tensor& logits, ErrorInfo& out_err);
    void acceptCommittedGrammarTokenLocked(int32_t token_id, ErrorInfo& out_err);
    int  runSpecVerifyLocked(const SpecLogitsProcessorRequest& request, ErrorInfo& out_err);

    void setError(ErrorCode code, std::string msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_error_.load(std::memory_order_relaxed)) {
            error_info_ = ErrorInfo(code, std::move(msg));
            has_error_.store(true, std::memory_order_release);
        }
    }
    void setError(const ErrorInfo& info) {
        if (!info.hasError()) {
            return;
        }
        setError(info.code(), info.ToString());
    }

    mutable std::mutex mutex_;
    GrammarMaskCore    mask_core_;
    int64_t            eos_token_id_;
    ThinkRouter        think_;

    std::atomic<bool> has_error_{false};
    ErrorInfo         error_info_;
};

using ReasoningGrammarLogitsProcessorPtr = std::shared_ptr<ReasoningGrammarLogitsProcessor>;

}  // namespace rtp_llm
