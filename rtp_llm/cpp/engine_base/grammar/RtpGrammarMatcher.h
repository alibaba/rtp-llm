#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

// Thin adapter over xgrammar::GrammarMatcher; one per stream, single sampler thread.
//
// When require_reasoning_ is set, the matcher tracks an in-stream think-body window:
// while inside the body the parser is frozen and fillBitmask returns false (caller
// emits an allow-all mask); once the think_end token sequence is matched (KMP), the
// matcher transitions to grammar-active and accepts/masks every subsequent token.
// rollback() unwinds both the parser and the reasoner state so MTP verify can DFS
// past the think boundary safely.
class RtpGrammarMatcher {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      bool                                       require_reasoning,
                      std::optional<std::vector<int>>            think_end_token_ids,
                      std::optional<std::vector<int32_t>>        override_stop_tokens         = std::nullopt,
                      bool                                       terminate_without_stop_token = false,
                      int                                        max_rollback_tokens          = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&)      = default;

    // Reset reasoner state. in_think_body=true → parser frozen until think_end matches.
    void initReasoning(bool in_think_body);

    [[nodiscard]] bool acceptToken(int32_t token_id);
    [[nodiscard]] bool acceptTokens(const std::vector<int32_t>& tokens);

    bool fillBitmask(DLTensor* bitmask, int32_t idx);

    bool isPassthroughForMask() const noexcept {
        return require_reasoning_ && tokens_after_think_end_ < 0;
    }

    bool isTerminated() const;
    void rollback(int n);

    int64_t numAcceptedTokens() const noexcept {
        return num_accepted_;
    }
    int32_t vocabSize() const {
        return compiled_->GetTokenizerInfo().GetVocabSize();
    }

    void markFinished() noexcept {
        finished_ = true;
    }
    bool finished() const noexcept {
        return finished_;
    }

private:
    void   transferReasonerState(int32_t token_id) noexcept;
    void   rollbackReasonerState() noexcept;
    size_t nextThinkEndMatchPos(int32_t token_id) const noexcept;

    struct ReasonerState {
        int    tokens_after_think_end;
        size_t think_end_match_pos;
    };

    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    const std::vector<int> think_end_token_ids_;
    const bool             require_reasoning_;
    std::vector<size_t>    think_end_lps_;

    // < 0: inside thinking body, parser frozen. >= 0: grammar is active.
    int                        tokens_after_think_end_ = 0;
    size_t                     think_end_match_pos_    = 0;
    std::vector<ReasonerState> reasoner_state_history_;
    int64_t                    num_accepted_ = 0;
    bool                       finished_     = false;
};

}  // namespace rtp_llm
