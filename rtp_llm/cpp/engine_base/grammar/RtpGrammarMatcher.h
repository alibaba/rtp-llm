#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

class RtpGrammarMatcher {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      bool                                       require_reasoning,
                      std::optional<std::vector<int>>            think_end_token_ids,
                      std::optional<std::vector<int>>            override_stop_tokens = std::nullopt,
                      int                                        max_rollback_tokens  = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;

    void initReasoning(bool in_think_body);

    bool acceptToken(int32_t token_id);
    bool acceptTokens(const std::vector<int32_t>& tokens);
    bool fillBitmask(DLTensor* bitmask, int32_t idx);
    void rollback(int n);

    bool isPassthroughForMask() const noexcept {
        return require_reasoning_ && tokens_after_think_end_ < 0;
    }
    bool    isTerminated() const;
    int32_t vocabSize() const;

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

private:
    struct ReasonerState {
        int    tokens_after_think_end;
        size_t think_end_match_pos;
    };
    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    const bool             require_reasoning_;
    const std::vector<int> think_end_token_ids_;
    std::vector<size_t>    think_end_lps_;

    // < 0: inside thinking body, parser frozen. >= 0: grammar is active.
    int                        tokens_after_think_end_ = 0;
    size_t                     think_end_match_pos_    = 0;
    std::vector<ReasonerState> reasoner_state_history_;
    int64_t                    num_accepted_ = 0;
    bool                       finished_     = false;
};

using RtpGrammarMatcherPtr = std::shared_ptr<RtpGrammarMatcher>;

}  // namespace rtp_llm
