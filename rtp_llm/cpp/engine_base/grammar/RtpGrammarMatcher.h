#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

// Per-stream xgrammar::GrammarMatcher adapter. With require_reasoning_, the parser
// is frozen inside the think body and resumes once the think_end KMP match completes.
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
    // Rolls back xgrammar by the count of xgrammar-accepted tokens in the last n.
    // Reasoner KMP state is untouched — pair with reasonerSnapshot/restoreReasoner.
    void rollback(int n);

    struct ReasonerSnapshot {
        int    tokens_after_think_end = 0;
        size_t think_end_match_pos    = 0;
    };

    ReasonerSnapshot reasonerSnapshot() const noexcept {
        return {tokens_after_think_end_, think_end_match_pos_};
    }
    void restoreReasoner(const ReasonerSnapshot& snap) noexcept {
        if (require_reasoning_) {
            tokens_after_think_end_ = snap.tokens_after_think_end;
            think_end_match_pos_    = snap.think_end_match_pos;
        }
    }

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
    size_t nextThinkEndMatchPos(int32_t token_id) const noexcept;

    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    const std::vector<int> think_end_token_ids_;
    const bool             require_reasoning_;
    std::vector<size_t>    think_end_lps_;

    // < 0: inside thinking body, parser frozen. >= 0: grammar is active.
    int     tokens_after_think_end_ = 0;
    size_t  think_end_match_pos_    = 0;
    int64_t num_accepted_           = 0;
    bool    finished_               = false;
};

}  // namespace rtp_llm
