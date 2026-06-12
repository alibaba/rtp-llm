#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

// Thin adapter over xgrammar::GrammarMatcher: bookkeeping of accepted-token
// count, finished flag, and a single-stop-token bitmask check. Has no
// awareness of think / reasoning gating — that lives in ReasoningGate, which
// the GrammarLogitsProcessor composes around the matcher.
//
// Failure model: AcceptToken returns false (no exception) for an illegal
// token in active mode; we propagate the bool unchanged.
//
// Threading: one matcher per GenerateStream, touched by one sampler thread per tick.
class RtpGrammarMatcher {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      std::optional<std::vector<int32_t>>        override_stop_tokens         = std::nullopt,
                      bool                                       terminate_without_stop_token = false,
                      int                                        max_rollback_tokens          = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&)      = default;

    [[nodiscard]] bool acceptToken(int32_t token_id);
    [[nodiscard]] bool acceptTokens(const std::vector<int32_t>& tokens);

    bool fillBitmask(DLTensor* bitmask, int32_t idx) const;

    bool isTerminated() const;
    bool onlyStopTokenLegalNext(int32_t stop_token_id) const;
    void rollback(int n);

    int64_t numAcceptedTokens() const noexcept { return num_accepted_; }
    int32_t vocabSize() const { return compiled_->GetTokenizerInfo().GetVocabSize(); }

    void markFinished() noexcept { finished_ = true; }
    bool finished() const noexcept { return finished_; }

    int maxRollbackTokens() const noexcept { return max_rollback_tokens_; }

private:
    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    int64_t num_accepted_        = 0;
    bool    finished_            = false;
    int     max_rollback_tokens_ = 200;
};

}  // namespace rtp_llm
