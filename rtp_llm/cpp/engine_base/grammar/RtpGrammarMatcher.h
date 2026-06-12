#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Per-stream xgrammar::GrammarMatcher adapter.
class RtpGrammarMatcher final {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      std::optional<std::vector<int32_t>>        override_stop_tokens         = std::nullopt,
                      bool                                       terminate_without_stop_token = false,
                      int                                        max_rollback_tokens          = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&) = default;

    [[nodiscard]] ErrorResult<bool> acceptToken(int32_t token_id);
    [[nodiscard]] ErrorResult<bool> acceptTokens(const std::vector<int32_t>& tokens);

    [[nodiscard]] ErrorResult<bool> fillBitmask(DLTensor* bitmask, int32_t idx);

    [[nodiscard]] ErrorResult<bool> isTerminated() const;
    [[nodiscard]] ErrorInfo         rollback(int n);

    int64_t numAcceptedTokens() const {
        return num_accepted_;
    }
    [[nodiscard]] ErrorResult<int32_t> vocabSize() const;

    void markFinished() {
        finished_ = true;
    }
    bool finished() const {
        return finished_;
    }

private:
    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;
    std::optional<std::vector<int32_t>>         override_stop_tokens_;
    bool                                         terminate_without_stop_token_ = false;
    int                                          max_rollback_tokens_          = 200;

    int64_t num_accepted_ = 0;
    bool    finished_     = false;
};

}  // namespace rtp_llm
