#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

ErrorInfo matcherError(const char* op, const char* detail) {
    return ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION,
                     std::string("grammar matcher ") + op + " exception: " + detail);
}

template<typename Fn>
using MatcherCallResult = std::conditional_t<std::is_void_v<std::invoke_result_t<Fn>>,
                                             ErrorInfo,
                                             ErrorResult<std::decay_t<std::invoke_result_t<Fn>>>>;

template<typename Fn>
auto matcherCall(const char* op, Fn&& fn) -> MatcherCallResult<Fn> {
    using ValueType = std::invoke_result_t<Fn>;

    try {
        if constexpr (std::is_void_v<ValueType>) {
            std::forward<Fn>(fn)();
            return ErrorInfo::OkStatus();
        } else {
            return std::forward<Fn>(fn)();
        }
    } catch (const std::exception& e) {
        return matcherError(op, e.what());
    } catch (...) {
        return matcherError(op, "unknown");
    }
}

}  // namespace

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     std::optional<std::vector<int32_t>>        override_stop_tokens,
                                     bool                                       terminate_without_stop_token,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)),
    override_stop_tokens_(std::move(override_stop_tokens)),
    terminate_without_stop_token_(terminate_without_stop_token),
    max_rollback_tokens_(max_rollback_tokens) {
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(
        *compiled_, override_stop_tokens_, terminate_without_stop_token_, max_rollback_tokens_);
}

ErrorResult<bool> RtpGrammarMatcher::acceptToken(int32_t token_id) {
    return matcherCall("acceptToken", [&] {
        const bool ok = matcher_->AcceptToken(token_id);
        if (!ok) {
            // Spec-verify DFS reacts on the bool; keep at DEBUG to avoid log floods.
            RTP_LLM_LOG_DEBUG("RtpGrammarMatcher::acceptToken REJECTED token=%d, num_accepted=%ld, terminated=%d",
                              token_id,
                              num_accepted_,
                              static_cast<int>(matcher_->IsTerminated()));
            return false;
        }
        ++num_accepted_;
        return true;
    });
}

ErrorResult<bool> RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t token_id : tokens) {
        auto accepted = acceptToken(token_id);
        if (!accepted.ok()) {
            return accepted.status();
        }
        if (!accepted.value()) {
            return false;
        }
    }
    return true;
}

ErrorResult<bool> RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    return matcherCall("fillBitmask", [&] { return matcher_->FillNextTokenBitmask(bitmask, idx); });
}

ErrorResult<bool> RtpGrammarMatcher::isTerminated() const {
    return matcherCall("isTerminated", [&] { return matcher_->IsTerminated(); });
}

ErrorInfo RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return ErrorInfo::OkStatus();
    }
    return matcherCall("rollback", [&] {
        matcher_->Rollback(n);
        num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
    });
}

ErrorResult<int32_t> RtpGrammarMatcher::vocabSize() const {
    return matcherCall("vocabSize", [&] { return compiled_->GetTokenizerInfo().GetVocabSize(); });
}

}  // namespace rtp_llm
