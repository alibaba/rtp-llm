#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     std::optional<std::vector<int32_t>>        override_stop_tokens,
                                     bool                                       terminate_without_stop_token,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)), max_rollback_tokens_(max_rollback_tokens) {
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }
    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(*compiled_,
                                                          std::move(override_stop_tokens),
                                                          terminate_without_stop_token,
                                                          max_rollback_tokens);
}

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
    const bool ok = matcher_->AcceptToken(token_id);
    if (!ok) {
        RTP_LLM_LOG_WARNING("RtpGrammarMatcher::acceptToken REJECTED token=%d, num_accepted=%ld, terminated=%d",
                            token_id,
                            num_accepted_,
                            static_cast<int>(matcher_->IsTerminated()));
        return false;
    }
    ++num_accepted_;
    return true;
}

bool RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t token_id : tokens) {
        if (!acceptToken(token_id)) {
            return false;
        }
    }
    return true;
}

bool RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) const {
    return matcher_->FillNextTokenBitmask(bitmask, idx);
}

bool RtpGrammarMatcher::isTerminated() const {
    return matcher_->IsTerminated();
}

bool RtpGrammarMatcher::onlyStopTokenLegalNext(int32_t stop_token_id) const {
    if (isTerminated() || finished()) {
        return true;
    }
    const int32_t grammar_vocab_size = vocabSize();
    if (grammar_vocab_size <= 0 || stop_token_id < 0 || stop_token_id >= grammar_vocab_size) {
        return false;
    }
    const size_t         words    = static_cast<size_t>((grammar_vocab_size + 31) / 32);
    std::vector<int32_t> bitmask(words, 0);
    int64_t              shape[2] = {1, static_cast<int64_t>(words)};
    DLTensor             dl;
    dl.data        = bitmask.data();
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    if (!fillBitmask(&dl, 0)) {
        return false;
    }
    // Word-level scan with early exit: clear the stop-token bit, then any remaining set
    // bit means another non-EOS token is legal -> not "EOS-only". This is hot during
    // MTP verify finalisation, so avoid the full O(grammar_vocab) per-token walk.
    const size_t   stop_word = static_cast<size_t>(stop_token_id) / 32;
    const uint32_t stop_bit  = 1u << (stop_token_id % 32);
    bool           stop_set  = false;
    for (size_t w = 0; w < words; ++w) {
        uint32_t bits = static_cast<uint32_t>(bitmask[w]);
        if (w == stop_word && (bits & stop_bit) != 0u) {
            stop_set = true;
            bits &= ~stop_bit;
        }
        // Clamp tail bits past grammar_vocab_size in the last word.
        if (w + 1 == words) {
            const uint32_t valid_bits = static_cast<uint32_t>(grammar_vocab_size) - static_cast<uint32_t>(w * 32);
            if (valid_bits < 32u) {
                bits &= (valid_bits == 0u) ? 0u : ((1u << valid_bits) - 1u);
            }
        }
        if (bits != 0u) {
            return false;
        }
    }
    return stop_set;
}

void RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }
    matcher_->Rollback(n);
    num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
}

}  // namespace rtp_llm
