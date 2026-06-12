#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace rtp_llm {

// Standalone in-stream reasoning (think-body) gate. Owned by
// GrammarLogitsProcessor; the matcher itself is unaware of think semantics.
//
// Lifecycle: constructed when a request enters with require_reasoning=true and
// non-empty end_token_ids. The gate starts in passthrough (parser frozen) and
// transitions to active once the end-token sequence is observed in the token
// stream, after which it just tracks per-token snapshots for rollback.
//
// Threading: one gate per GenerateStream, touched by one sampler thread per tick.
class ReasoningGate {
public:
    // in_think_body=true keeps the gate in passthrough until end_token_ids is
    // observed; false starts already-active (still useful for symmetric rollback
    // accounting on streams that began past the think body).
    // max_history caps how many token snapshots are retained for rollback;
    // older snapshots are dropped from the front. Must match (or be ≥) the
    // matcher's max_rollback_tokens so the two stay in sync.
    ReasoningGate(std::vector<int> end_token_ids, bool in_think_body, size_t max_history);

    ReasoningGate(const ReasoningGate&)            = delete;
    ReasoningGate& operator=(const ReasoningGate&) = delete;
    ReasoningGate(ReasoningGate&&)                 = default;
    ReasoningGate& operator=(ReasoningGate&&)      = default;

    // True iff the parser is frozen and the caller should publish an allow-all
    // mask (skip xgrammar bitmask filling for this position).
    bool inPassthrough() const noexcept { return tokens_after_end_ < 0; }

    // Step one token through the gate. `forwarded_to_matcher` reflects whether
    // the caller forwarded the token to the xgrammar matcher (always false in
    // passthrough; true in active mode). The flag is recorded so rollback can
    // tell the caller exactly how many matcher accepts to undo.
    void observe(int32_t token_id, bool forwarded_to_matcher);

    // Walk back `n` snapshots. Returns the number of tokens that had been
    // forwarded to the matcher among those n — caller must matcher_->rollback()
    // by exactly that count.
    int rollback(int n);

private:
    size_t nextEndMatchPos(int32_t token_id) const noexcept;

    struct Snapshot {
        int    tokens_after_end;
        size_t match_pos;
        bool   was_active;
    };

    std::vector<int>     end_token_ids_;
    std::vector<size_t>  end_lps_;
    int                  tokens_after_end_;
    size_t               match_pos_;
    size_t               max_history_;
    std::deque<Snapshot> history_;
};

}  // namespace rtp_llm
