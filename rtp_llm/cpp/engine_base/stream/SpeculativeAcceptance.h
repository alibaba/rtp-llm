#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm {

// Counts only real request rows, up to the generation/EOS/stop frontier.
// A rejection does not shorten the denominator: the remaining valid draft
// candidates were still evaluated by target verify.
inline int eligibleSpeculativeCandidates(const std::vector<int>& draft,
                                          const std::vector<int>& history_tail,
                                          int generated_before,
                                          int remaining,
                                          int min_new_tokens,
                                          int eos_token,
                                          bool ignore_eos,
                                          const std::vector<std::vector<int>>& stop_words) {
    auto prefix = history_tail;
    const int limit = std::min<int>(draft.size(), std::max(0, remaining));
    for (int i = 0; i < limit; ++i) {
        if (draft[i] < 0) {
            return i;
        }
        prefix.push_back(draft[i]);
        if (generated_before + i + 1 < min_new_tokens) {
            continue;
        }
        if (!ignore_eos && draft[i] == eos_token) {
            return i + 1;
        }
        for (const auto& stop : stop_words) {
            if (stop.empty() || stop.size() > prefix.size()
                || (ignore_eos && stop.size() == 1 && stop[0] == eos_token)) {
                continue;
            }
            if (std::equal(stop.rbegin(), stop.rend(), prefix.rbegin())) {
                return i + 1;
            }
        }
    }
    return limit;
}

struct SpeculativeRoundMetrics {
    uint64_t id = 0;
    int sequence_length_before = 0;
    int proposed_candidates = 0;
    int verified_candidates = 0;
    int accepted_draft_tokens = 0;
    bool forced_fixture = false;
};

// Owned by the request's authoritative Decode TP root, under the stream lock.
// Response snapshots are cumulative; clients replace snapshots by request ID
// rather than summing TP copies or repeated streaming responses.
class SpeculativeAcceptanceStats {
public:
    void record(const SpeculativeRoundMetrics& round, int committed_tokens) {
        if (!round.id || round.id <= last_id_) {
            return;
        }
        if (round.verified_candidates < 0 || round.proposed_candidates < round.verified_candidates
            || round.accepted_draft_tokens < 0 || round.accepted_draft_tokens > round.proposed_candidates
            || committed_tokens < 0) {
            throw std::invalid_argument("invalid speculative acceptance accounting");
        }
        last_id_ = round.id;
        if (values_.empty()) {
            values_ = {{"version", 1}, {"proposal_rounds", 0}, {"proposal_candidates", 0},
                       {"verify_rounds", 0}, {"verified_candidates", 0}, {"accepted_draft_tokens", 0},
                       {"generated_tokens", 0}, {"excluded_forced_rounds", 0}};
        }
        if (round.forced_fixture) {
            ++values_["excluded_forced_rounds"];
            return;
        }
        const int accepted = std::min({round.accepted_draft_tokens, round.verified_candidates, committed_tokens});
        ++values_["proposal_rounds"];
        ++values_["verify_rounds"];
        values_["proposal_candidates"] += round.proposed_candidates;
        values_["verified_candidates"] += round.verified_candidates;
        values_["accepted_draft_tokens"] += accepted;
        values_["generated_tokens"] += committed_tokens;
        for (int n = 0; n <= round.proposed_candidates; ++n) {
            values_.try_emplace("accept_length_" + std::to_string(n), 0);
        }
        ++values_["accept_length_" + std::to_string(accepted)];
    }

    const std::map<std::string, int64_t>& values() const {
        return values_;
    }

private:
    uint64_t last_id_ = 0;
    std::map<std::string, int64_t> values_;
};

}  // namespace rtp_llm
