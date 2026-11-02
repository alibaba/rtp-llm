#include "rtp_llm/cpp/engine_base/stream/SpeculativeAcceptance.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace rtp_llm;
    const auto count = [](std::vector<int> d, int remaining = 10, int generated = 1,
                          int min_new = 0, bool ignore = false,
                          std::vector<int> tail = {}, std::vector<std::vector<int>> stops = {}) {
        return eligibleSpeculativeCandidates(d, tail, generated, remaining, min_new, 99, ignore, stops);
    };
    assert(count({1, 2, 3}) == 3);
    assert(count({1, 2, 3}, 1) == 1);
    assert(count({1, 2, 3}, 0) == 0);
    assert(count({1, 99, 3}) == 2);
    assert(count({1, 99, 3}, 10, 1, 0, true) == 3);
    assert(count({99, 2, 3}, 10, 0, 2) == 3);
    assert(count({1, -1, 3}) == 1);
    assert(count({8, 9, 3}, 10, 1, 0, false, {7}, {{7, 8, 9}}) == 2);
    assert(count({99, 2, 3}, 10, 1, 0, true, {}, {{99}}) == 3);
    SpeculativeAcceptanceStats stats;
    assert(stats.values().empty());
    // Rejection at position zero still verified all three valid candidates.
    stats.record({1, 100, 3, 3, 0, false}, 1);
    stats.record({2, 101, 3, 3, 1, false}, 2);
    stats.record({3, 103, 3, 3, 2, false}, 3);
    stats.record({4, 106, 3, 3, 3, false}, 4);
    assert(stats.values().at("accepted_draft_tokens") == 6);
    assert(stats.values().at("verified_candidates") == 12);
    assert(stats.values().at("generated_tokens") == 10);
    for (int n = 0; n <= 3; ++n) assert(stats.values().at("accept_length_" + std::to_string(n)) == 1);
    // EOS/max truncation can retain a draft token while removing the bonus.
    stats.record({5, 110, 3, 1, 3, false}, 1);
    assert(stats.values().at("accepted_draft_tokens") == 7);
    stats.record({6, 111, 3, 0, 0, false}, 0);
    assert(stats.values().at("verify_rounds") == 6);
    assert(stats.values().at("verified_candidates") == 13);
    stats.record({6, 111, 3, 3, 3, false}, 4); // duplicate
    stats.record({3, 103, 3, 3, 2, false}, 3); // stale
    assert(stats.values().at("verify_rounds") == 6);
    stats.record({7, 111, 3, 3, 3, true}, 4);
    assert(stats.values().at("excluded_forced_rounds") == 1);
    assert(stats.values().at("verify_rounds") == 6);
    bool rejected = false;
    try { stats.record({8, 111, 3, 4, 3, false}, 4); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "Speculative acceptance CPU contracts passed\n";
}
