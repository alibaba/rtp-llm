#include "rtp_llm/cpp/repetition/OnlineRepetitionTracker.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define RTP_EXPECT(expr)                                                                            \
    do {                                                                                            \
        if (!(expr)) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " check failed: " << #expr << std::endl;   \
            std::abort();                                                                           \
        }                                                                                           \
    } while (false)

namespace rtp_llm {
namespace {

OnlineRepetitionConfig testConfig() {
    OnlineRepetitionConfig config;
    config.min_repeats = 3;
    config.min_duplicate_tokens = 0;
    config.max_period = 512;
    return config;
}

void testDetectsTruncatedPeriodicRun() {
    std::vector<int> tokens;
    for (int i = 0; i < 8; ++i) {
        tokens.insert(tokens.end(), {10, 20, 30, 99});
    }
    tokens.insert(tokens.end(), {10, 20});

    const auto result = detectOnlineRepetitionMax(tokens, testConfig());
    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_unit_size == 4);
    RTP_EXPECT(result.repeat_count == 8);
    RTP_EXPECT(result.partial_tail_tokens == 2);
    RTP_EXPECT(result.covered_token_count == 34);
    RTP_EXPECT(result.duplicate_token_count == 30);
    RTP_EXPECT(result.start_index == 0);
    RTP_EXPECT(result.end_index == static_cast<int>(tokens.size()));
}

void testIgnoresTwoRepeats() {
    std::vector<int> tokens;
    for (int i = 0; i < 2; ++i) {
        tokens.insert(tokens.end(), {10, 20, 30, 99});
    }

    const auto result = detectOnlineRepetitionMax(tokens, testConfig());
    RTP_EXPECT(!result.hit);
}

void testMiddleRepeatFollowedBySuffix() {
    std::vector<int> tokens = {7, 8};
    for (int i = 0; i < 5; ++i) {
        tokens.insert(tokens.end(), {1, 2, 3});
    }
    tokens.insert(tokens.end(), {4, 5, 6});

    const auto result = detectOnlineRepetitionMax(tokens, testConfig());
    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_unit_size == 3);
    RTP_EXPECT(result.duplicate_token_count == 12);
    RTP_EXPECT(result.start_index == 2);
    RTP_EXPECT(result.end_index == 17);
}

void testMinDuplicateThreshold() {
    OnlineRepetitionConfig config = testConfig();
    config.min_duplicate_tokens = 32;
    const std::vector<int> tokens = {9, 8, 7, 9, 8, 7, 9, 8, 7};

    const auto result = detectOnlineRepetitionMax(tokens, config);
    RTP_EXPECT(!result.hit);
}

void testFinalPartialTailCountsTowardThreshold() {
    OnlineRepetitionConfig config = testConfig();
    config.min_duplicate_tokens = 32;

    std::vector<int> unit;
    for (int i = 0; i < 15; ++i) {
        unit.push_back(100 + i);
    }
    std::vector<int> tokens;
    for (int i = 0; i < 3; ++i) {
        tokens.insert(tokens.end(), unit.begin(), unit.end());
    }
    tokens.insert(tokens.end(), unit.begin(), unit.begin() + 14);

    const auto result = detectOnlineRepetitionMax(tokens, config);
    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_unit_size == 15);
    RTP_EXPECT(result.repeat_count == 3);
    RTP_EXPECT(result.partial_tail_tokens == 14);
    RTP_EXPECT(result.covered_token_count == 59);
    RTP_EXPECT(result.duplicate_token_count == 44);
    RTP_EXPECT(result.start_index == 0);
    RTP_EXPECT(result.end_index == static_cast<int>(tokens.size()));
}

void testLongSameToken() {
    OnlineRepetitionConfig config = testConfig();
    config.min_duplicate_tokens = 32;
    const std::vector<int> tokens(1000, 95553);

    const auto result = detectOnlineRepetitionHitOnly(tokens, config);
    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_unit_size == 1);
    RTP_EXPECT(result.duplicate_token_count >= 32);
}

void testRandomNoHit() {
    OnlineRepetitionConfig config = testConfig();
    config.min_duplicate_tokens = 32;
    std::mt19937 rng(20260610);
    std::vector<int> tokens;
    tokens.reserve(4096);
    for (int i = 0; i < 4096; ++i) {
        tokens.push_back(static_cast<int>(rng() + 1000000U));
    }

    const auto result = detectOnlineRepetitionMax(tokens, config);
    RTP_EXPECT(!result.hit);
}

}  // namespace
}  // namespace rtp_llm

int main() {
    rtp_llm::testDetectsTruncatedPeriodicRun();
    rtp_llm::testIgnoresTwoRepeats();
    rtp_llm::testMiddleRepeatFollowedBySuffix();
    rtp_llm::testMinDuplicateThreshold();
    rtp_llm::testFinalPartialTailCountsTowardThreshold();
    rtp_llm::testLongSameToken();
    rtp_llm::testRandomNoHit();
    std::cout << "OnlineRepetitionTracker tests passed\n";
    return 0;
}
