#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#include <gtest/gtest.h>

#include <map>
#include <vector>

using Reporter        = rtp_llm::WallClockMetricsLoopReporter<rtp_llm::RtpLLMWallClockTokenPSMetrics,
                                                      rtp_llm::RtpLLMTokenPSMetricsCollector>;
using PlanCollectors  = std::map<int32_t, rtp_llm::RtpLLMTokenPSMetricsCollector>;
using TokenPlan       = std::map<int32_t, bool>;

namespace rtp_llm {

// A service that never reported a priority bucket keeps the legacy untagged
// "priority=0" idle report instead of emitting per-priority samples.
TEST(WallClockMetricsLoopReporterPlan, NeverActiveServiceUsesLegacyIdleReport) {
    std::vector<int32_t> remembered;
    TokenPlan            has_tokens;
    PlanCollectors       none;

    EXPECT_FALSE(Reporter::planRoundReports(false, none, remembered, has_tokens));
    EXPECT_TRUE(remembered.empty());
    EXPECT_TRUE(has_tokens.empty());
}

// An active round with priority buckets emits its priorities and remembers them.
TEST(WallClockMetricsLoopReporterPlan, ActiveRoundEmitsAndRemembersPriorities) {
    std::vector<int32_t> remembered;
    TokenPlan            has_tokens;
    PlanCollectors       round;
    round[50].addTokenSize(1000, 1000, 0, 1000, 1000000);

    EXPECT_TRUE(Reporter::planRoundReports(true, round, remembered, has_tokens));
    EXPECT_EQ(remembered, (std::vector<int32_t>{50}));
    EXPECT_EQ(has_tokens, (TokenPlan{{50, true}}));
}

// The core regression: a remembered priority without tokens this round must
// emit a ZERO sample rather than going silent, so per-series time averages
// divide by the full wall clock (idle seconds included) instead of silently
// inflating to the busy-time rate.
TEST(WallClockMetricsLoopReporterPlan, IdleRoundEmitsZeroSamplesForRememberedPriorities) {
    std::vector<int32_t> remembered{50};
    TokenPlan            has_tokens;
    PlanCollectors       none;

    EXPECT_TRUE(Reporter::planRoundReports(false, none, remembered, has_tokens));
    EXPECT_EQ(has_tokens, (TokenPlan{{50, false}}));
    EXPECT_EQ(remembered, (std::vector<int32_t>{50}));
}

// A mixed active round: the token-carrying priority reports metrics while the
// other remembered priority reports a zero sample in the same round.
TEST(WallClockMetricsLoopReporterPlan, ActiveRoundKeepsIdlePrioritySampled) {
    std::vector<int32_t> remembered{30, 50};
    TokenPlan            has_tokens;
    PlanCollectors       round;
    round[50].addTokenSize(1000, 1000, 0, 1000, 1000000);

    EXPECT_TRUE(Reporter::planRoundReports(true, round, remembered, has_tokens));
    EXPECT_EQ(has_tokens, (TokenPlan{{30, false}, {50, true}}));
    // Recency order: 50 stays remembered, 30 is preserved behind it.
    EXPECT_EQ(remembered, (std::vector<int32_t>{30, 50}));
}

// Token metrics without any priority bucket keep the legacy untagged report.
TEST(WallClockMetricsLoopReporterPlan, ActiveRoundWithoutBucketsUsesLegacyReport) {
    std::vector<int32_t> remembered{50};
    TokenPlan            has_tokens;
    PlanCollectors       none;

    EXPECT_FALSE(Reporter::planRoundReports(true, none, remembered, has_tokens));
    EXPECT_EQ(remembered, (std::vector<int32_t>{50}));
    EXPECT_TRUE(has_tokens.empty());
}

// The remembered set is bounded: the oldest priority is evicted once more than
// kMaxRememberedPriorities distinct priorities have carried tokens.
TEST(WallClockMetricsLoopReporterPlan, RememberedPrioritiesAreBounded) {
    std::vector<int32_t> remembered;
    TokenPlan            has_tokens;
    for (int32_t priority = 0; priority <= 50; ++priority) {
        PlanCollectors round;
        round[priority].addTokenSize(1, 1, 0, 1, 1000);
        EXPECT_TRUE(Reporter::planRoundReports(true, round, remembered, has_tokens));
    }
    EXPECT_EQ(remembered.size(), 16u);
    EXPECT_EQ(remembered.front(), 35);
    EXPECT_EQ(remembered.back(), 50);
}

// The zero sample a remembered-but-idle priority emits reports 0 wall TPS with
// the round's report window, not a stale or missing value.
TEST(WallClockMetricsLoopReporterPlan, IdleZeroSampleReportsZeroWallTpsWithWindow) {
    RtpLLMTokenPSMetricsCollector idle;
    idle.setReportWindowUs(1000000);
    EXPECT_DOUBLE_EQ(idle.contextWallTPSWithCache(), 0.0);
    EXPECT_DOUBLE_EQ(idle.contextWallTPS(), 0.0);
    EXPECT_FALSE(idle.reportZeroTPS());
    EXPECT_EQ(idle.reportWindowUs(), 1000000);
}

}  // namespace rtp_llm
