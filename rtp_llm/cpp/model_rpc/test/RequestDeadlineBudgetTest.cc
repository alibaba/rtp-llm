#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/RequestDeadlineBudget.h"

namespace rtp_llm {

TEST(RequestDeadlineBudgetTest, LegacyRequestStartsAtLocalReceipt) {
    const auto budget = makeRequestDeadlineBudget(0, 500, 10'000'000);

    EXPECT_FALSE(budget.has_deadline);
    EXPECT_FALSE(budget.expired);
    EXPECT_EQ(budget.timeout_ms, 500);
    EXPECT_EQ(budget.begin_time_us, 10'000'000);
}

TEST(RequestDeadlineBudgetTest, AbsoluteDeadlineRetainsElapsedTimeAcrossHops) {
    const auto first = makeRequestDeadlineBudget(10'300, 500, 10'000'000);
    const auto next  = makeRequestDeadlineBudget(10'300, 500, 10'200'000);

    EXPECT_TRUE(first.has_deadline);
    EXPECT_FALSE(first.expired);
    EXPECT_FALSE(next.expired);
    EXPECT_EQ(first.begin_time_us, 9'800'000);
    EXPECT_EQ(next.begin_time_us, first.begin_time_us);
    EXPECT_EQ(next.timeout_ms, 500);
}

TEST(RequestDeadlineBudgetTest, DeadlineCannotExtendRelativeTimeout) {
    const auto budget = makeRequestDeadlineBudget(20'000, 500, 10'000'000);

    EXPECT_FALSE(budget.expired);
    EXPECT_EQ(budget.begin_time_us, 10'000'000);
    EXPECT_EQ(budget.timeout_ms, 500);
}

TEST(RequestDeadlineBudgetTest, ExpiredAndMalformedDeadlinesFailClosed) {
    const auto expired = makeRequestDeadlineBudget(9'999, 500, 10'000'000);
    const auto exact = makeRequestDeadlineBudget(10'000, 500, 10'000'000);
    const auto negative = makeRequestDeadlineBudget(-1, 500, 10'000'000);
    const auto no_relative = makeRequestDeadlineBudget(10'200, 0, 10'000'000);

    EXPECT_TRUE(expired.expired);
    EXPECT_TRUE(exact.expired);
    EXPECT_TRUE(negative.expired);
    EXPECT_FALSE(no_relative.expired);
    EXPECT_EQ(no_relative.timeout_ms, 200);
    EXPECT_EQ(no_relative.begin_time_us, 10'000'000);
}

TEST(RequestDeadlineBudgetTest, LocalGrpcDeadlineIsAuthoritativeAcrossClockSkew) {
    const auto future = makeRequestDeadlineBudget(9'900, 500, 10'000'000, 10'200'000);
    const auto expired = makeRequestDeadlineBudget(10'500, 500, 10'000'000, 9'999'000);

    EXPECT_FALSE(future.expired);
    EXPECT_EQ(future.deadline_unix_us, 10'200'000);
    EXPECT_EQ(future.begin_time_us, 9'700'000);
    EXPECT_TRUE(expired.expired);
    EXPECT_EQ(expired.deadline_unix_us, 9'999'000);
}

TEST(RequestDeadlineBudgetTest, BatchItemDeadlineRetainsTransportElapsedTime) {
    const auto batch_deadline_us = 10'920'000;

    EXPECT_EQ(deriveBatchItemDeadlineUnixUs(batch_deadline_us, 1000, 100), 10'020'000);
    EXPECT_EQ(deriveBatchItemDeadlineUnixUs(batch_deadline_us, 1000, 1000), batch_deadline_us);
    EXPECT_EQ(deriveBatchItemDeadlineUnixUs(batch_deadline_us, 1000, 0), batch_deadline_us);
    EXPECT_FALSE(requestDeadlineReached(10'000'000, 20, 10'019'999));
    EXPECT_TRUE(requestDeadlineReached(10'000'000, 20, 10'020'000));
}

TEST(RequestDeadlineBudgetTest, MalformedFutureDeadlineFitsGenerateConfigTimeout) {
    const auto budget = makeRequestDeadlineBudget(
        std::numeric_limits<int64_t>::max(), 0, 10'000'000);

    EXPECT_FALSE(budget.expired);
    EXPECT_EQ(budget.timeout_ms, std::numeric_limits<int>::max());
    EXPECT_EQ(budget.begin_time_us, 10'000'000);
}

}  // namespace rtp_llm
