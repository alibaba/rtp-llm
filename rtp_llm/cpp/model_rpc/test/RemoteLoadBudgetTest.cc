#include "rtp_llm/cpp/model_rpc/PropagatedClientContext.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadBudget.h"

#include <chrono>
#include <cstdint>
#include <limits>

#include "grpc/impl/codegen/propagation_bits.h"
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

RemoteLoadSystemClock::time_point systemAt(int64_t unix_ms) {
    return RemoteLoadSystemClock::time_point(std::chrono::milliseconds(unix_ms));
}

RemoteLoadSteadyClock::time_point steadyAt(int64_t elapsed_ms) {
    return RemoteLoadSteadyClock::time_point(std::chrono::milliseconds(elapsed_ms));
}

TEST(RemoteLoadBudgetTest, ConsumedNinetyOfHundredMillisecondsLeavesTen) {
    const auto absolute_deadline = saturatingDeadlineUnixMs(/*start_unix_ms=*/1'000, /*timeout_ms=*/100);
    const auto budget = makeRemoteLoadBudget(absolute_deadline,
                                             RemoteLoadSystemClock::time_point::max(),
                                             systemAt(1'090),
                                             steadyAt(5'000));

    ASSERT_FALSE(budget.expired());
    EXPECT_EQ(budget.deadline_unix_ms, 1'100);
    EXPECT_EQ(budget.remaining_ms, 10);
    EXPECT_EQ(budget.system_deadline, systemAt(1'100));
    EXPECT_EQ(budget.steady_deadline, steadyAt(5'010));
}

TEST(RemoteLoadBudgetTest, FenceWaitReusesAdmissionDeadlineAfterTransfer) {
    const auto budget = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'100,
                                             RemoteLoadSystemClock::time_point::max(),
                                             systemAt(1'000),
                                             steadyAt(5'000));
    const auto after_transfer = steadyAt(5'090);

    ASSERT_FALSE(budget.expired());
    ASSERT_EQ(budget.remaining_ms, 100);
    EXPECT_EQ(budget.steady_deadline, steadyAt(5'100));
    EXPECT_EQ(std::chrono::duration_cast<std::chrono::milliseconds>(budget.steady_deadline - after_transfer), 10ms);
    EXPECT_NE(budget.steady_deadline, after_transfer + std::chrono::milliseconds(budget.remaining_ms));
}

TEST(RemoteLoadBudgetTest, ParentDeadlineWinsWithoutResettingBudget) {
    const auto budget = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'100,
                                             systemAt(1'070),
                                             systemAt(1'020),
                                             steadyAt(8'000));

    ASSERT_FALSE(budget.expired());
    EXPECT_EQ(budget.deadline_unix_ms, 1'070);
    EXPECT_EQ(budget.remaining_ms, 50);
    EXPECT_EQ(budget.system_deadline, systemAt(1'070));
    EXPECT_EQ(budget.steady_deadline, steadyAt(8'050));
}

TEST(RemoteLoadBudgetTest, ReceivingRpcUsesLocallyTranslatedParentDeadlineAcrossClockSkew) {
    const auto budget = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/900,
                                             systemAt(1'070),
                                             systemAt(1'020),
                                             steadyAt(8'000),
                                             /*relative_timeout_cap_ms=*/100,
                                             /*parent_deadline_authoritative=*/true);

    ASSERT_FALSE(budget.expired());
    EXPECT_EQ(budget.deadline_unix_ms, 1'070);
    EXPECT_EQ(budget.remaining_ms, 50);
    EXPECT_EQ(budget.system_deadline, systemAt(1'070));
    EXPECT_EQ(budget.steady_deadline, steadyAt(8'050));
}

TEST(RemoteLoadBudgetTest, RelativeTimeoutCanOnlyShortenImmutableDeadline) {
    const auto shortened = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'100,
                                                RemoteLoadSystemClock::time_point::max(),
                                                systemAt(1'020),
                                                steadyAt(8'000),
                                                /*relative_timeout_cap_ms=*/30);
    const auto immutable_wins = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'050,
                                                     RemoteLoadSystemClock::time_point::max(),
                                                     systemAt(1'020),
                                                     steadyAt(8'000),
                                                     /*relative_timeout_cap_ms=*/100);

    ASSERT_FALSE(shortened.expired());
    EXPECT_EQ(shortened.deadline_unix_ms, 1'050);
    EXPECT_EQ(shortened.remaining_ms, 30);
    ASSERT_FALSE(immutable_wins.expired());
    EXPECT_EQ(immutable_wins.deadline_unix_ms, 1'050);
    EXPECT_EQ(immutable_wins.remaining_ms, 30);
}

TEST(RemoteLoadBudgetTest, ExpiredParentDeadlineDoesNotAdmitDownstreamWork) {
    const auto budget = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'100,
                                             systemAt(1'019),
                                             systemAt(1'020),
                                             steadyAt(8'000));

    EXPECT_TRUE(budget.expired());
    EXPECT_EQ(budget.deadline_unix_ms, 1'019);
    EXPECT_EQ(budget.remaining_ms, 0);
    EXPECT_EQ(budget.system_deadline, systemAt(1'020));
    EXPECT_EQ(budget.steady_deadline, steadyAt(8'000));
}

TEST(RemoteLoadBudgetTest, ExpiredBudgetDoesNotAdmitDownstreamWork) {
    const auto budget = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'100,
                                             RemoteLoadSystemClock::time_point::max(),
                                             systemAt(1'100),
                                             steadyAt(9'000));
    int downstream_calls = 0;
    if (canAdmitRemoteLoad(budget, /*cancelled=*/false)) {
        ++downstream_calls;
    }

    EXPECT_TRUE(budget.expired());
    EXPECT_EQ(budget.remaining_ms, 0);
    EXPECT_EQ(budget.system_deadline, systemAt(1'100));
    EXPECT_EQ(budget.steady_deadline, steadyAt(9'000));
    EXPECT_EQ(downstream_calls, 0);
}

TEST(RemoteLoadBudgetTest, CancelledRequestDoesNotAdmitDownstreamWork) {
    const auto budget = makeRemoteLoadBudget(/*absolute_deadline_unix_ms=*/1'100,
                                             RemoteLoadSystemClock::time_point::max(),
                                             systemAt(1'020),
                                             steadyAt(9'000));
    int downstream_calls = 0;
    if (canAdmitRemoteLoad(budget, /*cancelled=*/true)) {
        ++downstream_calls;
    }

    ASSERT_FALSE(budget.expired());
    EXPECT_FALSE(canAdmitRemoteLoad(budget, /*cancelled=*/true));
    EXPECT_EQ(downstream_calls, 0);
}

TEST(RemoteLoadBudgetTest, DeadlineAdditionSaturatesAndNonPositiveTimeoutDoesNotExtend) {
    EXPECT_EQ(saturatingDeadlineUnixMs(std::numeric_limits<int64_t>::max() - 5, 10),
              std::numeric_limits<int64_t>::max());
    EXPECT_EQ(saturatingDeadlineUnixMs(1'000, 0), 1'000);
    EXPECT_EQ(saturatingDeadlineUnixMs(1'000, -1), 1'000);
}

TEST(RemoteLoadBudgetTest, SteadyDeadlineAdditionSaturates) {
    const auto now = RemoteLoadSteadyClock::time_point::max() - 5ms;

    EXPECT_EQ(saturatingSteadyDeadline(now, /*remaining_ms=*/10),
              RemoteLoadSteadyClock::time_point::max());
}

TEST(RemoteLoadBudgetTest, ChildContextEnablesDeadlineAndCancellationPropagation) {
    const auto propagation = remoteLoadPropagationOptions();

    EXPECT_NE(propagation.c_bitmask() & GRPC_PROPAGATE_DEADLINE, 0u);
    EXPECT_NE(propagation.c_bitmask() & GRPC_PROPAGATE_CANCELLATION, 0u);
}

}  // namespace
}  // namespace rtp_llm
