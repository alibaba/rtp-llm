#include "rtp_llm/cpp/utils/DecodePublicationBudget.h"

#include <gtest/gtest.h>

#include <array>
#include <future>
#include <limits>
#include <thread>

namespace rtp_llm {

TEST(DecodePublicationBudgetTest, RejectsInvalidCapacityAndEmptyWork) {
    EXPECT_THROW(DecodePublicationBudget(0, 1), std::invalid_argument);
    EXPECT_THROW(DecodePublicationBudget(1, 0), std::invalid_argument);
    DecodePublicationBudget budget(512, 4);
    EXPECT_FALSE(budget.tryAcquire(0));
    EXPECT_FALSE(budget.tryAcquire(513));
    EXPECT_EQ(budget.snapshot().jobs, 0u);
}

TEST(DecodePublicationBudgetTest, ChargesBytesUntilLastOwnerReleases) {
    DecodePublicationBudget budget(512, 4);
    auto first = budget.tryAcquire(256);
    auto second = budget.tryAcquire(256);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_FALSE(budget.tryAcquire(1));
    auto retained = first;
    first.reset();
    EXPECT_EQ(budget.snapshot().bytes, 512u);
    retained.reset();
    EXPECT_EQ(budget.snapshot().bytes, 256u);
    EXPECT_TRUE(budget.tryAcquire(256));
    second.reset();
    EXPECT_EQ(budget.snapshot().jobs, 0u);
}

TEST(DecodePublicationBudgetTest, JobBoundAppliesIndependentlyOfByteBudget) {
    DecodePublicationBudget budget(4096, 2);
    auto first = budget.tryAcquire(1);
    auto second = budget.tryAcquire(1);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_FALSE(budget.tryAcquire(1));
    EXPECT_EQ(budget.snapshot().bytes, 2u);
    second.reset();
    EXPECT_TRUE(budget.tryAcquire(1));
}

TEST(DecodePublicationBudgetTest, QuarantineOwnerKeepsAdmissionCharged) {
    DecodePublicationBudget budget(512, 1);
    auto lease = budget.tryAcquire(512);
    ASSERT_TRUE(lease);
    std::shared_ptr<void> quarantined = lease;
    lease.reset();
    EXPECT_EQ(budget.snapshot().bytes, 512u);
    EXPECT_FALSE(budget.tryAcquire(1));
    // Only the caller that proves buffer safety may drop this final owner.
    quarantined.reset();
    EXPECT_EQ(budget.snapshot().bytes, 0u);
    EXPECT_EQ(budget.snapshot().jobs, 0u);
}

TEST(DecodePublicationBudgetTest, StopPreventsNewWorkWithoutForgivingHeldPages) {
    DecodePublicationBudget budget(512, 4);
    auto lease = budget.tryAcquire(128);
    ASSERT_TRUE(lease);
    budget.stopAdmission();
    EXPECT_FALSE(budget.snapshot().accepting);
    EXPECT_EQ(budget.snapshot().bytes, 128u);
    EXPECT_FALSE(budget.tryAcquire(1));
    lease.reset();
    EXPECT_EQ(budget.snapshot().bytes, 0u);
    EXPECT_FALSE(budget.tryAcquire(1));
}

TEST(DecodePublicationBudgetTest, SizeArithmeticCannotOverflow) {
    const auto maximum = std::numeric_limits<size_t>::max();
    DecodePublicationBudget budget(maximum, 3);
    auto first = budget.tryAcquire(maximum - 1);
    ASSERT_TRUE(first);
    EXPECT_FALSE(budget.tryAcquire(2));
    auto last = budget.tryAcquire(1);
    ASSERT_TRUE(last);
    EXPECT_EQ(budget.snapshot().bytes, maximum);
}

TEST(DecodePublicationBudgetTest, ContendingAdmissionsCannotExceedCapacity) {
    DecodePublicationBudget budget(16, 2);
    std::promise<void> go;
    auto start = go.get_future().share();
    std::array<std::shared_ptr<DecodePublicationBudget::Lease>, 2> leases;
    std::array<std::thread, 2> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] { start.wait(); leases[i] = budget.tryAcquire(15); });
    }
    go.set_value();
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(static_cast<bool>(leases[0]) + static_cast<bool>(leases[1]), 1);
    EXPECT_EQ(budget.snapshot().bytes, 15u);
    leases = {};
    EXPECT_EQ(budget.snapshot().bytes, 0u);
    EXPECT_EQ(budget.snapshot().jobs, 0u);
}

}  // namespace rtp_llm
