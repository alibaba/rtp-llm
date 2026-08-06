#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/cuda_graph/graph_capture_lifecycle.h"

namespace rtp_llm::cuda_graph {

class GraphOwnerLeaseTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

private:
    bool old_core_dump_on_exception_{false};
};

TEST_F(GraphOwnerLeaseTest, RejectsDuplicateAcquireAndResetIsIdempotent) {
    int             acquire_count = 0;
    int             release_count = 0;
    GraphOwnerLease lease(
        [&](uintptr_t owner_id) {
            ++acquire_count;
            return GraphLifecycleContext{owner_id, 7};
        },
        [&](const GraphLifecycleContext& context) {
            EXPECT_EQ(context.owner_token, 42);
            EXPECT_EQ(context.generation, 7);
            ++release_count;
        });

    lease.acquire(42);
    try {
        lease.acquire(43);
        FAIL() << "duplicate acquire unexpectedly succeeded";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("already acquired"), std::string::npos);
    }
    EXPECT_EQ(acquire_count, 1);
    EXPECT_EQ(lease.context().owner_token, 42);
    lease.reset();
    lease.reset();
    EXPECT_EQ(release_count, 1);
}

TEST_F(GraphOwnerLeaseTest, ResetClearsLeaseWhenReleaseThrows) {
    int             release_count = 0;
    GraphOwnerLease lease([](uintptr_t owner_id) { return GraphLifecycleContext{owner_id, 9}; },
                          [&](const GraphLifecycleContext&) {
                              ++release_count;
                              throw std::runtime_error("injected release failure");
                          });

    lease.acquire(11);
    lease.reset();
    lease.reset();
    EXPECT_EQ(release_count, 1);
    EXPECT_EQ(lease.context().owner_token, 0);
}

TEST_F(GraphOwnerLeaseTest, ZeroTokenIsInertAndResetIsIdempotent) {
    int             release_count = 0;
    GraphOwnerLease lease([](uintptr_t) { return GraphLifecycleContext{0, 0}; },
                          [&](const GraphLifecycleContext&) { ++release_count; });

    lease.acquire(11);
    EXPECT_EQ(lease.context().owner_token, 0);
    EXPECT_THROW(lease.acquire(12), std::exception);
    lease.reset();
    lease.reset();
    EXPECT_EQ(release_count, 0);
}

TEST_F(GraphOwnerLeaseTest, FailedAcquireDoesNotCreateAReleasableLease) {
    int             acquire_count = 0;
    int             release_count = 0;
    GraphOwnerLease lease(
        [&](uintptr_t owner_id) {
            ++acquire_count;
            if (acquire_count == 1) {
                throw std::runtime_error("injected acquire failure");
            }
            return GraphLifecycleContext{owner_id, 13};
        },
        [&](const GraphLifecycleContext&) { ++release_count; });

    EXPECT_THROW(lease.acquire(31), std::runtime_error);
    EXPECT_EQ(lease.context().owner_token, 0);
    lease.reset();
    EXPECT_EQ(release_count, 0);

    lease.acquire(32);
    EXPECT_EQ(lease.context().owner_token, 32);
    lease.reset();
    EXPECT_EQ(release_count, 1);
}

TEST_F(GraphOwnerLeaseTest, DestructorReleasesExactlyOnceDuringUnwind) {
    int release_count = 0;
    try {
        GraphOwnerLease lease([](uintptr_t owner_id) { return GraphLifecycleContext{owner_id, 3}; },
                              [&](const GraphLifecycleContext& context) {
                                  EXPECT_EQ(context.owner_token, 19);
                                  ++release_count;
                              });
        lease.acquire(19);
        throw std::runtime_error("constructor tail failed");
    } catch (const std::runtime_error&) {}
    EXPECT_EQ(release_count, 1);
}

TEST_F(GraphOwnerLeaseTest, CaptureContextIsStableAndReadOnlyUntilReset) {
    GraphOwnerLease lease([](uintptr_t owner_id) { return GraphLifecycleContext{owner_id, 5}; },
                          [](const GraphLifecycleContext&) {});
    lease.acquire(23);
    const GraphLifecycleContext* context = lease.contextPtr();
    static_assert(std::is_same_v<decltype(lease.contextPtr()), const GraphLifecycleContext*>);
    static_assert(std::is_same_v<decltype(lease.context()), const GraphLifecycleContext&>);
    EXPECT_EQ(context, &lease.context());
    EXPECT_EQ(context->owner_token, 23);
    EXPECT_EQ(context->generation, 5);
    lease.reset();
    EXPECT_EQ(context->owner_token, 0);
    EXPECT_EQ(context->generation, 0);
}

TEST(GraphCaptureLifecycleTest, PlanningSequenceUsesSecondWarmupAsFinalPlan) {
    std::vector<std::string> events;
    runCapturePlanning([&]() { events.emplace_back("begin-planning"); },
                       [&]() { events.emplace_back("forward"); },
                       [&]() { events.emplace_back("prepare-arena"); },
                       [&]() { events.emplace_back("cancel-planning"); });

    EXPECT_EQ(events,
              (std::vector<std::string>{"begin-planning", "forward", "begin-planning", "forward", "prepare-arena"}));
}

TEST(GraphCaptureLifecycleTest, WarmupFailureCancelsPlanningBeforeCapture) {
    std::vector<std::string> events;
    EXPECT_THROW(runCapturePlanning([&]() { events.emplace_back("begin-planning"); },
                                    [&]() {
                                        events.emplace_back("forward");
                                        throw std::runtime_error("warmup failed");
                                    },
                                    [&]() { events.emplace_back("prepare-arena"); },
                                    [&]() { events.emplace_back("cancel-planning"); }),
                 std::runtime_error);

    EXPECT_EQ(events, (std::vector<std::string>{"begin-planning", "forward", "cancel-planning"}));
}

TEST(GraphCaptureLifecycleTest, PrepareFailureCancelsPlanningExactlyOnce) {
    std::vector<std::string> events;
    int                      cancel_count = 0;
    EXPECT_THROW(runCapturePlanning([&]() { events.emplace_back("begin-planning"); },
                                    [&]() { events.emplace_back("forward"); },
                                    [&]() {
                                        events.emplace_back("prepare-arena");
                                        throw std::runtime_error("prepare failed");
                                    },
                                    [&]() {
                                        ++cancel_count;
                                        events.emplace_back("cancel-planning");
                                    }),
                 std::runtime_error);

    EXPECT_EQ(cancel_count, 1);
    EXPECT_EQ(events,
              (std::vector<std::string>{
                  "begin-planning", "forward", "begin-planning", "forward", "prepare-arena", "cancel-planning"}));
}

TEST(GraphCaptureLifecycleTest, RocmAndCudaGuardOrdersAreExplicit) {
    auto run = [](CaptureGuardOrder order) {
        std::vector<std::string> events;
        runCaptureTransaction(
            order,
            [&]() { events.emplace_back("python-enter"); },
            [&]() { events.emplace_back("capture-begin"); },
            [&]() { events.emplace_back("capture"); },
            [&]() { events.emplace_back("capture-end"); },
            [&]() { events.emplace_back("python-exit"); });
        return events;
    };

    EXPECT_EQ(run(CaptureGuardOrder::BEFORE_CAPTURE_BEGIN),
              (std::vector<std::string>{"python-enter", "capture-begin", "capture", "capture-end", "python-exit"}));
    EXPECT_EQ(run(CaptureGuardOrder::AFTER_CAPTURE_BEGIN),
              (std::vector<std::string>{"capture-begin", "python-enter", "capture", "capture-end", "python-exit"}));
}

TEST(GraphCaptureLifecycleTest, CaptureFailureEndsExactlyOnceAndExitsGuard) {
    std::vector<std::string> events;
    int                      end_count = 0;
    EXPECT_THROW(runCaptureTransaction(
                     CaptureGuardOrder::BEFORE_CAPTURE_BEGIN,
                     [&]() { events.emplace_back("python-enter"); },
                     [&]() { events.emplace_back("capture-begin"); },
                     [&]() {
                         events.emplace_back("capture");
                         throw std::runtime_error("capture failed");
                     },
                     [&]() {
                         ++end_count;
                         events.emplace_back("capture-end");
                     },
                     [&]() { events.emplace_back("python-exit"); }),
                 std::runtime_error);

    EXPECT_EQ(end_count, 1);
    EXPECT_EQ(events,
              (std::vector<std::string>{"python-enter", "capture-begin", "capture", "capture-end", "python-exit"}));
}

TEST(GraphCaptureLifecycleTest, EndFailureIsNotRetriedAndStillExitsGuard) {
    std::vector<std::string> events;
    int                      end_count = 0;
    EXPECT_THROW(runCaptureTransaction(
                     CaptureGuardOrder::BEFORE_CAPTURE_BEGIN,
                     [&]() { events.emplace_back("python-enter"); },
                     [&]() { events.emplace_back("capture-begin"); },
                     [&]() { events.emplace_back("capture"); },
                     [&]() {
                         ++end_count;
                         events.emplace_back("capture-end");
                         throw std::runtime_error("end failed");
                     },
                     [&]() { events.emplace_back("python-exit"); }),
                 std::runtime_error);

    EXPECT_EQ(end_count, 1);
    EXPECT_EQ(events,
              (std::vector<std::string>{"python-enter", "capture-begin", "capture", "capture-end", "python-exit"}));
}

TEST(GraphCaptureLifecycleTest, ExitFailureDoesNotMaskCaptureFailure) {
    try {
        runCaptureTransaction(
            CaptureGuardOrder::BEFORE_CAPTURE_BEGIN,
            []() {},
            []() {},
            []() { throw std::runtime_error("capture failed"); },
            []() {},
            []() { throw std::runtime_error("exit failed"); });
        FAIL() << "capture failure was not propagated";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "capture failed");
    }
}

TEST(GraphCaptureLifecycleTest, ShutdownSynchronizationIsClaimedAtMostOnce) {
    ShutdownSynchronizationGate gate;
    EXPECT_FALSE(gate.claim(false));
    EXPECT_TRUE(gate.claim(true));
    EXPECT_FALSE(gate.claim(true));
}

}  // namespace rtp_llm::cuda_graph
