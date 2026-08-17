#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/DecodeAdmissionController.h"
#include "rtp_llm/cpp/model_rpc/DecodeAdmissionStatus.h"

namespace rtp_llm {

// CPU-only on purpose: the admission gate is a counting semaphore plus a status mapping, so
// these cases need no device and must run on every CI pipeline (and support --runs_per_test
// for the concurrency ones) instead of queueing for a GPU executor.

// Named for what it asserts: the controller is a counting semaphore. Ordering against
// resource allocation is a property of RemoteGenerate, not of this class, and is not
// covered here.
TEST(DecodeAdmissionControllerTest, SemaphoreQueuesAndReleasesSlots) {
    DecodeAdmissionController controller(/*limit=*/2);
    {
        auto outcome = controller.acquire(/*slots=*/2, [] { return false; }, /*timeout_ms=*/100);
        ASSERT_EQ(outcome.result, DecodeAdmissionController::AcquireResult::ACQUIRED);
        ASSERT_TRUE(outcome.lease.holdsSlots());
        EXPECT_EQ(controller.activeSlots(), 2);

        auto queued_outcome = DecodeAdmissionController::AcquireOutcome{};
        std::thread queued(
            [&] { queued_outcome = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/20); });
        queued.join();
        EXPECT_EQ(queued_outcome.result, DecodeAdmissionController::AcquireResult::TIMED_OUT);
        EXPECT_FALSE(queued_outcome.lease.holdsSlots());
        EXPECT_EQ(controller.activeSlots(), 2);
    }

    // The lease released the slots when it left scope.
    EXPECT_EQ(controller.activeSlots(), 0);
    auto reacquired = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
    EXPECT_EQ(reacquired.result, DecodeAdmissionController::AcquireResult::ACQUIRED);
    EXPECT_EQ(controller.activeSlots(), 1);
}

TEST(DecodeAdmissionControllerTest, RejectsCancelledAndOversizedRequests) {
    DecodeAdmissionController controller(/*limit=*/8);
    auto cancelled = controller.acquire(/*slots=*/1, [] { return true; }, /*timeout_ms=*/100);
    EXPECT_EQ(cancelled.result, DecodeAdmissionController::AcquireResult::CANCELLED);
    EXPECT_FALSE(cancelled.lease.holdsSlots());
    // OVERSIZED guards the caller contract: DecodeRpcServer always asks for one slot,
    // so a request wider than the limit means a caller miscounted. Failing fast beats
    // spinning to the deadline on a wait that can never be satisfied.
    auto oversized = controller.acquire(/*slots=*/9, [] { return false; }, /*timeout_ms=*/100);
    EXPECT_EQ(oversized.result, DecodeAdmissionController::AcquireResult::OVERSIZED);
    EXPECT_FALSE(oversized.lease.holdsSlots());
    EXPECT_EQ(controller.activeSlots(), 0);
}

// The server charges one slot per stream so that admission and
// FIFOScheduler::evaluateRunningMemory count the same unit. A beam request must not
// cost num_beams slots, or concurrency would collapse to limit/num_beams.
TEST(DecodeAdmissionControllerTest, OneSlotPerStreamAdmitsUpToLimit) {
    DecodeAdmissionController controller(/*limit=*/8);
    std::vector<DecodeAdmissionLease> leases;
    for (int i = 0; i < 8; ++i) {
        auto outcome = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
        ASSERT_EQ(outcome.result, DecodeAdmissionController::AcquireResult::ACQUIRED)
            << "stream " << i << " should fit within the limit";
        leases.push_back(std::move(outcome.lease));
    }
    EXPECT_EQ(controller.activeSlots(), 8u);
    // The ninth stream waits rather than being rejected outright.
    EXPECT_EQ(controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/20).result,
              DecodeAdmissionController::AcquireResult::TIMED_OUT);
    leases.clear();
    EXPECT_EQ(controller.activeSlots(), 0u);
}

TEST(DecodeAdmissionControllerTest, BlockedWaiterAcquiresAfterRelease) {
    DecodeAdmissionController controller(/*limit=*/2);
    std::atomic<bool>         finished{false};
    auto                      result = DecodeAdmissionController::AcquireResult::TIMED_OUT;
    std::thread               queued;
    {
        auto initial = controller.acquire(/*slots=*/2, [] { return false; }, /*timeout_ms=*/100);
        ASSERT_EQ(initial.result, DecodeAdmissionController::AcquireResult::ACQUIRED);
        queued = std::thread([&] {
            auto outcome = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/1000);
            result       = outcome.result;
            if (outcome.result == DecodeAdmissionController::AcquireResult::ACQUIRED) {
                finished.store(true, std::memory_order_release);
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        EXPECT_FALSE(finished.load(std::memory_order_acquire));
    }
    queued.join();

    EXPECT_EQ(result, DecodeAdmissionController::AcquireResult::ACQUIRED);
    EXPECT_TRUE(finished.load(std::memory_order_acquire));
    // The waiter's lease died with its thread, so nothing stays charged.
    EXPECT_EQ(controller.activeSlots(), 0);
}

TEST(DecodeAdmissionControllerTest, BlockedWaiterObservesCancellation) {
    DecodeAdmissionController controller(/*limit=*/1);
    auto                      initial = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
    ASSERT_EQ(initial.result, DecodeAdmissionController::AcquireResult::ACQUIRED);

    std::atomic<bool> cancelled{false};
    auto              result = DecodeAdmissionController::AcquireResult::ACQUIRED;
    std::thread       queued([&] {
        result = controller
                     .acquire(
                         /*slots=*/1, [&] { return cancelled.load(std::memory_order_acquire); }, /*timeout_ms=*/1000)
                     .result;
    });
    cancelled.store(true, std::memory_order_release);
    queued.join();

    EXPECT_EQ(result, DecodeAdmissionController::AcquireResult::CANCELLED);
    EXPECT_EQ(controller.activeSlots(), 1);
}

// A lease is the only way to charge slots, so a caller cannot return what it never took --
// the hole that made hand-written guards able to drive the counter negative.
TEST(DecodeAdmissionControllerTest, DefaultLeaseOwnsNothingAndMoveTransfersOwnership) {
    DecodeAdmissionController controller(/*limit=*/2);
    {
        DecodeAdmissionLease empty;
        EXPECT_FALSE(empty.holdsSlots());
    }
    EXPECT_EQ(controller.activeSlots(), 0u);

    auto outcome = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
    ASSERT_EQ(outcome.result, DecodeAdmissionController::AcquireResult::ACQUIRED);
    {
        DecodeAdmissionLease moved = std::move(outcome.lease);
        EXPECT_TRUE(moved.holdsSlots());
        EXPECT_FALSE(outcome.lease.holdsSlots());
        EXPECT_EQ(controller.activeSlots(), 1u);
    }
    // Released exactly once: by the object that owned it after the move.
    EXPECT_EQ(controller.activeSlots(), 0u);
}

// The AcquireResult -> grpc::Status mapping is a cross-module contract with
// PrefillRpcServer, so it is asserted directly instead of through the gRPC handler:
// RESOURCE_EXHAUSTED becomes DECODE_MALLOC_FAILED there (PrefillRpcServer.cc:138) and the
// two substrings below make it tear the connection down (:126, :129).
TEST(DecodeAdmissionControllerTest, AcquireResultStatusMappingHoldsCrossModuleContract) {
    struct MappingCase {
        DecodeAdmissionController::AcquireResult result;
        grpc::StatusCode                        expected_code;
    };
    const std::vector<MappingCase> cases = {
        {DecodeAdmissionController::AcquireResult::ACQUIRED, grpc::StatusCode::OK},
        {DecodeAdmissionController::AcquireResult::CANCELLED, grpc::StatusCode::CANCELLED},
        {DecodeAdmissionController::AcquireResult::TIMED_OUT, grpc::StatusCode::DEADLINE_EXCEEDED},
        {DecodeAdmissionController::AcquireResult::OVERSIZED, grpc::StatusCode::RESOURCE_EXHAUSTED},
    };

    for (const auto& mapping_case : cases) {
        const auto status = admissionResultToStatus(mapping_case.result);
        EXPECT_EQ(status.error_code(), mapping_case.expected_code)
            << "unexpected code for result " << static_cast<int>(mapping_case.result);
        const auto message = status.error_message();
        EXPECT_EQ(message.find("Deadline Exceeded"), std::string::npos) << message;
        EXPECT_EQ(message.find("Connection timed out"), std::string::npos) << message;
    }

    // A saturated role is healthy: queueing timeouts must not be reported as decode KV
    // allocation failures.
    EXPECT_NE(admissionResultToStatus(DecodeAdmissionController::AcquireResult::TIMED_OUT).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST(DecodeAdmissionControllerTest, InitialLimitIsHonoredAndShrinkStopsNewAdmissions) {
    DecodeAdmissionController controller(/*limit=*/4);
    EXPECT_EQ(controller.limit(), 4u);

    auto first  = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
    auto second = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
    ASSERT_EQ(first.result, DecodeAdmissionController::AcquireResult::ACQUIRED);
    ASSERT_EQ(second.result, DecodeAdmissionController::AcquireResult::ACQUIRED);

    // Shrinking below the slots already handed out must not retroactively evict live
    // requests, but it must stop admitting new ones until they drain.
    controller.setLimit(1);
    EXPECT_EQ(controller.limit(), 1u);
    EXPECT_EQ(controller.activeSlots(), 2u);
    EXPECT_EQ(controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/20).result,
              DecodeAdmissionController::AcquireResult::TIMED_OUT);

    first  = DecodeAdmissionController::AcquireOutcome{};
    second = DecodeAdmissionController::AcquireOutcome{};
    EXPECT_EQ(controller.activeSlots(), 0u);
    EXPECT_EQ(controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100).result,
              DecodeAdmissionController::AcquireResult::ACQUIRED);
}

TEST(DecodeAdmissionControllerTest, SetLimitGrowWakesBlockedWaiter) {
    DecodeAdmissionController controller(/*limit=*/1);
    auto                      held = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/100);
    ASSERT_EQ(held.result, DecodeAdmissionController::AcquireResult::ACQUIRED);

    auto        result = DecodeAdmissionController::AcquireResult::TIMED_OUT;
    std::thread queued([&] {
        result = controller.acquire(/*slots=*/1, [] { return false; }, /*timeout_ms=*/2000).result;
    });
    // setLimit must notify the waiter; without notify_all it would only notice on its next
    // 50ms cancel poll.
    controller.setLimit(2);
    queued.join();

    EXPECT_EQ(result, DecodeAdmissionController::AcquireResult::ACQUIRED);
    // The waiter's lease is already gone with its thread; only `held` is still charged.
    EXPECT_EQ(controller.activeSlots(), 1u);
}

}  // namespace rtp_llm
