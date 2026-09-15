#include "rtp_llm/cpp/engine_base/sleep/SleepRoundFence.h"
#include <algorithm>
#include <array>
#include <future>
#include <thread>
#include "gtest/gtest.h"

namespace rtp_llm {
using namespace std::chrono_literals;

TEST(SleepRoundFenceTest, FreezeCountsAdmittedNotCompletedWork) {
    SleepRoundFence ahead, behind;
    EXPECT_EQ(ahead.next().round, 1u);
    // Round 1 may be waiting for the peer: freeze MUST still return immediately.
    EXPECT_EQ(ahead.freeze(), 1u);
    EXPECT_EQ(behind.freeze(), 0u);
    ASSERT_TRUE(ahead.setTarget(1));
    ASSERT_TRUE(behind.setTarget(1));
    const auto catchup = behind.next();
    EXPECT_EQ(catchup.action, SleepRoundFence::Action::RUN);
    EXPECT_EQ(catchup.round, 1u);
    for (auto* fence : {&ahead, &behind}) {
        const auto permit = fence->next();
        EXPECT_EQ(permit.action, SleepRoundFence::Action::QUIESCE);
        EXPECT_FALSE(fence->wait(1ms));
        fence->finishQuiesce(permit.generation);
        EXPECT_TRUE(fence->wait(1ms));
        EXPECT_TRUE(fence->resume());
        EXPECT_EQ(fence->next().round, 2u);
    }
}

TEST(SleepRoundFenceTest, LateTpWorkerTargetDoesNotNeedAnotherInputBroadcast) {
    SleepRoundFence root, worker;
    // Both ranks have admitted the same TP input broadcast. They need not
    // receive either control RPC while that broadcast/forward is running.
    EXPECT_EQ(root.next().round, 1u);
    EXPECT_EQ(worker.next().round, 1u);
    EXPECT_EQ(root.freeze(), 1u);
    EXPECT_EQ(worker.freeze(), 1u);
    ASSERT_TRUE(root.setTarget(1));
    const auto root_end = root.next();
    ASSERT_EQ(root_end.action, SleepRoundFence::Action::QUIESCE);
    root.finishQuiesce(root_end.generation);
    EXPECT_TRUE(root.wait(1ms));

    // The root has already parked, but the worker has not received its target.
    // Unlike rank-local pause, it cannot start a second TP input broadcast.
    auto worker_next = std::async(std::launch::async, [&] { return worker.next(); });
    EXPECT_EQ(worker_next.wait_for(20ms), std::future_status::timeout);
    EXPECT_TRUE(worker.setTarget(1));
    const auto ready = worker_next.wait_for(1s);
    EXPECT_EQ(ready, std::future_status::ready);
    if (ready != std::future_status::ready) {
        worker.stop();
    }
    const auto worker_end = worker_next.get();
    EXPECT_EQ(worker_end.action, SleepRoundFence::Action::QUIESCE);
    EXPECT_EQ(worker_end.round, 1u);
    worker.finishQuiesce(worker_end.generation);
    EXPECT_TRUE(worker.wait(1ms));
    EXPECT_TRUE(root.resume());
    EXPECT_TRUE(worker.resume());
    EXPECT_EQ(root.next().round, 2u);
    EXPECT_EQ(worker.next().round, 2u);
}

TEST(SleepRoundFenceTest, NoForwardBeforeTargetOrPastTarget) {
    SleepRoundFence fence;
    EXPECT_EQ(fence.freeze(), 0u);
    auto pending = std::async(std::launch::async, [&] { return fence.next(); });
    EXPECT_EQ(pending.wait_for(20ms), std::future_status::timeout);
    EXPECT_TRUE(fence.setTarget(1));
    const auto ready = pending.wait_for(1s);
    EXPECT_EQ(ready, std::future_status::ready);
    if (ready != std::future_status::ready) {
        fence.stop();
    }
    EXPECT_EQ(pending.get().action, SleepRoundFence::Action::RUN);
    const auto end = fence.next();
    EXPECT_EQ(end.action, SleepRoundFence::Action::QUIESCE);
    fence.finishQuiesce(end.generation);
    auto parked = std::async(std::launch::async, [&] { return fence.next(); });
    EXPECT_EQ(parked.wait_for(20ms), std::future_status::timeout);
    fence.stop();
    EXPECT_EQ(parked.get().action, SleepRoundFence::Action::STOP);
}

TEST(SleepRoundFenceTest, RepeatedFreezeCannotChangeTheSnapshotOrTarget) {
    SleepRoundFence fence;
    EXPECT_FALSE(fence.setTarget(1));
    EXPECT_EQ(fence.next().round, 1u);
    EXPECT_EQ(fence.freeze(), 1u);
    EXPECT_FALSE(fence.setTarget(0));
    EXPECT_FALSE(fence.setTarget(uint64_t{1} << 63));
    EXPECT_TRUE(fence.setTarget(3));
    EXPECT_EQ(fence.next().round, 2u);
    EXPECT_EQ(fence.freeze(), 1u);
    EXPECT_TRUE(fence.setTarget(3));
    EXPECT_FALSE(fence.setTarget(4));
    EXPECT_EQ(fence.next().round, 3u);
    EXPECT_EQ(fence.next().action, SleepRoundFence::Action::QUIESCE);
}

TEST(SleepRoundFenceTest, CancelResumesWithoutWaitingForAPeerOrResettingTickets) {
    SleepRoundFence fence;
    EXPECT_EQ(fence.next().round, 1u);
    fence.freeze();
    auto pending = std::async(std::launch::async, [&] { return fence.next(); });
    EXPECT_EQ(pending.wait_for(20ms), std::future_status::timeout);
    EXPECT_TRUE(fence.resume());
    const auto ready = pending.wait_for(1s);
    EXPECT_EQ(ready, std::future_status::ready);
    if (ready != std::future_status::ready) {
        fence.stop();
    }
    EXPECT_EQ(pending.get().round, 2u);
    EXPECT_EQ(fence.freeze(), 2u);
}

TEST(SleepRoundFenceTest, StaleAcknowledgementCannotQuiesceANewGeneration) {
    SleepRoundFence fence;
    fence.freeze();
    ASSERT_TRUE(fence.setTarget(0));
    const auto old = fence.next();
    fence.finishQuiesce(old.generation);
    ASSERT_TRUE(fence.resume());
    fence.freeze();
    ASSERT_TRUE(fence.setTarget(0));
    const auto current = fence.next();
    fence.finishQuiesce(old.generation);
    EXPECT_FALSE(fence.wait(1ms));
    fence.finishQuiesce(current.generation);
    EXPECT_TRUE(fence.wait(1ms));
}

TEST(SleepRoundFenceTest, StuckOrFailedDeviceDrainCannotReportRollbackSuccess) {
    SleepRoundFence fence;
    fence.freeze();
    ASSERT_TRUE(fence.setTarget(0));
    const auto permit = fence.next();
    EXPECT_FALSE(fence.resume());
    EXPECT_FALSE(fence.wait(1ms));
    fence.finishQuiesce(permit.generation, "device drain failed");
    EXPECT_FALSE(fence.wait(1ms));
    EXPECT_FALSE(fence.resume());
    fence.stop();
    EXPECT_EQ(fence.next().action, SleepRoundFence::Action::STOP);
}

TEST(SleepRoundFenceTest, FreezeRacesAdmissionWithoutAdmittingAnUncountedRound) {
    for (int iteration = 0; iteration < 200; ++iteration) {
        SleepRoundFence       fence;
        std::atomic<uint64_t> completed{0};
        auto                  worker = std::async(std::launch::async, [&] {
            for (;;) {
                auto permit = fence.next();
                if (permit.action == SleepRoundFence::Action::STOP) {
                    return;
                }
                if (permit.action == SleepRoundFence::Action::QUIESCE) {
                    fence.finishQuiesce(permit.generation);
                } else {
                    completed.store(permit.round, std::memory_order_release);
                }
            }
        });
        const auto            frozen = fence.freeze();
        EXPECT_TRUE(fence.setTarget(frozen));
        EXPECT_TRUE(fence.wait(1s));
        EXPECT_EQ(completed.load(std::memory_order_acquire), frozen);
        fence.stop();
        worker.get();
    }
}

TEST(SleepRoundFenceTest, FourRanksCatchUpWithSkewedControlAndMultipleForwardPhases) {
    // Model collectives may already be waiting for another rank when freeze
    // arrives. A host barrier models that dependency without needing a GPU.
    struct ForwardBarrier {
        std::mutex              mutex;
        std::condition_variable cv;
        int                     arrived    = 0;
        int                     generation = 0;
        bool                    stopped    = false;
        bool                    wait() {
            std::unique_lock<std::mutex> lock(mutex);
            const int                    old = generation;
            if (++arrived == 4) {
                arrived = 0;
                ++generation;
                cv.notify_all();
            } else {
                cv.wait(lock, [&] { return stopped || generation != old; });
            }
            return !stopped;
        }
        void stop() {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
            cv.notify_all();
        }
    } barrier;
    std::array<SleepRoundFence, 4>       fences;
    std::array<std::atomic<uint64_t>, 4> completed{};
    std::array<std::future<void>, 4>     workers;
    for (size_t rank = 0; rank < fences.size(); ++rank) {
        workers[rank] = std::async(std::launch::async, [&, rank] {
            for (;;) {
                const auto permit = fences[rank].next();
                if (permit.action == SleepRoundFence::Action::STOP) {
                    return;
                }
                if (permit.action == SleepRoundFence::Action::QUIESCE) {
                    fences[rank].finishQuiesce(permit.generation);
                    continue;
                }
                // Like fixed MTP draft/target phases: every admitted round
                // must complete the full sequence, including fake rounds.
                for (int phase = 0; phase < 5; ++phase) {
                    if (!barrier.wait()) {
                        return;
                    }
                }
                completed[rank].store(permit.round);
            }
        });
    }
    for (int cycle = 0; cycle < 3; ++cycle) {
        std::this_thread::sleep_for(2ms);
        std::array<uint64_t, 4> snapshots;
        for (size_t rank = 0; rank < fences.size(); ++rank) {
            snapshots[rank] = fences[rank].freeze();
            std::this_thread::sleep_for(1ms);
        }
        const auto target = *std::max_element(snapshots.begin(), snapshots.end());
        for (auto& fence : fences) {
            EXPECT_TRUE(fence.setTarget(target));
            std::this_thread::sleep_for(1ms);
        }
        for (size_t rank = 0; rank < fences.size(); ++rank) {
            EXPECT_TRUE(fences[rank].wait(1s));
            EXPECT_EQ(completed[rank].load(), target);
        }
        for (auto& fence : fences) {
            EXPECT_TRUE(fence.resume());
        }
    }
    // Always unblock both the fence and mock collectives even on an EXPECT
    // failure, so an unsuccessful test cannot hang in a future destructor.
    for (auto& fence : fences) {
        fence.stop();
    }
    barrier.stop();
    for (auto& worker : workers) {
        worker.get();
    }
}

}  // namespace rtp_llm
