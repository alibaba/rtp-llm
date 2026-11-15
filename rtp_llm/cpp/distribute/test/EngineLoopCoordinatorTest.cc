#include "rtp_llm/cpp/distribute/EngineLoopCoordinator.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

class Collective {
public:
    explicit Collective(int size): size_(size) {}
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const int                    generation = generation_;
        if (++arrivals_ == size_) {
            arrivals_ = 0;
            ++generation_;
            cv_.notify_all();
            return true;
        }
        if (!cv_.wait_for(lock, 500ms, [&] { return broken_ || generation_ != generation; })) {
            broken_ = true;
            cv_.notify_all();
        }
        return !broken_;
    }

private:
    int                     size_, arrivals_ = 0, generation_ = 0;
    bool                    broken_ = false;
    std::mutex              mutex_;
    std::condition_variable cv_;
};

class EngineLoopCoordinatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        char  path[]  = "/tmp/rtp_engine_stop_XXXXXX";
        auto* created = mkdtemp(path);
        ASSERT_NE(created, nullptr);
        directory_ = created;
    }
    void TearDown() override {
        unlink((directory_ + "/state").c_str());
        unlink((directory_ + "/link").c_str());
        rmdir(directory_.c_str());
    }
    void runCohort(bool coordinated) {
        constexpr int                                            size = 8;
        Collective                                               collective(size);
        std::array<std::atomic<bool>, size>                      running, drained;
        std::array<std::atomic<uint64_t>, size>                  progress;
        std::array<uint64_t, size>                               stopped{};
        std::array<std::unique_ptr<EngineLoopCoordinator>, size> coordinators;
        std::array<std::thread, size>                            threads;
        std::atomic<int>                                         failures{0}, released_early{0};
        for (int rank = 0; rank < size; ++rank) {
            running[rank]  = true;
            drained[rank]  = false;
            progress[rank] = 0;
            if (coordinated) {
                coordinators[rank] = std::make_unique<EngineLoopCoordinator>(directory_ + "/state", rank, size, 5s);
            }
        }
        for (int rank = 0; rank < size; ++rank) {
            threads[rank] = std::thread([&, rank] {
                try {
                    while (coordinated ? coordinators[rank]->beforeStep(running[rank]) : running[rank].load()) {
                        if (!collective.wait()) {
                            throw std::runtime_error("collective participant exited early");
                        }
                        if (rank == 3) {
                            std::this_thread::sleep_for(100us);
                        }
                        if (coordinated) {
                            coordinators[rank]->afterStep();
                        }
                        ++progress[rank];
                    }
                    stopped[rank] = progress[rank].load();
                    if (rank == 2) {
                        std::this_thread::sleep_for(10ms);
                    }
                    drained[rank] = true;
                    if (coordinated) {
                        coordinators[rank]->waitForGpuDrain();
                    }
                    for (const auto& ready : drained) {
                        if (!ready.load()) {
                            ++released_early;
                            break;
                        }
                    }
                } catch (const std::exception&) {
                    ++failures;
                }
            });
        }
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (progress[0].load() < 10 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(1ms);
        }
        running[5] = false;
        if (coordinated) {
            coordinators[5]->requestStop();
        }
        // Like independent backend signal handlers, the other ranks may request
        // stop after a peer has already reached the next model round.
        std::this_thread::sleep_for(30ms);
        for (int rank = 0; rank < size; ++rank) {
            running[rank] = false;
            if (coordinated) {
                coordinators[rank]->requestStop();
            }
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (coordinated) {
            EXPECT_EQ(failures.load(), 0);
            EXPECT_EQ(released_early.load(), 0);
            for (auto count : stopped) {
                EXPECT_EQ(count, stopped[0]);
            }
            EXPECT_GT(stopped[0], 0);
            EXPECT_NE(access((directory_ + "/state").c_str(), F_OK), 0);
        } else {
            EXPECT_GT(failures.load(), 0);
            EXPECT_GT(released_early.load(), 0);
        }
    }
    std::string directory_;
};

TEST_F(EngineLoopCoordinatorTest, IndependentStopLosesACollectiveParticipant) {
    runCohort(false);
}

TEST_F(EngineLoopCoordinatorTest, LocalTpAndDpEpGroupsUseTheSameShutdownProtocol) {
    EXPECT_TRUE(EngineLoopCoordinator::supportsLocalGroup(8, 8, 8, 8));  // Prefill TP8/DP1/CP8
    EXPECT_TRUE(EngineLoopCoordinator::supportsLocalGroup(8, 8, 4, 8));  // Decode TP4/DP2
    EXPECT_FALSE(EngineLoopCoordinator::supportsLocalGroup(8, 4, 4, 8));
    EXPECT_FALSE(EngineLoopCoordinator::supportsLocalGroup(8, 8, 4, 4));
    EXPECT_FALSE(EngineLoopCoordinator::supportsLocalGroup(1, 1, 1, 1));
}
TEST_F(EngineLoopCoordinatorTest, StaggeredStopPreservesAllRoundsAndGpuOwners) {
    runCohort(true);
}

TEST_F(EngineLoopCoordinatorTest, MissingPeerHasABoundedFailure) {
    EngineLoopCoordinator coordinator(directory_ + "/state", 0, 2, 30ms);
    std::atomic<bool>     running{false};
    coordinator.requestStop();
    ASSERT_TRUE(coordinator.beforeStep(running));
    coordinator.afterStep();
    EXPECT_THROW(coordinator.beforeStep(running), std::runtime_error);
    EXPECT_THROW(coordinator.waitForGpuDrain(), std::logic_error);
}

TEST_F(EngineLoopCoordinatorTest, WaitingForPeerIngressDrainDoesNotStopModelRounds) {
    EngineLoopCoordinator first(directory_ + "/state", 0, 2, 2s);
    EngineLoopCoordinator last(directory_ + "/state", 1, 2, 2s);
    auto                  first_ready = std::async(std::launch::async, [&]() { first.waitForStopReady(0ms); });
    EXPECT_EQ(first_ready.wait_for(20ms), std::future_status::timeout);
    std::atomic<bool> running{true};
    for (int round = 0; round < 3; ++round) {
        ASSERT_TRUE(first.beforeStep(running));
        ASSERT_TRUE(last.beforeStep(running));
        first.afterStep();
        last.afterStep();
    }
    last.waitForStopReady(0ms);
    EXPECT_EQ(first_ready.wait_for(1s), std::future_status::ready);
    first_ready.get();
    EXPECT_EQ(first.completedSteps(), 3u);
    EXPECT_EQ(last.completedSteps(), 3u);
}

TEST_F(EngineLoopCoordinatorTest, PeerAlreadyAdmittedNextRoundKeepsTheStoppingPeerParticipating) {
    EngineLoopCoordinator fast(directory_ + "/state", 0, 2, 2s);
    EngineLoopCoordinator slow(directory_ + "/state", 1, 2, 2s);
    std::atomic<bool>     fast_running{true}, slow_running{true};
    ASSERT_TRUE(fast.beforeStep(fast_running));
    ASSERT_TRUE(slow.beforeStep(slow_running));
    fast.afterStep();
    slow.afterStep();
    // Rank 0 finished round 2 and is already admitted to round 3, while
    // rank 1 is still retiring round 1. Publish that admission before stop.
    ASSERT_TRUE(fast.beforeStep(fast_running));
    fast.afterStep();
    ASSERT_TRUE(fast.beforeStep(fast_running));
    fast_running = false;
    fast.requestStop();
    ASSERT_TRUE(fast.beforeStep(fast_running));
    slow_running = false;
    slow.requestStop();
    ASSERT_TRUE(slow.beforeStep(slow_running));
    slow.afterStep();
    EXPECT_EQ(slow.completedSteps(), 2u);
    // A local stop-only implementation would stop here and strand rank 0.
    ASSERT_TRUE(slow.beforeStep(slow_running));
    slow.afterStep();
    fast.afterStep();
    bool        fast_done = false, slow_done = false;
    std::thread peer([&]() {
        while (fast.beforeStep(fast_running)) {
            fast.afterStep();
        }
        fast_done = true;
    });
    while (slow.beforeStep(slow_running)) {
        slow.afterStep();
    }
    slow_done = true;
    peer.join();
    EXPECT_TRUE(fast_done);
    EXPECT_TRUE(slow_done);
    EXPECT_EQ(fast.completedSteps(), slow.completedSteps());
    EXPECT_EQ(fast.completedSteps(), 3u);
}

TEST_F(EngineLoopCoordinatorTest, SymlinksAndDuplicateRanksAreRejected) {
    EngineLoopCoordinator coordinator(directory_ + "/state", 0, 2, 30ms);
    EXPECT_THROW(EngineLoopCoordinator(directory_ + "/state", 0, 2), std::runtime_error);
    ASSERT_EQ(symlink((directory_ + "/state").c_str(), (directory_ + "/link").c_str()), 0);
    EXPECT_THROW(EngineLoopCoordinator(directory_ + "/link", 1, 2), std::runtime_error);
}

TEST_F(EngineLoopCoordinatorTest, SeparateProcessesRetainGpuOwnersUntilEveryRankDrains) {
    constexpr int size = 8;
    struct Shared {
        std::atomic<int>  arrivals{0}, generation{0};
        std::atomic<bool> stop[size]{}, drained[size]{};
        std::atomic<int>  progress[size]{};
    };
    void* memory = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(memory, MAP_FAILED);
    auto*                   shared = new (memory) Shared;
    std::array<pid_t, size> children{};
    for (int rank = 0; rank < size; ++rank) {
        children[rank] = fork();
        if (children[rank] == 0) {
            int result = 0;
            try {
                EngineLoopCoordinator coordinator(directory_ + "/state", rank, size, 5s);
                std::atomic<bool>     running{true};
                std::thread           engine([&] {
                    try {
                        while (coordinator.beforeStep(running)) {
                            const auto deadline   = std::chrono::steady_clock::now() + 3s;
                            const int  generation = shared->generation.load();
                            if (shared->arrivals.fetch_add(1) + 1 == size) {
                                shared->arrivals = 0;
                                ++shared->generation;
                            } else {
                                while (shared->generation.load() == generation) {
                                    if (std::chrono::steady_clock::now() >= deadline) {
                                        throw std::runtime_error("process collective lost a rank");
                                    }
                                    std::this_thread::yield();
                                }
                            }
                            coordinator.afterStep();
                            ++shared->progress[rank];
                        }
                    } catch (...) {
                        result = 1;
                    }
                });
                const auto            deadline = std::chrono::steady_clock::now() + 6s;
                while (!shared->stop[rank] && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(1ms);
                }
                running = false;
                coordinator.requestStop();
                engine.join();
                if (rank == 2) {
                    std::this_thread::sleep_for(20ms);
                }
                shared->drained[rank] = true;
                if (!result) {
                    coordinator.waitForGpuDrain();
                    for (auto& drained : shared->drained) {
                        if (!drained.load()) {
                            result = 2;
                        }
                    }
                }
            } catch (...) {
                result = 3;
            }
            _exit(result);
        }
        EXPECT_GT(children[rank], 0);
    }
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (shared->progress[0].load() < 10 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    shared->stop[5] = true;
    std::this_thread::sleep_for(30ms);
    for (auto& stop : shared->stop) {
        stop = true;
    }
    for (auto pid : children) {
        if (pid > 0) {
            int status = 0;
            EXPECT_EQ(waitpid(pid, &status, 0), pid);
            EXPECT_TRUE(WIFEXITED(status));
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
    for (auto& progress : shared->progress) {
        EXPECT_EQ(progress.load(), shared->progress[0].load());
    }
    EXPECT_GT(shared->progress[0].load(), 0);
    shared->~Shared();
    munmap(memory, sizeof(Shared));
}
}  // namespace
}  // namespace rtp_llm
