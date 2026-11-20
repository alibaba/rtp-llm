#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskIOExecutor.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

// Bounded barriers keep a regression from hanging teardown indefinitely.
class Signal {
public:
    void arrive() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_all();
    }

    bool wait(size_t count = 1) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, 5s, [&] { return count_ >= count; });
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  count_{0};
};

struct Completion {
    DiskIOExecutor::Callback callback() {
        return [this](DiskBlockIOStatus status) {
            ++calls;
            promise.set_value(status);
        };
    }

    DiskBlockIOStatus get() {
        if (future.wait_for(5s) != std::future_status::ready) {
            ADD_FAILURE() << "disk completion timed out";
            return DiskBlockIOStatus::IO_ERROR;
        }
        return future.get();
    }

    std::atomic<size_t>             calls{0};
    std::promise<DiskBlockIOStatus> promise;
    std::future<DiskBlockIOStatus>  future{promise.get_future()};
};

class FakeDiskIO: public DiskBlockIO {
public:
    ~FakeDiskIO() override {
        close();
    }

    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override {
        return DiskBlockIOStatus::OK;
    }
    void        close() override {}
    std::string debugString() const override {
        return "FakeDiskIO";
    }

    DiskBlockIOStatus read(uint64_t offset, void*, size_t) override {
        ++reads;
        return operation ? operation(false, offset) : DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(uint64_t offset, const void*, size_t) override {
        ++writes;
        return operation ? operation(true, offset) : DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& items) override {
        for (const auto& item : items) {
            const auto status = read(item.offset, item.buffer, item.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& items) override {
        for (const auto& item : items) {
            const auto status = write(item.offset, item.buffer, item.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }

    std::function<DiskBlockIOStatus(bool, uint64_t)> operation;
    std::atomic<size_t>                              reads{0};
    std::atomic<size_t>                              writes{0};
};

DiskIOExecutorConfig smallConfig() {
    DiskIOExecutorConfig config;
    config.read_thread_count  = 2;
    config.write_thread_count = 2;
    return config;
}

TEST(DiskIOExecutorConfigTest, ConvertsBatchBudgetsAndCapsFanoutAtDirectionWorkerCount) {
    DiskIOExecutorConfig config;
    ASSERT_TRUE(config.setQueueSizesFromBatchCapacity(20000, 10000, 8));
    EXPECT_EQ(config.read_queue_size, 160000u);
    EXPECT_EQ(config.write_queue_size, 10000u);
    ASSERT_TRUE(config.setQueueSizesFromBatchCapacity(20000, 10000, 64));
    EXPECT_EQ(config.read_queue_size, 320000u);
    EXPECT_EQ(config.write_queue_size, 10000u);
    config.read_thread_count  = 3;
    config.write_thread_count = 5;
    ASSERT_TRUE(config.setQueueSizesFromBatchCapacity(14, 7, 9));
    EXPECT_EQ(config.read_queue_size, 42u);
    EXPECT_EQ(config.write_queue_size, 7u);
}

TEST(DiskIOExecutorConfigTest, RejectsInvalidOrOverflowingBudgetsWithoutPartialUpdates) {
    DiskIOExecutorConfig config;
    const auto           reads  = config.read_queue_size;
    const auto           writes = config.write_queue_size;
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(0, 1, 1));
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(1, 0, 1));
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(1, 1, 0));
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(std::numeric_limits<size_t>::max(), 1, 16));
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(1, std::numeric_limits<size_t>::max(), 16));
    EXPECT_EQ(config.read_queue_size, reads);
    EXPECT_EQ(config.write_queue_size, writes);
    config.read_thread_count = 0;
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(1, 1, 1));
}

TEST(DiskIOExecutorConfigTest, EnforcesAutilCircularQueueCapacityLimit) {
    auto config               = smallConfig();
    config.read_thread_count  = 1;
    config.write_thread_count = 1;
    const size_t limit        = std::numeric_limits<int32_t>::max() - 1;
    ASSERT_TRUE(config.setQueueSizesFromBatchCapacity(limit, 1, 1));
    EXPECT_EQ(config.read_queue_size, limit);
    EXPECT_FALSE(config.setQueueSizesFromBatchCapacity(limit + 1, 1, 1));
    EXPECT_EQ(config.read_queue_size, limit);
    EXPECT_EQ(config.write_queue_size, 1u);
}

TEST(DiskIOExecutorTest, ReadBatchFansOutAndWriteBatchUsesOneWorkerInOffsetOrder) {
    for (bool write : {false, true}) {
        Signal                started;
        Signal                release;
        Completion            done, queued;
        FakeDiskIO            io;
        DiskIOExecutor        executor(smallConfig());
        executor.start();
        std::mutex            mutex;
        std::vector<uint64_t> offsets;
        std::vector<unsigned char> buffer(64, 0xa5);
        const auto                 submitting_thread = std::this_thread::get_id();
        io.operation = [&](bool actual_write, uint64_t offset) {
            EXPECT_EQ(actual_write, write);
            EXPECT_NE(std::this_thread::get_id(), submitting_thread);
            if (write) {
                EXPECT_EQ(std::vector<unsigned char>(buffer.begin(), buffer.begin() + 36),
                          std::vector<unsigned char>(36, 0xa5));
                EXPECT_EQ(std::vector<unsigned char>(buffer.begin() + 36, buffer.end()),
                          std::vector<unsigned char>(28, 0xa5));
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                offsets.push_back(offset);
            }
            started.arrive();
            EXPECT_TRUE(release.wait());
            return DiskBlockIOStatus::OK;
        };
        ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
        const auto status =
            write ? executor.asyncWrite(io, {{1, buffer.data(), 64}, {0, buffer.data(), 64}}, done.callback()) :
                    executor.asyncRead(io, {{0, buffer.data(), 1}, {1, buffer.data(), 1}}, done.callback());
        EXPECT_EQ(status, DiskBlockIOStatus::OK);
        EXPECT_TRUE(started.wait(write ? 1 : 2));
        if (write) {
            // A second worker stays available while this batch's first write is blocked.
            EXPECT_EQ(executor.asyncWrite(io, {}, queued.callback()), DiskBlockIOStatus::OK);
            EXPECT_EQ(queued.get(), DiskBlockIOStatus::OK);
        }
        EXPECT_EQ(done.calls, 0u);
        release.arrive();
        EXPECT_EQ(done.get(), DiskBlockIOStatus::OK);
        executor.stop();
        io.close();
        EXPECT_EQ(done.calls, 1u);
        if (write) {
            EXPECT_EQ(offsets, (std::vector<uint64_t>{0, 1}));
        }
    }
}

TEST(DiskIOExecutorTest, PartialEnqueueOwnsWholeBatchAndFullQueueRejectsWithoutCallback) {
    Signal     started;
    Signal     release;
    Completion first, second, queued, rejected, write;
    auto       config      = smallConfig();
    config.read_queue_size = 1;
    FakeDiskIO                       io;
    DiskIOExecutor                   executor(config);
    executor.start();
    std::vector<std::atomic<size_t>> visits(10);
    for (auto& visit : visits) {
        visit = 0;
    }
    io.operation = [&](bool writing, uint64_t offset) {
        if (!writing) {
            ++visits.at(offset);
            if (offset < 2) {
                started.arrive();
                EXPECT_TRUE(release.wait());
            }
        }
        return DiskBlockIOStatus::OK;
    };
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    char buffer = 0;
    ASSERT_EQ(executor.asyncRead(io, {{0, &buffer, 1}}, first.callback()), DiskBlockIOStatus::OK);
    EXPECT_TRUE(started.wait());
    ASSERT_EQ(executor.asyncRead(io, {{1, &buffer, 1}}, second.callback()), DiskBlockIOStatus::OK);
    EXPECT_TRUE(started.wait(2));
    std::vector<DiskRead> batch;
    for (size_t i = 2; i < visits.size(); ++i) {
        batch.push_back({i, &buffer, 1});
    }
    // Both workers are blocked: exactly one participant fits in the queue.
    EXPECT_EQ(executor.asyncRead(io, std::move(batch), queued.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(executor.asyncRead(io, {{0, &buffer, 1}}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(io.reads, 2u);
    // Saturated reads cannot prevent writes from completing.
    EXPECT_EQ(executor.asyncWrite(io, {{0, &buffer, 1}}, write.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(write.get(), DiskBlockIOStatus::OK);
    release.arrive();
    EXPECT_EQ(first.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(second.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(queued.get(), DiskBlockIOStatus::OK);
    executor.stop();
    io.close();
    EXPECT_EQ(rejected.calls, 0u);
    EXPECT_EQ(queued.calls, 1u);
    for (const auto& visit : visits) {
        EXPECT_EQ(visit, 1u);
    }
}

TEST(DiskIOExecutorTest, FailureWaitsForOtherIssuedIOAndKeepsCapturedOwnerAlive) {
    Signal             started, fail, failure_returned, release_other;
    Completion         done, failure_processed;
    FakeDiskIO         io;
    DiskIOExecutor     executor(smallConfig());
    executor.start();
    auto               owner = std::make_shared<int>(42);
    std::weak_ptr<int> weak  = owner;
    io.operation             = [&](bool, uint64_t offset) {
        started.arrive();
        if (offset == 0) {
            EXPECT_TRUE(fail.wait());
            failure_returned.arrive();
            return DiskBlockIOStatus::PARTIAL_FAILURE;
        }
        EXPECT_TRUE(release_other.wait());
        EXPECT_FALSE(weak.expired());
        return DiskBlockIOStatus::OK;
    };
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    char buffer = 0;
    ASSERT_EQ(executor.asyncRead(io,
                                 {{0, &buffer, 1}, {1, &buffer, 1}, {2, &buffer, 1}},
                                 [owner, cb = done.callback()](DiskBlockIOStatus status) { cb(status); }),
              DiskBlockIOStatus::OK);
    owner.reset();
    EXPECT_TRUE(started.wait(2));
    // An empty queued batch can complete only after the failing participant
    // records its status and exits. Keep the other participant blocked until it
    // does, so checking that block 2 remains unissued is deterministic.
    EXPECT_EQ(executor.asyncRead(io, {}, failure_processed.callback()), DiskBlockIOStatus::OK);
    fail.arrive();
    EXPECT_TRUE(failure_returned.wait());
    EXPECT_EQ(failure_processed.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(done.calls, 0u);
    EXPECT_FALSE(weak.expired());
    release_other.arrive();
    EXPECT_EQ(done.get(), DiskBlockIOStatus::PARTIAL_FAILURE);
    executor.stop();
    io.close();
    EXPECT_EQ(io.reads, 2u);
    EXPECT_EQ(done.calls, 1u);
    EXPECT_TRUE(weak.expired());
}

TEST(DiskIOExecutorTest, IOAndCallbackExceptionsDoNotKillWorkersOrRepeatCompletion) {
    Completion failure, success;
    Signal     thrown_callback;
    auto       config        = smallConfig();
    config.read_thread_count = 1;
    FakeDiskIO     io;
    DiskIOExecutor executor(config);
    executor.start();
    io.operation = [](bool, uint64_t offset) {
        if (offset == 0) {
            throw std::runtime_error("injected I/O failure");
        }
        if (offset == 1) {
            throw 42;
        }
        return DiskBlockIOStatus::OK;
    };
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    char buffer = 0;
    EXPECT_EQ(executor.asyncRead(io, {{0, &buffer, 1}}, failure.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(failure.get(), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(executor.asyncRead(io,
                                 {{1, &buffer, 1}},
                                 [&](DiskBlockIOStatus status) {
                                     EXPECT_EQ(status, DiskBlockIOStatus::IO_ERROR);
                                     thrown_callback.arrive();
                                     throw std::runtime_error("injected callback failure");
                                 }),
              DiskBlockIOStatus::OK);
    EXPECT_TRUE(thrown_callback.wait());
    EXPECT_EQ(executor.asyncRead(io, {{2, &buffer, 1}}, success.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(success.get(), DiskBlockIOStatus::OK);
    executor.stop();
    io.close();
    EXPECT_EQ(failure.calls, 1u);
    EXPECT_EQ(success.calls, 1u);
}

TEST(DiskIOExecutorTest, ExplicitStartIsRequiredAndStoppedExecutorCannotRestart) {
    Completion     rejected, read, write;
    FakeDiskIO     io;
    DiskIOExecutor executor(smallConfig());
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    char buffer = 0;
    EXPECT_EQ(executor.asyncRead(io, {{0, &buffer, 1}}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(executor.asyncWrite(io, {{0, &buffer, 1}}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(rejected.calls, 0u);
    EXPECT_EQ(io.reads, 0u);
    ASSERT_NO_THROW(executor.start());
    EXPECT_NO_THROW(executor.start());
    EXPECT_EQ(executor.asyncRead(io, {}, {}), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(executor.asyncRead(io, {{0, &buffer, 1}}, read.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(executor.asyncWrite(io, {{0, &buffer, 1}}, write.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(read.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(write.get(), DiskBlockIOStatus::OK);
    executor.stop();
    EXPECT_EQ(executor.asyncWrite(io, {{0, &buffer, 1}}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(rejected.calls, 0u);
    EXPECT_EQ(io.reads, 1u);
    EXPECT_EQ(io.writes, 1u);
    EXPECT_THROW(executor.start(), std::logic_error);
}

TEST(DiskIOExecutorTest, EmptyBatchAndResubmissionCallbacksAlwaysRunOnWorkers) {
    Completion     read, write;
    FakeDiskIO     io;
    DiskIOExecutor executor(smallConfig());
    executor.start();
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    const auto submitting_thread = std::this_thread::get_id();
    EXPECT_EQ(executor.asyncRead(io,
                                 {},
                                 [&](DiskBlockIOStatus status) {
                                     EXPECT_NE(std::this_thread::get_id(), submitting_thread);
                                     EXPECT_EQ(executor.asyncWrite(io, {}, write.callback()), DiskBlockIOStatus::OK);
                                     read.callback()(status);
                                 }),
              DiskBlockIOStatus::OK);
    EXPECT_EQ(read.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(write.get(), DiskBlockIOStatus::OK);
    executor.stop();
    io.close();
    EXPECT_EQ(io.reads, 0u);
    EXPECT_EQ(io.writes, 0u);
}

TEST(DiskIOExecutorTest, AcceptsUpperLayerBatchesBeyondFormerByteAndBlockBudgets) {
    Completion     read, write;
    FakeDiskIO     io;
    DiskIOExecutor executor(smallConfig());
    executor.start();
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    char buffer = 0;
    // The fake backend does not dereference buffers; no large allocation is needed.
    EXPECT_EQ(executor.asyncRead(io, std::vector<DiskRead>(4097, {0, &buffer, 1}), read.callback()),
              DiskBlockIOStatus::OK);
    EXPECT_EQ(executor.asyncWrite(io, {{0, &buffer, 320 * 1024 * 1024}}, write.callback()), DiskBlockIOStatus::OK);
    EXPECT_EQ(read.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(write.get(), DiskBlockIOStatus::OK);
    executor.stop();
    EXPECT_EQ(io.reads, 4097u);
    EXPECT_EQ(io.writes, 1u);
}

TEST(DiskIOExecutorTest, StopDrainsQueuedWorkAndWaitsForCallbackReturn) {
    Signal     entered_callback, release_callback, closing;
    Completion queued;
    auto       config        = smallConfig();
    config.read_thread_count = 1;
    FakeDiskIO     io;
    DiskIOExecutor executor(config);
    executor.start();
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    char buffer = 0;
    ASSERT_EQ(executor.asyncRead(io,
                                 {{0, &buffer, 1}},
                                 [&](DiskBlockIOStatus) {
                                     entered_callback.arrive();
                                     EXPECT_TRUE(release_callback.wait());
                                 }),
              DiskBlockIOStatus::OK);
    EXPECT_TRUE(entered_callback.wait());
    EXPECT_EQ(executor.asyncRead(io, {{1, &buffer, 1}}, queued.callback()), DiskBlockIOStatus::OK);
    auto close = std::async(std::launch::async, [&] {
        closing.arrive();
        executor.stop();
        io.close();
    });
    EXPECT_TRUE(closing.wait());
    EXPECT_EQ(close.wait_for(20ms), std::future_status::timeout);
    EXPECT_EQ(queued.calls, 0u);
    release_callback.arrive();
    close.get();
    EXPECT_EQ(queued.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(io.reads, 2u);
    EXPECT_EQ(queued.calls, 1u);
    executor.stop();
    io.close();
}

TEST(DiskIOExecutorTest, SubmissionsRacingStopHaveExactlyOneCompletionPerAcceptance) {
    std::atomic<size_t> accepted{0}, completed{0};
    Signal              start, submitted;
    FakeDiskIO          io;
    DiskIOExecutor      executor(smallConfig());
    executor.start();
    ASSERT_EQ(io.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    std::vector<std::thread> submitters;
    for (size_t thread = 0; thread < 4; ++thread) {
        submitters.emplace_back([&] {
            EXPECT_TRUE(start.wait());
            for (size_t i = 0; i < 200; ++i) {
                if (executor.asyncRead(io,
                                       {},
                                       [&](DiskBlockIOStatus status) {
                                           EXPECT_EQ(status, DiskBlockIOStatus::OK);
                                           ++completed;
                                       })
                    == DiskBlockIOStatus::OK) {
                    ++accepted;
                }
                if (i == 0) {
                    submitted.arrive();
                }
            }
        });
    }
    start.arrive();
    EXPECT_TRUE(submitted.wait(4));
    executor.stop();
    io.close();
    for (auto& submitter : submitters) {
        submitter.join();
    }
    EXPECT_EQ(accepted, completed);
    EXPECT_GT(accepted, 0u);
}

TEST(DiskIOExecutorTest, IndependentFilesShareDirectionWorkersAndQueueCapacity) {
    for (bool write : {false, true}) {
        auto config              = smallConfig();
        config.read_thread_count = config.write_thread_count = 1;
        config.read_queue_size = config.write_queue_size = 1;
        DiskIOExecutor executor(config);
        executor.start();
        FakeDiskIO     full, swa;

        ASSERT_EQ(full.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
        ASSERT_EQ(swa.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
        Signal     started, release;
        Completion first, queued, rejected, opposite;
        full.operation = [&](bool writing, uint64_t) {
            if (writing == write) {
                started.arrive();
                EXPECT_TRUE(release.wait());
            }
            return DiskBlockIOStatus::OK;
        };
        char       buffer = 0;
        const auto submit = [&](FakeDiskIO& io, Completion& done, bool writing) {
            return writing ? executor.asyncWrite(io, {{0, &buffer, 1}}, done.callback()) :
                             executor.asyncRead(io, {{0, &buffer, 1}}, done.callback());
        };
        ASSERT_EQ(submit(full, first, write), DiskBlockIOStatus::OK);
        ASSERT_TRUE(started.wait());
        EXPECT_EQ(submit(swa, queued, write), DiskBlockIOStatus::OK);
        // A second file gets neither another worker nor another queue budget.
        EXPECT_EQ(submit(full, rejected, write), DiskBlockIOStatus::IO_ERROR);
        EXPECT_EQ(swa.reads + swa.writes, 0u);
        EXPECT_EQ(submit(full, opposite, !write), DiskBlockIOStatus::OK);
        EXPECT_EQ(opposite.get(), DiskBlockIOStatus::OK);
        release.arrive();
        EXPECT_EQ(first.get(), DiskBlockIOStatus::OK);
        EXPECT_EQ(queued.get(), DiskBlockIOStatus::OK);
        executor.stop();
        full.close();
        swa.close();
        EXPECT_EQ(rejected.calls, 0u);
        EXPECT_EQ(first.calls, 1u);
        EXPECT_EQ(queued.calls, 1u);
    }
}

TEST(DiskIOExecutorTest, SharedExecutorStopDrainsCallbacksAndRejectsEveryFile) {
    DiskIOExecutor executor(smallConfig());
    executor.start();
    FakeDiskIO     full, swa;

    ASSERT_EQ(full.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    ASSERT_EQ(swa.openAndPreallocate("", 0, true), DiskBlockIOStatus::OK);
    Signal entered, release;
    ASSERT_EQ(executor.asyncRead(full,
                                 {},
                                 [&](DiskBlockIOStatus) {
                                     entered.arrive();
                                     EXPECT_TRUE(release.wait());
                                 }),
              DiskBlockIOStatus::OK);
    ASSERT_TRUE(entered.wait());
    auto stopping = std::async(std::launch::async, [&] { executor.stop(); });
    EXPECT_EQ(stopping.wait_for(20ms), std::future_status::timeout);
    Completion rejected;
    EXPECT_EQ(executor.asyncWrite(swa, {}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    release.arrive();
    stopping.get();
    EXPECT_EQ(executor.asyncRead(full, {}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    full.close();
    swa.close();
    EXPECT_EQ(rejected.calls, 0u);
}

TEST(DiskIOExecutorTest, InvalidExecutorConfigDoesNotAffectSynchronousFileIO) {
    auto config             = smallConfig();
    config.write_queue_size = 0;
    BlockTreePosixDiskBlockIO io;
    DiskIOExecutor            executor(config);
    EXPECT_THROW(executor.start(), std::invalid_argument);
    const auto                path = ::testing::TempDir() + "/disk_executor_invalid_config.bin";
    ::remove(path.c_str());
    ASSERT_EQ(io.openAndPreallocate(path, 4096, true), DiskBlockIOStatus::OK);
    char buffer = 'x';
    EXPECT_EQ(io.write(0, &buffer, 1), DiskBlockIOStatus::OK);
    Completion rejected;
    EXPECT_EQ(executor.asyncWrite(io, {{0, &buffer, 1}}, rejected.callback()), DiskBlockIOStatus::IO_ERROR);
    EXPECT_EQ(rejected.calls, 0u);
    executor.stop();
    io.close();
    ::remove(path.c_str());
}

class PosixDiskIOAsyncTest: public ::testing::TestWithParam<bool> {};

TEST_P(PosixDiskIOAsyncTest, RoundTripValidationDrainAndReopen) {
    Completion                write, read, invalid;
    BlockTreePosixDiskBlockIO io;
    DiskIOExecutor            executor(smallConfig());
    executor.start();
    const auto path = ::testing::TempDir() + (GetParam() ? "/disk_async_buffered.bin" : "/disk_async_direct.bin");
    ::remove(path.c_str());
    const auto opened = io.openAndPreallocate(path, 8192, GetParam());
    if (!GetParam() && opened != DiskBlockIOStatus::OK) {
        ::remove(path.c_str());
        GTEST_SKIP() << "filesystem does not support direct I/O";
    }
    ASSERT_EQ(opened, DiskBlockIOStatus::OK);
    alignas(4096) char source[8192], destination[8192] = {};
    for (size_t i = 0; i < sizeof(source); ++i) {
        source[i] = static_cast<char>(i % 251);
    }
    ASSERT_EQ(executor.asyncWrite(io, {{4096, source + 4096, 4096}, {0, source, 4096}}, write.callback()),
              DiskBlockIOStatus::OK);
    EXPECT_EQ(write.get(), DiskBlockIOStatus::OK);
    ASSERT_EQ(executor.asyncRead(io, {{0, destination, 4096}, {4096, destination + 4096, 4096}}, read.callback()),
              DiskBlockIOStatus::OK);
    // The caller drains the executor before closing or reopening the synchronous file.
    executor.stop();
    io.close();
    EXPECT_EQ(read.get(), DiskBlockIOStatus::OK);
    EXPECT_EQ(std::vector<char>(source, source + 8192), std::vector<char>(destination, destination + 8192));
    ::remove(path.c_str());
    ASSERT_EQ(io.openAndPreallocate(path, 8192, GetParam()), DiskBlockIOStatus::OK);
    DiskIOExecutor reopened(smallConfig());
    reopened.start();
    EXPECT_EQ(reopened.asyncRead(io, {{GetParam() ? 8192u : 1u, destination, 4096}}, invalid.callback()),
              DiskBlockIOStatus::OK);
    EXPECT_EQ(invalid.get(), GetParam() ? DiskBlockIOStatus::INVALID_SIZE : DiskBlockIOStatus::ALIGNMENT_ERROR);
    reopened.stop();
    io.close();
    EXPECT_EQ(write.calls, 1u);
    EXPECT_EQ(read.calls, 1u);
    ::remove(path.c_str());
}

INSTANTIATE_TEST_SUITE_P(IOModes, PosixDiskIOAsyncTest, ::testing::Values(true, false));

}  // namespace
}  // namespace rtp_llm
