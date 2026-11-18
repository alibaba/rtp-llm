#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;
using namespace block_transfer_engine_test;
constexpr size_t kBlockBytes = 4096;
constexpr size_t kRequestCount = 4096;
constexpr size_t kProducerCount = 16;
constexpr size_t kBudgetCount = 3;
constexpr unsigned char kDiskSeed = 0x25;
constexpr unsigned char kSource = 0x73;
constexpr unsigned char kSuccessor = 0xB6;

// Observe actual POSIX I/O without injected waits, errors, or completion proofs.
class QueueSmokePosixIO final: public DiskBlockIO {
public:
    struct Event {
        uint64_t offset;
        bool write;
        DiskBlockIOStatus status;
    };

    QueueSmokePosixIO() { events_.reserve(kRequestCount); }

    DiskBlockIOStatus openAndPreallocate(const std::string& path, size_t bytes, bool buffered) override {
        EXPECT_FALSE(buffered);
        return io_.openAndPreallocate(path, bytes, buffered);
    }
    void close() override { io_.close(); }
    std::string debugString() const override { return io_.debugString(); }

    DiskBlockIOStatus read(uint64_t offset, void* buffer, size_t bytes) override {
        const auto status = io_.read(offset, buffer, bytes);
        record({offset, false, status});
        return status;
    }
    DiskBlockIOStatus write(uint64_t offset, const void* buffer, size_t bytes) override {
        const auto status = io_.write(offset, buffer, bytes);
        record({offset, true, status});
        return status;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& requests) override {
        for (const auto& request : requests) {
            const auto status = read(request.offset, request.buffer, request.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& requests) override {
        for (const auto& request : requests) {
            const auto status = write(request.offset, request.buffer, request.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }
    std::vector<Event> events() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
    }

private:
    void record(Event event) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }
    BlockTreePosixDiskBlockIO io_;
    mutable std::mutex mutex_;
    std::vector<Event> events_;
};

// runTransfer creates an independent execution trace below the caller's task.
// Each request has one producer and one descriptor, so recording its actual
// child trace needs no shared hot-path lock. All execution delegates to base.
class QueueSmokeEngine final: public PerRankBlockTransferEngine {
public:
    struct Observation {
        uint64_t trace{0};
        size_t calls{0};
    };

    explicit QueueSmokeEngine(const std::vector<GroupSetPtr>& groups):
        PerRankBlockTransferEngine(groups, true, DeviceHostCopyOptions{}, 1, 1, 1, kRequestCount,
                                   nullptr, 0, 1, true, diskConfig()),
        observations_(kBudgetCount * kRequestCount + 1) {}

    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask task) override {
        const auto request = task.requestId();
        if (request && *request >= 0 && static_cast<size_t>(*request) < observations_.size()) {
            auto& observation = observations_[static_cast<size_t>(*request)];
            observation.trace = task.traceId();
            ++observation.calls;
        } else {
            invalid_request_.store(true);
        }
        return PerRankBlockTransferEngine::executeBatch(std::move(task));
    }

    const Observation& observation(size_t request) const { return observations_.at(request); }
    bool invalidRequest() const { return invalid_request_.load(); }

private:
    static DiskIOExecutorConfig diskConfig() {
        DiskIOExecutorConfig config;
        config.read_thread_count = 1;
        config.write_thread_count = 1;
        return config;
    }
    std::vector<Observation> observations_;
    std::atomic<bool> invalid_request_{false};
};

struct QueueSmokeCompletion {
    std::promise<TransferResult> promise;
    std::atomic<size_t> callbacks{0};
};

class JoinProducers {
public:
    explicit JoinProducers(std::vector<std::thread>& threads): threads_(threads) {}
    ~JoinProducers() { join(); }
    void join() {
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
private:
    std::vector<std::thread>& threads_;
};

class WorkLoopWriteQueueTimeoutSmokeTest: public ::testing::Test {
protected:
    void SetUp() override {
        // One rank: approximately 16 MiB host buffers and 16 MiB disk space.
        host_ = makeHostPool(kBlockBytes, kRequestCount);
        auto io = std::make_unique<QueueSmokePosixIO>();
        io_ = io.get();
        disk_ = makeDiskPool(kBlockBytes, kRequestCount, directory_.path, std::move(io),
                             "workloop_write_queue_smoke", false);
        auto topology = makeTestTopology({makeTestGroupBase(
            defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, kBlockBytes)});
        groups_ = {makeTestGroupSet(0, topology, topology->groupTags(), {}, host_, disk_)};
        engine_ = std::make_shared<QueueSmokeEngine>(groups_);
        dispatcher_ = std::make_unique<BlockTransferDispatcher>(engine_);
        ASSERT_TRUE(engine_->asyncEnabled());
        ASSERT_EQ(engine_->transferWorkerCount(), 1u);
        ASSERT_NE(engine_->diskIOExecutor(), nullptr);
        ASSERT_EQ(engine_->diskIOExecutor()->config().write_thread_count, 1u);
        ASSERT_GE(engine_->diskIOExecutor()->config().write_queue_size, kRequestCount);

        references_held_ = true;
        for (size_t i = 0; i < kRequestCount; ++i) {
            const auto host_block = poolMalloc(*host_);
            ASSERT_NE(host_block, NULL_BLOCK_IDX);
            host_->incTreeRef(host_block, BlockTreeRefType::CACHE);
            hosts_.push_back(host_block);
            const auto disk_block = poolMalloc(*disk_);
            ASSERT_NE(disk_block, NULL_BLOCK_IDX);
            disk_->incTreeRef(disk_block, BlockTreeRefType::CACHE);
            disks_.push_back(disk_block);
        }
    }

    void TearDown() override {
        // Also drain after an assertion failure. Callbacks retain only their
        // shared completion slots, and producer threads always join on unwind.
        if (engine_) {
            engine_->waitForIdle();
        }
        if (dispatcher_) {
            dispatcher_->waitForTransfers();
        }
        dispatcher_.reset();
        releaseReferences();
        engine_.reset();
    }

    static bool containsOnly(const void* buffer, unsigned char value) {
        const auto* bytes = static_cast<const unsigned char*>(buffer);
        return std::all_of(bytes, bytes + kBlockBytes, [value](unsigned char byte) { return byte == value; });
    }

    void seedDisk() {
        for (size_t i = 0; i < kRequestCount; ++i) {
            const auto buffer = host_->blockBuffer(hosts_[i]);
            std::memset(buffer.addr, kDiskSeed, kBlockBytes);
            ASSERT_EQ(io_->write(disk_->blockOffset(disks_[i]), buffer.addr, kBlockBytes), DiskBlockIOStatus::OK);
            std::memset(buffer.addr, kSource, kBlockBytes);
        }
        io_->clear();
    }

    static BlockTransferDispatcher::TransferDoneCallback callback(
        const std::shared_ptr<QueueSmokeCompletion>& completion) {
        return [completion](TransferResult result) {
            if (completion->callbacks.fetch_add(1) == 0) {
                completion->promise.set_value(std::move(result));
            }
        };
    }

    void releaseReferences() {
        if (!references_held_) {
            return;
        }
        for (const auto block : hosts_) {
            host_->decTreeRef(block, BlockTreeRefType::CACHE);
        }
        for (const auto block : disks_) {
            disk_->decTreeRef(block, BlockTreeRefType::CACHE);
        }
        references_held_ = false;
    }

    void checkNoFrozenReferences() {
        EXPECT_TRUE(dispatcher_->frozenTasks().empty());
        EXPECT_EQ(host_->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
        EXPECT_EQ(disk_->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    }

    void longTimeoutSuccessor(size_t index) {
        const auto buffer = host_->blockBuffer(hosts_[index]);
        std::memset(buffer.addr, kSuccessor, kBlockBytes);
        auto completion = std::make_shared<QueueSmokeCompletion>();
        auto future = completion->promise.get_future();
        auto done = callback(completion);
        std::vector<TransferDescriptor> descriptors{
            TransferDescriptor::hostToDisk(0, hosts_[index], disks_[index])};
        const size_t request = kBudgetCount * kRequestCount;
        TransferTask task(std::move(descriptors), 10s, static_cast<int64_t>(request));
        dispatcher_->runTransfer(std::move(task), std::move(done), "evict");
        ASSERT_EQ(future.wait_for(15s), std::future_status::ready);
        const auto result = future.get();
        engine_->waitForIdle();
        dispatcher_->waitForTransfers();
        ASSERT_TRUE(result.ok()) << result.status.ToString();
        EXPECT_TRUE(result.buffer_safe);
        EXPECT_EQ(completion->callbacks.load(), 1u);
        EXPECT_EQ(engine_->observation(request).calls, 1u);
        EXPECT_EQ(dispatcher_->taskRegistry().queryLocal(
                      std::vector<uint64_t>{engine_->observation(request).trace}).front(),
                  TransferTaskRegistry::State::BUFFER_SAFE);
        alignas(kBlockBytes) std::array<unsigned char, kBlockBytes> readback{};
        ASSERT_EQ(io_->read(disk_->blockOffset(disks_[index]), readback.data(), readback.size()),
                  DiskBlockIOStatus::OK);
        EXPECT_TRUE(containsOnly(readback.data(), kSuccessor));
        EXPECT_TRUE(containsOnly(buffer.addr, kSuccessor));
        checkNoFrozenReferences();
    }

    TempDirGuard directory_{"workloop_write_queue_smoke"};
    std::shared_ptr<HostBlockPool> host_;
    DiskBlockPoolPtr disk_;
    QueueSmokePosixIO* io_{nullptr};
    std::vector<GroupSetPtr> groups_;
    std::vector<BlockIdxType> hosts_, disks_;
    std::shared_ptr<QueueSmokeEngine> engine_;
    std::unique_ptr<BlockTransferDispatcher> dispatcher_;
    bool references_held_{false};
};

TEST_F(WorkLoopWriteQueueTimeoutSmokeTest, NaturalQueueTimeoutIsSafeAndLongDeadlineWrites) {
    size_t total_queue_timeouts = 0;
    size_t total_writes = 0;
    size_t successor_index = 0;
    size_t round = 0;
    for (const int budget : {1, 2, 5}) {
        SCOPED_TRACE(::testing::Message() << "timeout_ms=" << budget);
        ASSERT_NO_FATAL_FAILURE(seedDisk());
        std::vector<std::shared_ptr<QueueSmokeCompletion>> completions;
        std::vector<std::future<TransferResult>> futures;
        std::vector<std::vector<TransferDescriptor>> descriptors(kRequestCount);
        completions.reserve(kRequestCount);
        futures.reserve(kRequestCount);
        for (size_t i = 0; i < kRequestCount; ++i) {
            auto completion = std::make_shared<QueueSmokeCompletion>();
            futures.push_back(completion->promise.get_future());
            completions.push_back(std::move(completion));
            descriptors[i].push_back(TransferDescriptor::hostToDisk(0, hosts_[i], disks_[i]));
        }
        const size_t first_request = round++ * kRequestCount;
        std::vector<std::exception_ptr> producer_errors(kProducerCount);
        std::vector<std::thread> producers;
        producers.reserve(kProducerCount);
        JoinProducers joiner(producers);
        for (size_t producer = 0; producer < kProducerCount; ++producer) {
            producers.emplace_back([&, producer] {
                try {
                    for (size_t i = producer; i < kRequestCount; i += kProducerCount) {
                        auto done = callback(completions[i]);
                        // No start gate: ordinary producers immediately submit
                        // independent, real writes to the one direction owner.
                        TransferTask task(std::move(descriptors[i]), std::chrono::milliseconds(budget),
                                          static_cast<int64_t>(first_request + i));
                        dispatcher_->runTransfer(std::move(task), std::move(done), "evict");
                    }
                } catch (...) {
                    producer_errors[producer] = std::current_exception();
                }
            });
        }
        joiner.join();
        for (const auto& error : producer_errors) {
            ASSERT_FALSE(static_cast<bool>(error)) << "producer threw while submitting real transfers";
        }
        const auto completion_deadline = std::chrono::steady_clock::now() + 30s;
        for (auto& future : futures) {
            ASSERT_EQ(future.wait_until(completion_deadline), std::future_status::ready);
        }
        engine_->waitForIdle();
        dispatcher_->waitForTransfers();
        ASSERT_FALSE(engine_->invalidRequest());
        checkNoFrozenReferences();

        // All real I/O and completion callbacks have finished before inspection.
        const auto events = io_->events();
        std::unordered_map<uint64_t, size_t> writes;
        for (const auto& event : events) {
            ASSERT_EQ(event.status, DiskBlockIOStatus::OK);
            EXPECT_TRUE(event.write);
            if (event.write) {
                ++writes[event.offset];
            }
        }
        size_t queue_timeouts = 0;
        size_t other_timeouts = 0;
        size_t successful_writes = 0;
        alignas(kBlockBytes) std::array<unsigned char, kBlockBytes> readback{};
        for (size_t i = 0; i < kRequestCount; ++i) {
            const auto result = futures[i].get();
            SCOPED_TRACE(::testing::Message() << "request=" << first_request + i);
            EXPECT_EQ(completions[i]->callbacks.load(), 1u);
            EXPECT_TRUE(result.buffer_safe) << result.status.ToString();
            ASSERT_TRUE(result.ok() || result.status.code() == ErrorCode::DEADLINE_EXCEEDED)
                << result.status.ToString();
            const auto& observed = engine_->observation(first_request + i);
            if (observed.calls != 0) {
                EXPECT_EQ(observed.calls, 1u);
                ASSERT_NE(observed.trace, 0u);
                EXPECT_EQ(dispatcher_->taskRegistry().queryLocal(
                              std::vector<uint64_t>{observed.trace}).front(),
                          TransferTaskRegistry::State::BUFFER_SAFE);
            }
            const auto offset = disk_->blockOffset(disks_[i]);
            EXPECT_TRUE(containsOnly(host_->blockBuffer(hosts_[i]).addr, kSource));
            if (result.ok()) {
                ++successful_writes;
                EXPECT_EQ(writes[offset], 1u);
                ASSERT_EQ(io_->read(offset, readback.data(), readback.size()), DiskBlockIOStatus::OK);
                EXPECT_TRUE(containsOnly(readback.data(), kSource));
            } else if (result.status.ToString() == "transfer expired in WorkLoop queue") {
                ++queue_timeouts;
                successor_index = i;
                EXPECT_EQ(observed.calls, 1u);
                EXPECT_EQ(result.copy_error, CacheCopyError::NOT_EXECUTED);
                EXPECT_EQ(result.block_errors, std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED});
                EXPECT_EQ(writes[offset], 0u) << "queue-expired transfer reached actual disk I/O";
                ASSERT_EQ(io_->read(offset, readback.data(), readback.size()), DiskBlockIOStatus::OK);
                EXPECT_TRUE(containsOnly(readback.data(), kDiskSeed));
            } else {
                ++other_timeouts;
            }
        }
        total_queue_timeouts += queue_timeouts;
        total_writes += successful_writes;
        std::cout << "natural WorkLoop write queue smoke: timeout_ms=" << budget
                  << " producers=" << kProducerCount << " requests=" << kRequestCount
                  << " queue_timeouts=" << queue_timeouts << " other_timeouts=" << other_timeouts
                  << " successful_real_writes=" << successful_writes
                  << " frozen=" << dispatcher_->frozenTasks().size() << std::endl;
        if (total_queue_timeouts > 0 && total_writes > 0) {
            break;
        }
    }
    RecordProperty("natural_workloop_queue_timeouts", static_cast<int>(total_queue_timeouts));
    RecordProperty("successful_real_writes", static_cast<int>(total_writes));
    EXPECT_GT(total_queue_timeouts, 0u) << "1/2/5 ms scan did not cover the actual WorkLoop queue";
    EXPECT_GT(total_writes, 0u) << "short-budget workload issued no successful real disk writes";
    ASSERT_NO_FATAL_FAILURE(longTimeoutSuccessor(successor_index));
    releaseReferences();
    EXPECT_EQ(host_->freeBlocksNum(), kRequestCount);
    EXPECT_EQ(disk_->freeBlocksNum(), kRequestCount);
}

}  // namespace
}  // namespace rtp_llm
