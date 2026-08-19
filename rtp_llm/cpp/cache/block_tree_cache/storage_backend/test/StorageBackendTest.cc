#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

class CoreDumpGuard {
public:
    CoreDumpGuard(): old_(StaticConfig::user_ft_core_dump_on_exception) {
        StaticConfig::user_ft_core_dump_on_exception = false;
    }
    ~CoreDumpGuard() {
        StaticConfig::user_ft_core_dump_on_exception = old_;
    }

private:
    bool old_;
};

class TestBlockPool: public IBlockPool {
public:
    TestBlockPool(): IBlockPool(makeConfig()) {
        markInitialized();
    }

    size_t blockSizeBytes() const override {
        return 16;
    }

private:
    static std::shared_ptr<const BlockPoolConfigBase> makeConfig() {
        auto config                  = std::make_shared<BlockPoolConfigBase>();
        config->pool_type            = BlockPoolType::DEVICE;
        config->pool_name            = "storage_backend_test";
        config->physical_block_count = 8;
        return config;
    }
};

class HoldingExecutor: public StorageBackendExecutor {
public:
    explicit HoldingExecutor(bool start_result = true): start_result_(start_result) {}

    bool start() override {
        started_ = start_result_;
        return started_;
    }

    bool submit(Task task) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (throw_on_submit_) {
            throw std::runtime_error("submit failed");
        }
        if (reject_ || !started_ || stopped_) {
            return false;
        }
        tasks_.push_back(std::move(task));
        return true;
    }

    void shutdown() noexcept override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        runAll();
    }

    size_t runAll() {
        std::deque<Task> tasks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks.swap(tasks_);
        }
        for (auto& task : tasks) {
            task();
            if (duplicate_) {
                task();
            }
        }
        return tasks.size();
    }

    size_t pendingCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    void setReject(bool value) {
        std::lock_guard<std::mutex> lock(mutex_);
        reject_ = value;
    }

    void setDuplicate(bool value) {
        std::lock_guard<std::mutex> lock(mutex_);
        duplicate_ = value;
    }

    void setThrowOnSubmit(bool value) {
        std::lock_guard<std::mutex> lock(mutex_);
        throw_on_submit_ = value;
    }

private:
    std::mutex       mutex_;
    std::deque<Task> tasks_;
    bool             start_result_{true};
    bool             started_{false};
    bool             stopped_{false};
    bool             reject_{false};
    bool             duplicate_{false};
    bool             throw_on_submit_{false};
};

class TestBackend: public StorageBackend {
public:
    explicit TestBackend(bool init_result = true, std::shared_ptr<StorageBackendExecutor> executor = nullptr):
        StorageBackend(std::move(executor)), init_result_(init_result) {}

    ~TestBackend() override {
        shutdown();
    }

    void blockMatch() {
        std::lock_guard<std::mutex> lock(mutex_);
        match_released_ = false;
    }

    void releaseMatch() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            match_released_ = true;
        }
        cv_.notify_all();
    }

    void failNextMatch() {
        fail_match_ = true;
    }

    void failNextRead() {
        fail_read_ = true;
    }

    void failNextWrite() {
        fail_write_ = true;
    }

    size_t readCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return read_calls_;
    }

    size_t writeHandleCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return write_handle_count_;
    }

    std::vector<BlockInfo> resolve(int layer_id, int group_id, int block_id) const {
        return convertIndexToBuffer(layer_id, group_id, block_id);
    }

    std::string groupTag(size_t group_id) const {
        return topology().groupById(group_id).tag;
    }

    size_t initCalls() const {
        return init_calls_;
    }

    void* initResolvedAddress() const {
        return init_resolved_address_;
    }

protected:
    bool initImpl() override {
        ++init_calls_;
        init_resolved_address_ = convertIndexToBuffer(0, 0, 0).front().addr;
        return init_result_;
    }

    StorageMatchResult matchImpl(const StorageRequest& request) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (fail_match_) {
            fail_match_ = false;
            throw std::runtime_error("match failed");
        }
        cv_.wait(lock, [this] { return match_released_; });
        return {request.handles.size(), nullptr};
    }

    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++read_calls_;
        if (fail_read_) {
            fail_read_ = false;
            throw std::runtime_error("read failed");
        }
    }

    void writeImpl(const StorageRequest& request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        write_handle_count_ = 0;
        for (const auto& key_handles : request.handles) {
            write_handle_count_ += key_handles.size();
        }
        if (fail_write_) {
            fail_write_ = false;
            throw std::runtime_error("write failed");
        }
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    match_released_{true};
    bool                    fail_match_{false};
    bool                    fail_read_{false};
    bool                    fail_write_{false};
    size_t                  read_calls_{0};
    size_t                  write_handle_count_{0};
    bool                    init_result_{true};
    size_t                  init_calls_{0};
    void*                   init_resolved_address_{nullptr};
};

std::shared_ptr<const CacheTopology> makeTopology() {
    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    GroupBase group;
    group.tag                       = spec->tag;
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = {0};
    group.seq_size_per_block        = 1;
    group.kernel_seq_size_per_block = 1;
    return CacheTopology::create({std::move(group)}, {{0, {"default"}}});
}

std::shared_ptr<const CacheTopology> makeSharedPoolTopology() {
    std::vector<GroupBase> groups;
    for (size_t group_id = 0; group_id < 2; ++group_id) {
        auto spec = std::make_shared<MHAKVCacheSpec>();
        spec->tag = "group_" + std::to_string(group_id);
        GroupBase group;
        group.tag                       = spec->tag;
        group.spec                      = std::move(spec);
        group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.layer_ids                 = {0};
        group.seq_size_per_block        = 1;
        group.kernel_seq_size_per_block = 1;
        groups.push_back(std::move(group));
    }
    return CacheTopology::create(std::move(groups), {{0, {"group_0", "group_1"}}});
}

bool initBackend(TestBackend& backend, const std::shared_ptr<IBlockPool>& pool) {
    return backend.init(
        makeTopology(),
        {pool},
        [](int layer_id, int group_id, int block_id) {
            auto address = reinterpret_cast<void*>(static_cast<uintptr_t>(block_id + 1));
            return std::vector<BlockInfo>{{false, layer_id, group_id, address, 16}};
        });
}

StorageRequest makeRequest(BlockIdxType block, size_t key_count = 1) {
    CacheKeysType keys;
    keys.reserve(key_count);
    std::vector<std::vector<StorageBlockHandle>> handles;
    handles.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        keys.push_back(i + 1);
        handles.push_back({{0, block}});
    }
    return {std::make_shared<CacheKeysType>(std::move(keys)), std::move(handles)};
}

TEST(StorageBackendTest, DefaultExecutorRunsOperationsAsynchronously) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    backend.blockMatch();
    ASSERT_TRUE(initBackend(backend, pool));

    std::promise<void> completed;
    auto               future = completed.get_future();
    backend.match(makeRequest(NULL_BLOCK_IDX, 2), [&](size_t count, auto, bool success) {
        EXPECT_EQ(count, 2u);
        EXPECT_TRUE(success);
        completed.set_value();
    });
    EXPECT_NE(future.wait_for(std::chrono::milliseconds(50)), std::future_status::ready);
    backend.releaseMatch();
    EXPECT_EQ(future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    backend.shutdown();
}

TEST(StorageBackendTest, CustomExecutorControlsScheduling) {
    auto        pool     = std::make_shared<TestBlockPool>();
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    bool completed = false;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t count, auto, bool success) {
        EXPECT_EQ(count, 1u);
        EXPECT_TRUE(success);
        completed = true;
    });
    EXPECT_FALSE(completed);
    EXPECT_EQ(executor->pendingCount(), 1u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(completed);
    backend.shutdown();
}

TEST(StorageBackendTest, ExecutorStartFailurePropagatesFromInit) {
    auto        pool     = std::make_shared<TestBlockPool>();
    auto        executor = std::make_shared<HoldingExecutor>(/*start_result=*/false);
    TestBackend backend(/*init_result=*/true, executor);
    EXPECT_FALSE(initBackend(backend, pool));
}

TEST(StorageBackendTest, SubmissionFailureCompletesOnceAndReleasesPins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    executor->setReject(true);
    size_t read_completions = 0;
    backend.read(makeRequest(block), nullptr, [&](bool success) {
        ++read_completions;
        EXPECT_FALSE(success);
    });
    EXPECT_EQ(read_completions, 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);

    size_t match_completions = 0;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
        ++match_completions;
        EXPECT_FALSE(success);
    });
    EXPECT_EQ(match_completions, 1u);

    backend.write(backend.prepareWrite(makeRequest(block)));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);

    executor->setReject(false);
    executor->setThrowOnSubmit(true);
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
        ++match_completions;
        EXPECT_FALSE(success);
    });
    EXPECT_EQ(match_completions, 2u);
    pool->decRef(block, BlockRefType::REQUEST);
    backend.shutdown();
}

TEST(StorageBackendTest, IoExceptionsPropagateFailureAndReleasePins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    backend.failNextRead();
    bool read_success = true;
    backend.read(makeRequest(block), nullptr, [&](bool success) { read_success = success; });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_FALSE(read_success);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);

    backend.failNextWrite();
    backend.write(backend.prepareWrite(makeRequest(block)));
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);

    backend.failNextMatch();
    bool match_success = true;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) { match_success = success; });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_FALSE(match_success);
    pool->decRef(block, BlockRefType::REQUEST);
    backend.shutdown();
}

TEST(StorageBackendTest, DuplicateExecutorInvocationCompletesExactlyOnce) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    auto executor = std::make_shared<HoldingExecutor>();
    executor->setDuplicate(true);
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    size_t completions = 0;
    backend.read(makeRequest(block), nullptr, [&](bool success) {
        EXPECT_TRUE(success);
        ++completions;
    });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(completions, 1u);
    EXPECT_EQ(backend.readCalls(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
    backend.shutdown();
}

TEST(StorageBackendTest, ShutdownDrainsAcceptedTasksAndRejectsNewOnes) {
    auto        pool     = std::make_shared<TestBlockPool>();
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    backend.blockMatch();
    ASSERT_TRUE(initBackend(backend, pool));

    std::atomic<bool> completed{false};
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) { completed.store(success); });
    ASSERT_EQ(executor->pendingCount(), 1u);

    auto shutdown = std::async(std::launch::async, [&] { backend.shutdown(); });
    EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    std::promise<void> rejection_entered;
    auto               rejection_entered_future = rejection_entered.get_future();
    std::promise<void> release_rejection;
    auto               release_rejection_future = release_rejection.get_future().share();
    auto               rejection                = std::async(std::launch::async, [&] {
        backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
            EXPECT_FALSE(success);
            rejection_entered.set_value();
            release_rejection_future.wait();
        });
    });
    EXPECT_EQ(rejection_entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    backend.releaseMatch();
    EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    release_rejection.set_value();
    rejection.get();
    EXPECT_EQ(shutdown.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(completed.load());

    size_t rejected = 0;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
        EXPECT_FALSE(success);
        ++rejected;
    });
    EXPECT_EQ(rejected, 1u);
}

TEST(StorageBackendTest, ShutdownFromCompletionIsRejectedWithoutDeadlock) {
    CoreDumpGuard guard;
    auto          pool     = std::make_shared<TestBlockPool>();
    auto          executor = std::make_shared<HoldingExecutor>();
    TestBackend   backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    bool rejected = false;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
        EXPECT_TRUE(success);
        try {
            backend.shutdown();
        } catch (...) {
            rejected = true;
        }
    });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(rejected);
    backend.shutdown();
}

TEST(StorageBackendTest, ReadAfterShutdownReleasesPin) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    TestBackend backend;
    ASSERT_TRUE(backend.init(
        makeTopology(),
        {pool},
        [](int, int, int) { return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}}; }));
    backend.shutdown();

    bool success = true;
    backend.read(makeRequest(block), nullptr, [&](bool current_success) { success = current_success; });
    EXPECT_FALSE(success);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
}

TEST(StorageBackendTest, UnsubmittedWriteTasksReleasePinsAcrossShutdown) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    TestBackend backend;
    ASSERT_TRUE(backend.init(
        makeTopology(),
        {pool},
        [](int, int, int) { return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}}; }));

    auto prepared_before_shutdown = backend.prepareWrite(makeRequest(block));
    backend.shutdown();
    prepared_before_shutdown = {};
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);

    auto prepared_after_shutdown = backend.prepareWrite(makeRequest(block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    prepared_after_shutdown = {};
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
}

TEST(StorageBackendTest, EmptyPerKeyRowsDoNotCreateWriteTask) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1}), {{}}};
    EXPECT_FALSE(backend.prepareWrite(std::move(request)));
    backend.shutdown();
}

TEST(StorageBackendTest, InitPopulatesBaseResourcesBeforeCallingDerivedImplementation) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;

    EXPECT_TRUE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
    EXPECT_EQ(backend.groupTag(0), "default");
    EXPECT_EQ(backend.initResolvedAddress(), reinterpret_cast<void*>(1));
    backend.shutdown();
}

TEST(StorageBackendTest, InitPropagatesDerivedFailure) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend(/*init_result=*/false);

    EXPECT_FALSE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
}

TEST(StorageBackendTest, WritePinsEachPhysicalBlockOnceUntilCompletion) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    auto task = backend.prepareWrite(makeRequest(block, 2));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    backend.write(std::move(task));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(backend.writeHandleCount(), 2u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
    backend.shutdown();
}

TEST(StorageBackendTest, SharedPoolPinsAndReleasesPhysicalBlockOnce) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(backend.init(
        makeSharedPoolTopology(),
        {pool, pool},
        [](int, int, int) { return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}}; }));

    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1}), {{{0, block}, {1, block}}}};
    backend.write(backend.prepareWrite(std::move(request)));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
    backend.shutdown();
}

TEST(StorageBackendTest, UninitializedBackendRejectsTaskAndReleasesPinnedSource) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    TestBackend initialized_backend;
    ASSERT_TRUE(initBackend(initialized_backend, pool));
    auto task = initialized_backend.prepareWrite(makeRequest(block));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);

    TestBackend uninitialized_backend;
    EXPECT_ANY_THROW(uninitialized_backend.write(std::move(task)));
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
    initialized_backend.shutdown();
}

TEST(StorageBackendTest, ReadPinsTargetsUntilCompletion) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block, BlockRefType::REQUEST);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    bool completed = false;

    backend.read(makeRequest(block), nullptr, [&](bool success) {
        EXPECT_TRUE(success);
        completed = true;
    });
    EXPECT_FALSE(completed);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 1u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(completed);
    EXPECT_EQ(backend.readCalls(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND), 0u);
    pool->decRef(block, BlockRefType::REQUEST);
    backend.shutdown();
}

TEST(StorageBackendTest, ResolvesGpuBufferAndGroupMetadataFromBoundResources) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));

    EXPECT_EQ(backend.groupTag(0), "default");
    const auto buffers = backend.resolve(0, 0, 3);
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers.front().addr, reinterpret_cast<void*>(4));
    backend.shutdown();
}

}  // namespace
}  // namespace rtp_llm
