#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/DecodePublicationBudget.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::BoundedThread;

TEST(StorageBackendExecutorTest, RejectsZeroDimensions) {
    EXPECT_THROW((void)makeStorageBackendExecutor(/*thread_count=*/0, /*queue_size=*/1), std::invalid_argument);
    EXPECT_THROW((void)makeStorageBackendExecutor(/*thread_count=*/1, /*queue_size=*/0), std::invalid_argument);
}

TEST(StorageBackendExecutorTest, HonorsConfiguredWorkerAndQueueCapacity) {
    auto executor = makeStorageBackendExecutor(/*thread_count=*/2, /*queue_size=*/1);
    ASSERT_TRUE(executor->start());

    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  started = 0;
    bool                    release = false;
    std::atomic<size_t>     completed{0};
    auto                    blocking_task = [&] {
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++started;
            cv.notify_all();
            cv.wait(lock, [&] { return release; });
        }
        completed.fetch_add(1);
        cv.notify_all();
    };

    const bool first_submitted = executor->submit(blocking_task);
    bool       first_started   = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        first_started = cv.wait_for(lock, std::chrono::seconds(5), [&] { return started >= 1; });
    }
    const bool second_submitted = first_started && executor->submit(blocking_task);
    bool       both_started     = false;
    if (second_submitted) {
        std::unique_lock<std::mutex> lock(mutex);
        both_started = cv.wait_for(lock, std::chrono::seconds(5), [&] { return started == 2; });
    }
    const bool queued_submitted  = both_started && executor->submit([&] {
        completed.fetch_add(1);
        cv.notify_all();
    });
    const bool overflow_rejected = both_started && !executor->submit([] {});
    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
    }
    cv.notify_all();
    bool all_completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        all_completed = cv.wait_for(lock, std::chrono::seconds(5), [&] { return completed.load() == 3; });
    }
    executor->shutdown();
    EXPECT_TRUE(first_submitted);
    EXPECT_TRUE(first_started);
    EXPECT_TRUE(second_submitted);
    EXPECT_TRUE(both_started);
    EXPECT_TRUE(queued_submitted);
    EXPECT_TRUE(overflow_rejected);
    EXPECT_TRUE(all_completed);
    EXPECT_EQ(completed.load(), 3u);
}

TEST(StorageBackendExecutorTest, ShutdownDrainsQueuedTasksAndWaitsForConcurrentShutdown) {
    auto executor = makeStorageBackendExecutor(/*thread_count=*/1, /*queue_size=*/8);
    ASSERT_TRUE(executor->start());
    auto               entered = std::make_shared<std::promise<void>>();
    auto               started = entered->get_future();
    std::promise<void> release;
    auto               released  = release.get_future().share();
    auto               completed = std::make_shared<std::atomic<size_t>>(0);
    ASSERT_TRUE(executor->submit([entered, released, completed] {
        entered->set_value();
        released.wait();
        completed->fetch_add(1);
    }));
    const auto started_status = started.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(started_status, std::future_status::ready);
    EXPECT_TRUE(executor->submit([completed] { completed->fetch_add(1); }));
    EXPECT_TRUE(executor->submit([] { throw std::runtime_error("task failure"); }));
    EXPECT_TRUE(executor->submit([completed] { completed->fetch_add(1); }));

    BoundedThread<void> first([executor] { executor->shutdown(); });
    EXPECT_EQ(first.waitFor(std::chrono::milliseconds(50)), std::future_status::timeout);
    BoundedThread<void> second([executor] { executor->shutdown(); });
    EXPECT_EQ(second.waitFor(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_FALSE(executor->submit([] {}));
    release.set_value();
    ASSERT_EQ(first.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    first.get();
    ASSERT_EQ(second.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    second.get();
    EXPECT_EQ(completed->load(), 3u);
    EXPECT_FALSE(executor->submit([] {}));
}

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

class TestBlockPool: public DeviceBlockPool {
public:
    TestBlockPool(): DeviceBlockPool(makeConfig()) {
        markInitialized();
    }

    size_t blockSizeBytes() const override {
        return 16;
    }

private:
    static std::shared_ptr<const DeviceBlockPoolConfig> makeConfig() {
        auto config                  = std::make_shared<DeviceBlockPoolConfig>();
        config->pool_type            = BlockPoolType::DEVICE;
        config->pool_name            = "storage_backend_test";
        config->physical_block_count = 8;
        MemoryLayoutConfig layout;
        layout.block_num                = 8;
        layout.layer_num                = 1;
        layout.kv_block_stride_bytes    = 16;
        layout.kv_block_pool_size_bytes = 128;
        config->memory_layouts.push_back(layout);
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
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (throw_on_submit_) {
                throw std::runtime_error("submit failed");
            }
            if (reject_ || !started_ || stopped_) {
                return false;
            }
            tasks_.push_back(std::move(task));
        }
        cv_.notify_all();
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

    bool runNext() {
        Task task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) {
                return false;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
        if (duplicate_) {
            task();
        }
        return true;
    }

    bool runLast() {
        Task task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) {
                return false;
            }
            task = std::move(tasks_.back());
            tasks_.pop_back();
        }
        task();
        if (duplicate_) {
            task();
        }
        return true;
    }

    size_t pendingCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    bool waitForPendingCount(size_t expected) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return tasks_.size() >= expected; });
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
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::deque<Task>        tasks_;
    bool                    start_result_{true};
    bool                    started_{false};
    bool                    stopped_{false};
    bool                    reject_{false};
    bool                    duplicate_{false};
    bool                    throw_on_submit_{false};
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
        match_entered_  = false;
    }

    bool waitUntilMatchEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return match_entered_; });
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

    std::vector<BlockInfo> resolve(int layer_id, const std::string& tag, int block_id) const {
        return convertIndexToBuffer(layer_id, tag, block_id);
    }

    const std::vector<std::string>& poolTags() const {
        return topology().groupTags();
    }

    size_t initCalls() const {
        return init_calls_;
    }

    void* initResolvedAddress() const {
        return init_resolved_address_;
    }

    bool shutdownCalled() const {
        return shutdown_called_.load();
    }

protected:
    bool initImpl() override {
        ++init_calls_;
        init_resolved_address_ = convertIndexToBuffer(0, topology().groupTags().front(), 0).front().addr;
        return init_result_;
    }

    StorageMatchResult matchImpl(const StorageRequest& request) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (fail_match_) {
            fail_match_ = false;
            throw std::runtime_error("match failed");
        }
        match_entered_ = true;
        cv_.notify_all();
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

    void shutdownImpl() noexcept override {
        shutdown_called_.store(true);
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    match_released_{true};
    bool                    match_entered_{false};
    bool                    fail_match_{false};
    bool                    fail_read_{false};
    bool                    fail_write_{false};
    size_t                  read_calls_{0};
    size_t                  write_handle_count_{0};
    bool                    init_result_{true};
    size_t                  init_calls_{0};
    void*                   init_resolved_address_{nullptr};
    std::atomic<bool>       shutdown_called_{false};
};

class QuarantineObservingBackend: public TestBackend {
public:
    using TestBackend::TestBackend;

    std::function<void()>   read_action;
    std::function<void()>   write_action;
    size_t                  quarantine_events{0};
    bool                    last_quarantine_is_write{false};
    StorageQuarantineReason last_quarantine_reason{StorageQuarantineReason::RPC_ERROR};
    size_t                  last_quarantine_blocks{0};
    size_t                  quarantined_tasks{0};
    size_t                  quarantined_blocks{0};

protected:
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) override {
        if (read_action) {
            read_action();
        } else {
            TestBackend::readImpl(request, match_meta);
        }
    }

    void writeImpl(const StorageRequest& request) override {
        if (write_action) {
            write_action();
        } else {
            TestBackend::writeImpl(request);
        }
    }

    void onQuarantined(bool is_write, StorageQuarantineReason reason, size_t block_count) noexcept override {
        ++quarantine_events;
        last_quarantine_is_write = is_write;
        last_quarantine_reason   = reason;
        last_quarantine_blocks   = block_count;
    }

    void onQuarantineChanged(uint64_t,
                             size_t task_count,
                             size_t block_count,
                             const StorageQuarantineBreakdown&) noexcept override {
        quarantined_tasks  = task_count;
        quarantined_blocks = block_count;
    }
};

class ProtectedBackend: public QuarantineObservingBackend {
public:
    using QuarantineObservingBackend::QuarantineObservingBackend;
    ~ProtectedBackend() override { shutdown(); }
    std::shared_ptr<void> resources = std::make_shared<int>(1);
    std::atomic<bool> quiescent{false};
    std::atomic<bool> external_unknown{false};
    ExternalIoGuard trackWorkerIo() { return trackExternalIo(); }
    void proveQuiescence() {
        quiescent.store(true);
        notifyCompletionQuiescence();
    }
protected:
    bool protectUnknownCompletion() const noexcept override { return true; }
    std::shared_ptr<void> completionResourceOwner() const override { return resources; }
    bool hasUnresolvedExternalIo() const noexcept override { return external_unknown.load(); }
    bool quiesceUnknownOperations() noexcept override { return quiescent.load(); }
};

class PredicateBoundaryBackend: public ProtectedBackend {
public:
    using ProtectedBackend::ProtectedBackend;
    ~PredicateBoundaryBackend() override { shutdown(); }
    std::promise<void> predicate_entered, resume_predicate;
    std::shared_future<void> resume = resume_predicate.get_future().share();
protected:
    bool quiesceUnknownOperations() noexcept override {
        const bool safe = quiescent.load();
        if (!safe && !paused_.exchange(true)) {
            predicate_entered.set_value();
            resume.wait();
        }
        return safe;
    }
private:
    std::atomic<bool> paused_{false};
};

std::shared_ptr<const CacheTopology> makeTopology() {
    auto      spec = std::make_shared<MHAKVCacheSpec>("default", 1, 1, 1);
    GroupBase group;
    group.tag    = spec->tag;
    group.spec   = std::move(spec);
    group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    return CacheTopology::create({std::move(group)}, {{0, {"default"}}});
}

std::shared_ptr<const CacheTopology> makeSharedPoolTopology() {
    std::vector<GroupBase> groups;
    for (size_t group_id = 0; group_id < 2; ++group_id) {
        auto      spec = std::make_shared<MHAKVCacheSpec>("group_" + std::to_string(group_id), 1, 1, 1);
        GroupBase group;
        group.tag    = spec->tag;
        group.spec   = std::move(spec);
        group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
        groups.push_back(std::move(group));
    }
    return CacheTopology::create(std::move(groups), {{0, {"group_0", "group_1"}}});
}

bool initBackend(TestBackend& backend, const DeviceBlockPoolPtr& pool) {
    return backend.init(makeTopology(), {{"default", pool}}, [](int layer_id, const std::string& tag, int block_id) {
        RTP_LLM_CHECK(tag == "default");
        auto address = reinterpret_cast<void*>(static_cast<uintptr_t>(block_id + 1));
        return std::vector<BlockInfo>{{false, layer_id, 0, address, 16}};
    });
}

StorageRequest makeRequest(BlockIdxType block, size_t key_count = 1) {
    CacheKeysType keys;
    keys.reserve(key_count);
    std::vector<std::vector<StorageBlockHandle>> handles;
    handles.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        keys.push_back(i + 1);
        handles.push_back({{"default", block}});
    }
    return {std::make_shared<CacheKeysType>(std::move(keys)), std::move(handles)};
}

TEST(StorageBackendTest, CompletedDecodePinsUniquePeersAndReleasesBudgetOnRejection) {
    auto pool = std::make_shared<TestBlockPool>();
    const auto first = pool->malloc().value(), second = pool->malloc().value();
    pool->incRef(first);
    pool->incRef(second);
    auto executor = std::make_shared<HoldingExecutor>();
    ProtectedBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    DecodePublicationBudget budget(32, 1);
    auto request = makeRequest(first, 2);
    request.decode_completion = true;
    request.completion_budget_lease = budget.tryAcquire(32);
    for (auto& row : request.handles) { row.front().peer_blocks = {first, second, first}; }
    auto task = backend.prepareWrite(std::move(request));
    EXPECT_EQ(pool->refCount(first), 2u);
    EXPECT_EQ(pool->refCount(second), 2u);
    executor->setReject(true);
    EXPECT_FALSE(backend.write(std::move(task)));
    EXPECT_EQ(pool->refCount(first), 1u);
    EXPECT_EQ(pool->refCount(second), 1u);
    EXPECT_EQ(budget.snapshot().jobs, 0u);
    pool->decRef(first);
    pool->decRef(second);
    backend.shutdown();
}

TEST(StorageBackendTest, MidPreparePinFailureRollsBackRecordedPinsAndBudgetExactlyOnce) {
    struct Observer {
        std::shared_ptr<size_t> copies;
        explicit Observer(std::shared_ptr<size_t> value): copies(std::move(value)) {}
        Observer(Observer&&) noexcept = default;
        Observer(const Observer& other): copies(other.copies) {
            if (++*copies == 2) { throw std::bad_alloc(); }
        }
        void operator()() const {}
    };
    auto pool = std::make_shared<TestBlockPool>();
    const auto first = pool->malloc().value(), second = pool->malloc().value();
    pool->incRef(first);
    pool->incRef(second);
    auto copies = std::make_shared<size_t>(0);
    pool->setCapacityChangeCallback(Observer(copies));
    *copies = 0;
    ProtectedBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));
    DecodePublicationBudget budget(32, 1);
    auto request = makeRequest(first);
    request.decode_completion = true;
    request.handles.front().front().peer_blocks = {first, second};
    request.completion_budget_lease = budget.tryAcquire(32);
    EXPECT_THROW(backend.prepareWrite(std::move(request)), std::bad_alloc);
    EXPECT_EQ(*copies, 2u);
    EXPECT_EQ(pool->refCount(first), 1u);
    EXPECT_EQ(pool->refCount(second), 1u);
    EXPECT_EQ(budget.snapshot().jobs, 0u);
    pool->decRef(first);
    pool->decRef(second);
    backend.shutdown();
}

TEST(StorageBackendTest, ProtectedReadTimeoutRetainsTargetUntilExplicitQuiescence) {
    auto pool = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    ProtectedBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    backend.read_action = [] {
        throw StorageOperationTimeout("accepted read timed out", StorageQuarantineReason::SDK_TIMEOUT);
    };
    bool completed = false;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) {
        EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_EQ(pool->refCount(block), 2u);
        EXPECT_FALSE(pool->isFrozen(block));
        completed = true;
    });
    ASSERT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(completed);
    EXPECT_EQ(backend.quarantine_events, 1u);
    backend.proveQuiescence();
    backend.shutdown();
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
}

TEST(StorageBackendTest, PeerProjectionRejectedForReadsAndOrdinaryWritesBeforePins) {
    auto pool = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    ProtectedBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));
    auto request = makeRequest(block);
    request.handles.front().front().peer_blocks = {block};
    EXPECT_ANY_THROW(backend.prepareWrite(request));
    EXPECT_ANY_THROW(backend.read(request, nullptr, {}));
    request.decode_completion = true;
    EXPECT_ANY_THROW(backend.read(request, nullptr, {}));
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, PostEntryAllocationFailureRetainsOwnersUntilExplicitQuiescence) {
    auto pool = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    auto backend = std::make_shared<ProtectedBackend>(true, executor);
    ASSERT_TRUE(initBackend(*backend, pool));
    DecodePublicationBudget budget(16, 1);
    auto request = makeRequest(block);
    request.decode_completion = true;
    request.completion_budget_lease = budget.tryAcquire(16);
    std::weak_ptr<void> owner = backend->resources;
    std::weak_ptr<const CacheKeysType> keys = request.keys;
    backend->write_action = [] { throw std::bad_alloc(); };
    ASSERT_TRUE(backend->write(backend->prepareWrite(std::move(request))));
    backend->resources.reset();
    ASSERT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(backend->quarantine_events, 1u);
    EXPECT_FALSE(owner.expired());
    EXPECT_FALSE(keys.expired());
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_EQ(budget.snapshot().jobs, 1u);
    BoundedThread<void> stopping([backend] { backend->shutdown(); });
    EXPECT_EQ(stopping.waitFor(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_FALSE(backend->shutdownCalled());
    EXPECT_EQ(pool->refCount(block), 2u);
    backend->proveQuiescence();
    ASSERT_EQ(stopping.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    stopping.get();
    EXPECT_TRUE(backend->shutdownCalled());
    EXPECT_TRUE(owner.expired());
    EXPECT_TRUE(keys.expired());
    EXPECT_EQ(budget.snapshot().jobs, 0u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
}

TEST(StorageBackendTest, WorkerTransactionRecordsUnknownBeforeShutdownCanDestroyTransport) {
    auto pool = std::make_shared<TestBlockPool>();
    auto backend = std::make_shared<ProtectedBackend>();
    ASSERT_TRUE(initBackend(*backend, pool));
    auto transaction = backend->trackWorkerIo();
    ASSERT_TRUE(transaction);
    BoundedThread<void> stopping([backend] { backend->shutdown(); });
    EXPECT_EQ(stopping.waitFor(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_FALSE(backend->shutdownCalled());
    backend->external_unknown.store(true);
    transaction = {};
    EXPECT_EQ(stopping.waitFor(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_FALSE(backend->shutdownCalled());
    backend->proveQuiescence();
    ASSERT_EQ(stopping.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    stopping.get();
    EXPECT_TRUE(backend->shutdownCalled());
    EXPECT_FALSE(backend->trackWorkerIo());
}

TEST(StorageBackendTest, QuiescenceNotificationCannotBeLostAtPredicateWaitBoundary) {
    auto pool = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    auto backend = std::make_shared<PredicateBoundaryBackend>(true, executor);
    ASSERT_TRUE(initBackend(*backend, pool));
    backend->write_action = [] { throw std::bad_alloc(); };
    auto request = makeRequest(block);
    request.decode_completion = true;
    ASSERT_TRUE(backend->write(backend->prepareWrite(std::move(request))));
    ASSERT_EQ(executor->runAll(), 1u);
    auto entered = backend->predicate_entered.get_future();
    BoundedThread<void> stopping([backend] { backend->shutdown(); });
    ASSERT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto announced = std::make_shared<std::promise<void>>();
    auto notifying = announced->get_future();
    BoundedThread<void> notifier([backend, announced] {
        announced->set_value();
        backend->proveQuiescence();
    });
    ASSERT_EQ(notifying.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // The predicate deliberately holds the registry mutex after reading false.
    // Notification must wait for that mutex, so it cannot slip before wait().
    EXPECT_EQ(notifier.waitFor(std::chrono::milliseconds(100)), std::future_status::timeout);
    backend->resume_predicate.set_value();
    ASSERT_EQ(notifier.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    notifier.get();
    ASSERT_EQ(stopping.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    stopping.get();
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
}

TEST(StorageBackendTest, ExplicitPayloadSafeFailureReleasesWithoutQuarantine) {
    auto pool = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    ProtectedBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    backend.write_action = [] { throw StorageOperationBufferSafeFailure("payload completed; metadata failed"); };
    DecodePublicationBudget budget(16, 1);
    auto request = makeRequest(block);
    request.decode_completion = true;
    request.completion_budget_lease = budget.tryAcquire(16);
    ASSERT_TRUE(backend.write(backend.prepareWrite(std::move(request))));
    ASSERT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(backend.quarantine_events, 0u);
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_EQ(budget.snapshot().jobs, 0u);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, InvalidHandlesFailBeforePinOrIoSubmission) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    auto unknown                        = makeRequest(block);
    unknown.handles.front().front().tag = "missing";
    EXPECT_ANY_THROW(backend.prepareWrite(unknown));
    EXPECT_ANY_THROW(backend.read(unknown, nullptr, {}));
    unknown.handles.front().front().block = NULL_BLOCK_IDX;
    EXPECT_ANY_THROW(backend.match(unknown, {}));

    auto late_unknown                       = makeRequest(block, 2);
    late_unknown.handles.back().front().tag = "missing";
    EXPECT_ANY_THROW(backend.prepareWrite(late_unknown));

    auto duplicate = makeRequest(block);
    duplicate.handles.front().push_back({"default", block});
    EXPECT_ANY_THROW(backend.prepareWrite(duplicate));

    EXPECT_EQ(executor->pendingCount(), 0u);
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_EQ(backend.readCalls(), 0u);
    EXPECT_EQ(backend.writeHandleCount(), 0u);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, ReorderedPoolInputsPreserveTagIdentityAndTopologyLifetime) {
    auto first_pool   = std::make_shared<TestBlockPool>();
    auto second_pool  = std::make_shared<TestBlockPool>();
    auto first_block  = first_pool->malloc().value();
    auto second_block = second_pool->malloc().value();
    first_pool->incRef(first_block);
    second_pool->incRef(second_block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(true, executor);
    auto        topology   = makeSharedPoolTopology();
    const auto* bound_tags = &topology->groupTags();
    ASSERT_TRUE(
        backend.init(topology, {{"group_1", second_pool}, {"group_0", first_pool}}, [](int, const std::string&, int) {
            return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}};
        }));
    topology.reset();
    EXPECT_EQ(&backend.poolTags(), bound_tags);
    EXPECT_EQ(backend.poolTags(), (std::vector<std::string>{"group_0", "group_1"}));
    StorageRequest first{std::make_shared<CacheKeysType>(CacheKeysType{1}), {{{"group_0", first_block}}}};
    auto           task = backend.prepareWrite(std::move(first));
    EXPECT_EQ(first_pool->refCount(first_block), 2u);
    EXPECT_EQ(first_pool->referencedBlocksNum(), 1u);
    EXPECT_EQ(first_pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 1u);
    EXPECT_EQ(second_pool->refCount(second_block), 1u);
    task = {};
    EXPECT_EQ(first_pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u);
    StorageRequest second{std::make_shared<CacheKeysType>(CacheKeysType{1}), {{{"group_1", second_block}}}};
    backend.read(std::move(second), nullptr, [](ErrorInfo error) { EXPECT_TRUE(error.ok()); });
    EXPECT_EQ(first_pool->refCount(first_block), 1u);
    EXPECT_EQ(second_pool->refCount(second_block), 2u);
    EXPECT_EQ(second_pool->referencedBlocksNum(), 1u);
    EXPECT_EQ(second_pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 1u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(second_pool->refCount(second_block), 1u);
    EXPECT_EQ(second_pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u);
    backend.shutdown();
    first_pool->decRef(first_block);
    second_pool->decRef(second_block);
}

TEST(StorageBackendTest, InvalidPoolBindingsFailBeforeDerivedInitOrIo) {
    CoreDumpGuard guard;
    auto          pool  = std::make_shared<TestBlockPool>();
    auto          block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(true, executor);
    auto        topology    = makeSharedPoolTopology();
    size_t      resolutions = 0;
    const auto  resolver    = [&](int, const std::string&, int) {
        ++resolutions;
        return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}};
    };
    EXPECT_ANY_THROW(backend.init(topology, {{"group_0", pool}}, resolver));
    EXPECT_ANY_THROW(backend.init(topology, {{"group_0", pool}, {"group_1", pool}}, resolver));
    EXPECT_ANY_THROW(backend.init(topology, {{"group_0", pool}, {"missing", pool}}, resolver));
    EXPECT_ANY_THROW(backend.init(topology, {{"group_0", pool}, {"", pool}}, resolver));
    EXPECT_ANY_THROW(backend.init(topology, {{"group_0", pool}, {"group_1", nullptr}}, resolver));
    EXPECT_EQ(backend.initCalls(), 0u);
    EXPECT_EQ(resolutions, 0u);
    EXPECT_EQ(executor->pendingCount(), 0u);
    EXPECT_EQ(pool->refCount(block), 1u);
    ASSERT_TRUE(backend.init(topology, {{"group_1", std::make_shared<TestBlockPool>()}, {"group_0", pool}}, resolver));
    EXPECT_EQ(backend.initCalls(), 1u);
    backend.shutdown();
    pool->decRef(block);
}

TEST(StorageBackendTest, RejectsSharedExecutorBeforeInitializingSecondBackend) {
    auto        pool     = std::make_shared<TestBlockPool>();
    auto        executor = makeStorageBackendExecutor(1, 8);
    TestBackend first(true, executor);
    TestBackend second(true, executor);
    ASSERT_TRUE(initBackend(first, pool));
    EXPECT_FALSE(initBackend(second, pool));
    EXPECT_EQ(second.initCalls(), 0u);
    second.shutdown();
    auto block = pool->malloc().value();
    first.write(first.prepareWrite(makeRequest(block)));
    first.shutdown();
}

TEST(StorageBackendTest, CallbackExceptionsSettleTasksAndPreserveExecutor) {
    BoundedThread<void> test([] {
        auto        pool     = std::make_shared<TestBlockPool>();
        auto        executor = std::make_shared<HoldingExecutor>();
        TestBackend backend(true, executor);
        ASSERT_TRUE(initBackend(backend, pool));
        auto block = pool->malloc().value();
        pool->incRef(block);
        size_t callbacks = 0;
        backend.match(makeRequest(block), [&](size_t, auto, bool) {
            ++callbacks;
            throw std::runtime_error("match callback failure");
        });
        backend.read(makeRequest(block), nullptr, [&](ErrorInfo) {
            ++callbacks;
            throw 42;
        });
        executor->setDuplicate(true);
        EXPECT_NO_THROW(executor->runAll());
        EXPECT_EQ(callbacks, 2u);
        EXPECT_EQ(pool->refCount(block), 1u);
        backend.match(makeRequest(block), [&](size_t, auto, bool success) {
            EXPECT_TRUE(success);
            ++callbacks;
        });
        EXPECT_NO_THROW(executor->runAll());
        EXPECT_EQ(callbacks, 3u);
        // Running shutdown on the same thread also checks the completion marker
        // was restored when either kind of callback exception was caught.
        EXPECT_NO_THROW(backend.shutdown());
        EXPECT_TRUE(backend.shutdownCalled());
        pool->decRef(block);
    });
    ASSERT_EQ(test.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    test.get();
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

TEST(StorageBackendTest, DefaultExecutorShutdownSettlesQueuedOperationsAndPins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor    = makeStorageBackendExecutor(/*thread_count=*/1, /*queue_size=*/8);
    auto backend     = std::make_shared<TestBackend>(true, executor);
    auto completions = std::make_shared<std::atomic<size_t>>(0);
    backend->blockMatch();
    ASSERT_TRUE(initBackend(*backend, pool));
    backend->match(makeRequest(NULL_BLOCK_IDX), [completions](size_t, auto, bool success) {
        EXPECT_TRUE(success);
        completions->fetch_add(1);
    });
    if (!backend->waitUntilMatchEntered()) {
        backend->releaseMatch();
        FAIL() << "match did not occupy the worker";
    }
    backend->read(makeRequest(block), nullptr, [completions](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        completions->fetch_add(1);
    });
    backend->write(backend->prepareWrite(makeRequest(block)));
    EXPECT_EQ(pool->refCount(block), 3u);

    BoundedThread<void> shutdown([backend] { backend->shutdown(); });
    EXPECT_EQ(shutdown.waitFor(std::chrono::milliseconds(50)), std::future_status::timeout);
    backend->releaseMatch();
    ASSERT_EQ(shutdown.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    shutdown.get();
    EXPECT_TRUE(backend->shutdownCalled());
    EXPECT_EQ(completions->load(), 2u);
    EXPECT_EQ(backend->readCalls(), 1u);
    EXPECT_EQ(backend->writeHandleCount(), 1u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
}

TEST(StorageBackendTest, TimelineSeparatesPrepareQueueIoSettlementAndCallback) {
    auto       pool  = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    executor->setDuplicate(true);
    TestBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    auto   session           = AsyncProfileSession::start();
    size_t callbacks         = 0;
    auto   match_request     = makeRequest(NULL_BLOCK_IDX);
    match_request.request_id = 42;
    backend.match(std::move(match_request), [&](size_t, auto, bool ok) {
        EXPECT_TRUE(ok);
        ++callbacks;
    });
    auto read_request       = makeRequest(block);
    read_request.request_id = 43;
    backend.read(std::move(read_request), nullptr, [&](ErrorInfo e) {
        EXPECT_TRUE(e.ok());
        ++callbacks;
    });
    auto write_request       = makeRequest(block);
    write_request.request_id = 44;
    EXPECT_TRUE(backend.write(backend.prepareWrite(std::move(write_request))));
    EXPECT_EQ(callbacks, 0u);
    EXPECT_EQ(executor->runAll(), 3u);
    EXPECT_EQ(callbacks, 2u);
    auto events = session->finish();
    EXPECT_EQ(events.size(), 15u);
    for (const auto* stage : {"match_total",
                              "match_queue_wait",
                              "match_io",
                              "match_callback",
                              "read_total",
                              "read_prepare",
                              "read_queue_wait",
                              "read_io",
                              "read_settle",
                              "read_callback",
                              "write_total",
                              "write_prepare",
                              "write_queue_wait",
                              "write_io",
                              "write_settle"}) {
        size_t count = 0;
        for (const auto& event : events) {
            if (event.name.find(std::string("cache.transfer.storage_backend.") + stage + "(") == 0)
                ++count;
            EXPECT_EQ(event.name.find("incomplete=1"), std::string::npos);
            EXPECT_GE(event.end_us, event.start_us);
        }
        EXPECT_EQ(count, 1u) << stage;
    }
    EXPECT_EQ(pool->refCount(block), 1u);
    backend.shutdown();
    pool->decRef(block);
}

TEST(StorageBackendTest, RejectedStorageOperationsHaveNoIoRangesAndStillReleasePins) {
    auto       pool  = std::make_shared<TestBlockPool>();
    const auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    executor->setReject(true);
    auto   session   = AsyncProfileSession::start();
    size_t callbacks = 0;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo e) {
        EXPECT_FALSE(e.ok());
        ++callbacks;
    });
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool ok) {
        EXPECT_FALSE(ok);
        ++callbacks;
    });
    EXPECT_FALSE(backend.write(backend.prepareWrite(makeRequest(block))));
    EXPECT_EQ(callbacks, 2u);
    const auto events = session->finish();
    EXPECT_EQ(events.size(), 12u);
    for (const auto& event : events) {
        EXPECT_EQ(event.name.find("_io("), std::string::npos);
        EXPECT_EQ(event.name.find("incomplete=1"), std::string::npos);
    }
    EXPECT_EQ(pool->refCount(block), 1u);
    backend.shutdown();
    pool->decRef(block);
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
    EXPECT_FALSE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
    TestBackend other(true, executor);
    EXPECT_FALSE(initBackend(other, pool));
    EXPECT_EQ(other.initCalls(), 0u);
}

TEST(StorageBackendTest, SubmissionFailureCompletesOnceAndReleasesPins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    executor->setReject(true);
    size_t read_completions = 0;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) {
        ++read_completions;
        EXPECT_FALSE(error.ok());
    });
    EXPECT_EQ(read_completions, 1u);
    EXPECT_EQ(pool->refCount(block), 1u);

    size_t match_completions = 0;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
        ++match_completions;
        EXPECT_FALSE(success);
    });
    EXPECT_EQ(match_completions, 1u);

    backend.write(backend.prepareWrite(makeRequest(block)));
    EXPECT_EQ(pool->refCount(block), 1u);

    executor->setReject(false);
    executor->setThrowOnSubmit(true);
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) {
        ++match_completions;
        EXPECT_FALSE(success);
    });
    EXPECT_EQ(match_completions, 2u);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, IoExceptionsPropagateFailureAndReleasePins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    backend.failNextRead();
    bool read_success = true;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) { read_success = error.ok(); });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_FALSE(read_success);
    EXPECT_EQ(pool->refCount(block), 1u);

    backend.failNextWrite();
    backend.write(backend.prepareWrite(makeRequest(block)));
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(pool->refCount(block), 1u);

    backend.failNextMatch();
    bool match_success = true;
    backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) { match_success = success; });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_FALSE(match_success);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, AsyncWritesReturnBeforeCompletionAndReleaseOnlyTheirOwnPins) {
    for (bool complete_first : {true, false}) {
        auto pool         = std::make_shared<TestBlockPool>();
        auto first_block  = pool->malloc().value();
        auto second_block = pool->malloc().value();
        pool->incRef(first_block);
        pool->incRef(second_block);
        auto        executor = std::make_shared<HoldingExecutor>();
        TestBackend backend(true, executor);
        ASSERT_TRUE(initBackend(backend, pool));
        EXPECT_TRUE(backend.write(backend.prepareWrite(makeRequest(first_block))));
        EXPECT_TRUE(backend.write(backend.prepareWrite(makeRequest(second_block))));
        EXPECT_EQ(executor->pendingCount(), 2u);
        EXPECT_EQ(pool->refCount(first_block), 2u);
        EXPECT_EQ(pool->refCount(second_block), 2u);
        ASSERT_TRUE(complete_first ? executor->runNext() : executor->runLast());
        EXPECT_EQ(pool->refCount(complete_first ? first_block : second_block), 1u);
        EXPECT_EQ(pool->refCount(complete_first ? second_block : first_block), 2u);
        ASSERT_TRUE(executor->runNext());
        EXPECT_EQ(pool->refCount(first_block), 1u);
        EXPECT_EQ(pool->refCount(second_block), 1u);
        backend.shutdown();
        pool->decRef(first_block);
        pool->decRef(second_block);
    }
}

TEST(StorageBackendTest, AsyncWriteFailuresAndRejectionReleasePins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    backend.failNextWrite();
    EXPECT_TRUE(backend.write(backend.prepareWrite(makeRequest(block))));
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_NO_THROW(executor->runAll());
    EXPECT_EQ(pool->refCount(block), 1u);
    executor->setReject(true);
    EXPECT_FALSE(backend.write(backend.prepareWrite(makeRequest(block))));
    EXPECT_EQ(pool->refCount(block), 1u);
    executor->setReject(false);
    executor->setThrowOnSubmit(true);
    EXPECT_FALSE(backend.write(backend.prepareWrite(makeRequest(block))));
    EXPECT_EQ(pool->refCount(block), 1u);
    executor->setThrowOnSubmit(false);
    backend.shutdown();
    EXPECT_FALSE(backend.write(backend.prepareWrite(makeRequest(block))));
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
}

TEST(StorageBackendTest, AsyncWriteDuringStoppingIsRejectedAndReleasesPins) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    auto backend  = std::make_shared<TestBackend>(/*init_result=*/true, executor);
    backend->blockMatch();
    ASSERT_TRUE(initBackend(*backend, pool));
    backend->match(makeRequest(NULL_BLOCK_IDX), [](size_t, auto, bool success) { EXPECT_TRUE(success); });

    BoundedThread<void> shutdown([backend] { backend->shutdown(); });
    if (!backend->waitUntilMatchEntered()) {
        backend->releaseMatch();
        if (shutdown.waitFor(std::chrono::seconds(5)) == std::future_status::ready) {
            shutdown.get();
        }
        FAIL() << "accepted match did not enter while shutdown was stopping";
    }

    BoundedThread<bool> stopping_write(
        [backend, block] { return backend->write(backend->prepareWrite(makeRequest(block))); });
    if (stopping_write.waitFor(std::chrono::seconds(5)) != std::future_status::ready) {
        backend->releaseMatch();
        if (shutdown.waitFor(std::chrono::seconds(5)) == std::future_status::ready) {
            shutdown.get();
        }
        executor->runAll();
        if (stopping_write.waitFor(std::chrono::seconds(5)) == std::future_status::ready) {
            (void)stopping_write.get();
        }
        FAIL() << "asynchronous submission blocked while backend was stopping";
    }
    EXPECT_FALSE(stopping_write.get());
    EXPECT_EQ(pool->refCount(block), 1u);
    backend->releaseMatch();
    if (shutdown.waitFor(std::chrono::seconds(5)) != std::future_status::ready) {
        FAIL() << "shutdown did not finish after releasing the accepted match";
    }
    shutdown.get();
    EXPECT_TRUE(backend->shutdownCalled());
    pool->decRef(block);
}

TEST(StorageBackendTest, AsyncWriteFromOwnCallbacksQueuesWithoutBlocking) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    for (bool read : {false, true}) {
        bool called = false;
        auto done   = [&](bool success) {
            EXPECT_TRUE(success);
            EXPECT_TRUE(backend.write(backend.prepareWrite(makeRequest(block))));
            called = true;
        };
        if (read) {
            backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) { done(error.ok()); });
        } else {
            backend.match(makeRequest(NULL_BLOCK_IDX), [&](size_t, auto, bool success) { done(success); });
        }
        ASSERT_TRUE(executor->runNext());
        EXPECT_TRUE(called);
        EXPECT_EQ(executor->pendingCount(), 1u);
        EXPECT_EQ(pool->refCount(block), 2u);
        EXPECT_EQ(executor->runAll(), 1u);
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    backend.shutdown();
    pool->decRef(block);
}

TEST(StorageBackendTest, DuplicateExecutorInvocationCompletesExactlyOnce) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto executor = std::make_shared<HoldingExecutor>();
    executor->setDuplicate(true);
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    size_t completions = 0;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++completions;
    });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(completions, 1u);
    EXPECT_EQ(backend.readCalls(), 1u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
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
    EXPECT_FALSE(backend.shutdownCalled());

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
    EXPECT_TRUE(backend.shutdownCalled());

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
        } catch (...) { rejected = true; }
    });
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(rejected);
    backend.shutdown();
}

TEST(StorageBackendTest, ReadAfterShutdownReleasesPin) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    TestBackend backend;
    ASSERT_TRUE(backend.init(makeTopology(), {{"default", pool}}, [](int, const std::string&, int) {
        return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}};
    }));
    backend.shutdown();

    bool success = true;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) { success = error.ok(); });
    EXPECT_FALSE(success);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
}

TEST(StorageBackendTest, UnsubmittedWriteTasksReleasePinsAcrossShutdown) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    TestBackend backend;
    ASSERT_TRUE(backend.init(makeTopology(), {{"default", pool}}, [](int, const std::string&, int) {
        return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}};
    }));

    auto prepared_before_shutdown = backend.prepareWrite(makeRequest(block));
    EXPECT_EQ(pool->refCount(block), 2u);
    backend.shutdown();
    prepared_before_shutdown = {};
    EXPECT_EQ(pool->refCount(block), 1u);

    auto prepared_after_shutdown = backend.prepareWrite(makeRequest(block));
    EXPECT_EQ(pool->refCount(block), 2u);
    prepared_after_shutdown = {};
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
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
    EXPECT_EQ(backend.poolTags().front(), "default");
    EXPECT_EQ(backend.initResolvedAddress(), reinterpret_cast<void*>(1));
    backend.shutdown();
}

TEST(StorageBackendTest, InitPropagatesDerivedFailure) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend(/*init_result=*/false);

    EXPECT_FALSE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
    EXPECT_FALSE(initBackend(backend, pool));
    EXPECT_EQ(backend.initCalls(), 1u);
}

TEST(StorageBackendTest, InvalidArgumentsDoNotConsumeInitializationAttempt) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    const auto  resolver = [](int, const std::string&, int) { return std::vector<BlockInfo>{}; };
    EXPECT_ANY_THROW(backend.init(nullptr, {}, resolver));
    EXPECT_ANY_THROW(backend.init(makeTopology(), {}, resolver));
    EXPECT_ANY_THROW(backend.init(makeTopology(), {{"default", nullptr}}, resolver));
    EXPECT_EQ(backend.initCalls(), 0u);
    ASSERT_TRUE(initBackend(backend, pool));
    backend.shutdown();
}

TEST(StorageBackendTest, FailedDerivedInitReleasesUnusedExecutorBinding) {
    auto        pool     = std::make_shared<TestBlockPool>();
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend failed(false, executor);
    EXPECT_FALSE(initBackend(failed, pool));
    TestBackend other(true, executor);
    ASSERT_TRUE(initBackend(other, pool));
    other.shutdown();
}

TEST(StorageBackendTest, WritePinsEachPhysicalBlockOnceUntilCompletion) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));

    auto task = backend.prepareWrite(makeRequest(block, 2));
    EXPECT_EQ(pool->refCount(block), 2u);
    backend.write(std::move(task));
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_EQ(backend.writeHandleCount(), 2u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, IndependentPoolsPinSameBlockIdSeparately) {
    auto       first  = std::make_shared<TestBlockPool>();
    auto       second = std::make_shared<TestBlockPool>();
    const auto block  = first->malloc().value();
    ASSERT_EQ(second->malloc().value(), block);
    first->incRef(block);
    second->incRef(block);
    TestBackend backend;
    ASSERT_TRUE(backend.init(
        makeSharedPoolTopology(), {{"group_1", second}, {"group_0", first}}, [](int, const std::string&, int) {
            return std::vector<BlockInfo>{{false, 0, 0, reinterpret_cast<void*>(1), 16}};
        }));
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1, 2}),
                           {{{"group_0", block}, {"group_1", block}}, {{"group_0", block}}}};
    auto           task = backend.prepareWrite(request);
    EXPECT_EQ(first->refCount(block), 2u);
    EXPECT_EQ(second->refCount(block), 2u);
    task = {};
    EXPECT_EQ(first->refCount(block), 1u);
    EXPECT_EQ(second->refCount(block), 1u);
    first->decRef(block);
    second->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, UninitializedBackendRejectsTaskAndReleasesPinnedSource) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    TestBackend initialized_backend;
    ASSERT_TRUE(initBackend(initialized_backend, pool));
    auto task = initialized_backend.prepareWrite(makeRequest(block));
    EXPECT_EQ(pool->refCount(block), 2u);

    TestBackend uninitialized_backend;
    EXPECT_ANY_THROW(uninitialized_backend.write(std::move(task)));
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    initialized_backend.shutdown();
}

TEST(StorageBackendTest, ReadPinsTargetsUntilCompletion) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto        executor = std::make_shared<HoldingExecutor>();
    TestBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    bool completed = false;

    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        completed = true;
    });
    EXPECT_FALSE(completed);
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(completed);
    EXPECT_EQ(backend.readCalls(), 1u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    backend.shutdown();
}

TEST(StorageBackendTest, UnknownReadFreezesBeforeCallbackAndReleasesTaskState) {
    for (const auto reason : {StorageQuarantineReason::SDK_TIMEOUT,
                              StorageQuarantineReason::RPC_TIMEOUT,
                              StorageQuarantineReason::RPC_ERROR,
                              StorageQuarantineReason::INVALID_RESPONSE}) {
        auto pool  = std::make_shared<TestBlockPool>();
        auto block = pool->malloc().value();
        pool->incRef(block);
        auto                       executor = std::make_shared<HoldingExecutor>();
        QuarantineObservingBackend backend(/*init_result=*/true, executor);
        ASSERT_TRUE(initBackend(backend, pool));
        const bool timeout =
            reason == StorageQuarantineReason::SDK_TIMEOUT || reason == StorageQuarantineReason::RPC_TIMEOUT;
        backend.read_action = [reason, timeout] {
            if (timeout) {
                throw StorageOperationTimeout("transfer timed out", reason);
            }
            throw StorageOperationCompletionUnknown("transfer completion unknown", reason);
        };
        auto                               request      = makeRequest(block, /*key_count=*/2);
        std::weak_ptr<const CacheKeysType> request_keys = request.keys;
        bool                               completed    = false;
        backend.read(std::move(request), nullptr, [&](ErrorInfo error) {
            EXPECT_EQ(error.code(), timeout ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION);
            EXPECT_TRUE(pool->isFrozen(block));
            EXPECT_EQ(pool->treeRefCount(block), 1u);
            EXPECT_EQ(pool->refCount(block), 2u);  // The request and the permanent tree hold.
            completed = true;
        });
        EXPECT_EQ(pool->refCount(block), 2u);
        EXPECT_FALSE(pool->isFrozen(block));
        ASSERT_EQ(executor->runAll(), 1u);
        EXPECT_TRUE(completed);
        EXPECT_TRUE(request_keys.expired());
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
        EXPECT_EQ(backend.quarantine_events, 1u);
        EXPECT_FALSE(backend.last_quarantine_is_write);
        EXPECT_EQ(backend.last_quarantine_reason, reason);
        EXPECT_EQ(backend.last_quarantine_blocks, 1u);
        EXPECT_EQ(backend.quarantined_tasks, 1u);
        EXPECT_EQ(backend.quarantined_blocks, 1u);

        pool->decRef(block);
        backend.shutdown();
        EXPECT_EQ(backend.quarantined_tasks, 0u);
        EXPECT_EQ(backend.quarantined_blocks, 0u);
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_TRUE(pool->isFrozen(block));
        EXPECT_EQ(pool->refCount(block), 1u);
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    }
}

TEST(StorageBackendTest, RepeatedUnknownWritesKeepFrozenReferencesAcrossShutdown) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto                       executor = std::make_shared<HoldingExecutor>();
    QuarantineObservingBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    backend.write_action = [] {
        throw StorageOperationCompletionUnknown("write completion unknown", StorageQuarantineReason::RPC_ERROR);
    };
    for (size_t task_index = 1; task_index <= 2; ++task_index) {
        auto                               request      = makeRequest(block, /*key_count=*/2);
        std::weak_ptr<const CacheKeysType> request_keys = request.keys;
        ASSERT_TRUE(backend.write(backend.prepareWrite(std::move(request))));
        ASSERT_EQ(executor->runAll(), 1u);
        EXPECT_TRUE(request_keys.expired());
        EXPECT_EQ(pool->refCount(block), 2u);
        EXPECT_EQ(pool->treeRefCount(block), task_index);
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
        EXPECT_EQ(backend.quarantine_events, task_index);
        EXPECT_TRUE(backend.last_quarantine_is_write);
        EXPECT_EQ(backend.last_quarantine_blocks, 1u);
        EXPECT_EQ(backend.quarantined_tasks, task_index);
        EXPECT_EQ(backend.quarantined_blocks, task_index);
    }
    // BTC can freeze the same block without changing the physical-block metric.
    pool->freezeBlock(block);
    EXPECT_EQ(pool->treeRefCount(block), 3u);
    backend.shutdown();
    EXPECT_EQ(backend.quarantined_tasks, 0u);
    EXPECT_EQ(backend.quarantined_blocks, 0u);
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(pool->treeRefCount(block), 3u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    pool->decRef(block);
    EXPECT_EQ(pool->refCount(block), 1u);
}

TEST(StorageBackendTest, CompletedReadAndWriteFailuresReleasePinsWithoutFreezing) {
    auto pool  = std::make_shared<TestBlockPool>();
    auto block = pool->malloc().value();
    pool->incRef(block);
    auto                       executor = std::make_shared<HoldingExecutor>();
    QuarantineObservingBackend backend(/*init_result=*/true, executor);
    ASSERT_TRUE(initBackend(backend, pool));
    backend.failNextRead();
    bool completed = false;
    backend.read(makeRequest(block), nullptr, [&](ErrorInfo error) {
        EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(pool->refCount(block), 1u);
        completed = true;
    });
    ASSERT_EQ(executor->runAll(), 1u);
    EXPECT_TRUE(completed);
    backend.failNextWrite();
    ASSERT_TRUE(backend.write(backend.prepareWrite(makeRequest(block))));
    ASSERT_EQ(executor->runAll(), 1u);
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_EQ(pool->treeRefCount(block), 0u);
    EXPECT_EQ(backend.quarantine_events, 0u);
    backend.shutdown();
    pool->decRef(block);
    EXPECT_FALSE(pool->isAllocated(block));
}

TEST(StorageBackendTest, ResolvesGpuBufferAndGroupMetadataFromBoundResources) {
    auto        pool = std::make_shared<TestBlockPool>();
    TestBackend backend;
    ASSERT_TRUE(initBackend(backend, pool));

    EXPECT_EQ(backend.poolTags().front(), "default");
    const auto buffers = backend.resolve(0, "default", 3);
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(buffers.front().addr, reinterpret_cast<void*>(4));
    backend.shutdown();
}

}  // namespace
}  // namespace rtp_llm
