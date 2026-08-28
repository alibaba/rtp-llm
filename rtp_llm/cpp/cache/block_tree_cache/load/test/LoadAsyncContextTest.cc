#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

struct TestMatchMeta: StorageBackendMatchMeta {
    size_t remote_version{0};
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
        config->pool_name            = "load_context_test";
        config->physical_block_count = 4;
        MemoryLayoutConfig layout;
        layout.block_num                = 4;
        layout.layer_num                = 1;
        layout.kv_block_stride_bytes    = 16;
        layout.kv_block_pool_size_bytes = 64;
        config->memory_layouts.push_back(layout);
        return config;
    }
};

class ManualExecutor: public StorageBackendExecutor {
public:
    bool start() override {
        return true;
    }
    bool submit(Task task) override {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));
        return true;
    }
    void shutdown() noexcept override {
        runAll();
    }

    size_t pendingCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    void runOne() {
        Task task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
    }

private:
    void runAll() {
        while (pendingCount() != 0) {
            runOne();
        }
    }

    std::mutex       mutex_;
    std::deque<Task> tasks_;
};

class ManualBackend: public StorageBackend {
public:
    ManualBackend(): ManualBackend(std::make_shared<ManualExecutor>()) {}

    ~ManualBackend() override {
        shutdown();
    }

    void completeMatch(size_t keys, std::shared_ptr<StorageBackendMatchMeta> match_meta = nullptr) {
        pending_match_ = {keys, std::move(match_meta)};
        executor_->runOne();
    }
    void completeRead() {
        executor_->runOne();
    }
    void failNextMatch() {
        fail_match_ = true;
    }
    void failNextRead() {
        fail_read_ = true;
    }
    bool readPending() const {
        return executor_->pendingCount() != 0;
    }
    const CacheKeysType& readKeys() const {
        return read_keys_;
    }
    const std::vector<size_t>& readHandleCounts() const {
        return read_handle_counts_;
    }
    const std::vector<std::vector<size_t>>& readGroupIds() const {
        return read_group_ids_;
    }
    const CacheKeysType& matchKeys() const {
        return match_keys_;
    }
    size_t matchLocalBlocks() const {
        return match_local_blocks_;
    }
    const std::shared_ptr<TestMatchMeta>& readMatchMeta() const {
        return read_match_meta_;
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        if (fail_match_) {
            fail_match_ = false;
            throw std::runtime_error("match failed");
        }
        match_keys_         = *request.keys;
        match_local_blocks_ = request.local_matched_blocks_num;
        return pending_match_;
    }
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) override {
        if (fail_read_) {
            fail_read_ = false;
            throw std::runtime_error("read failed");
        }
        read_match_meta_ = std::dynamic_pointer_cast<TestMatchMeta>(match_meta);
        read_keys_       = *request.keys;
        for (const auto& key_handles : request.handles) {
            read_handle_counts_.push_back(key_handles.size());
            std::vector<size_t> group_ids;
            for (const auto& handle : key_handles) {
                group_ids.push_back(handle.group_id);
            }
            read_group_ids_.push_back(std::move(group_ids));
        }
    }
    void writeImpl(const StorageRequest&) override {}

private:
    explicit ManualBackend(std::shared_ptr<ManualExecutor> executor):
        StorageBackend(executor), executor_(std::move(executor)) {}

    std::shared_ptr<ManualExecutor>  executor_;
    StorageMatchResult               pending_match_;
    CacheKeysType                    read_keys_;
    CacheKeysType                    match_keys_;
    size_t                           match_local_blocks_{0};
    std::vector<size_t>              read_handle_counts_;
    std::vector<std::vector<size_t>> read_group_ids_;
    std::shared_ptr<TestMatchMeta>   read_match_meta_;
    bool                             fail_match_{false};
    bool                             fail_read_{false};
};

std::shared_ptr<const CacheTopology> makeTopology(std::vector<CacheGroupType> types = {CacheGroupType::FULL}) {
    std::vector<GroupBase>   groups;
    std::vector<std::string> tags;
    for (size_t group_id = 0; group_id < types.size(); ++group_id) {
        auto spec = std::make_shared<MHAKVCacheSpec>();
        spec->tag = "group_" + std::to_string(group_id);
        GroupBase group;
        group.tag                        = spec->tag;
        group.spec                       = std::move(spec);
        group.policy                     = defaultCacheGroupPolicy(types[group_id]);
        group.policy.enable_prefix_reuse = true;
        if (types[group_id] == CacheGroupType::SWA) {
            group.policy.sliding_window_size = 2;
        }
        group.layer_ids                 = {0};
        group.seq_size_per_block        = 1;
        group.kernel_seq_size_per_block = 1;
        tags.push_back(group.tag);
        groups.push_back(std::move(group));
    }
    return CacheTopology::create(std::move(groups), {{0, std::move(tags)}});
}

void initBackend(ManualBackend& backend, const DeviceBlockPoolPtr& pool) {
    RTP_LLM_CHECK(backend.init(makeTopology(), {pool}, [](int, int, int) { return std::vector<BlockInfo>{}; }));
}

void initBackend(ManualBackend&                         backend,
                 std::shared_ptr<const CacheTopology>   topology,
                 const std::vector<DeviceBlockPoolPtr>& pools) {
    RTP_LLM_CHECK(
        backend.init(std::move(topology), pools, [](int, int, int) { return std::vector<BlockInfo>{}; }));
}

StorageRequest makeRequest(size_t key_count) {
    CacheKeysType keys;
    keys.reserve(key_count);
    std::vector<std::vector<StorageBlockHandle>> handles;
    handles.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        keys.push_back(i + 1);
        handles.push_back({{0, NULL_BLOCK_IDX}});
    }
    return {std::make_shared<CacheKeysType>(std::move(keys)), std::move(handles)};
}

std::shared_ptr<LoadContextCoordinator> makeCoordinator(size_t& commits, size_t& aborts) {
    return std::make_shared<LoadContextCoordinator>(
        [&](const auto&) {
            ++commits;
            return true;
        },
        [&](auto&) { ++aborts; });
}

TEST(LoadAsyncContextTest, EmptyStorageMatchStillRunsDeferredAllocationAndCommit) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));
    size_t callbacks = 0;
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        ++callbacks;
        EXPECT_EQ(matched, 0u);
        return current.commit();
    });

    context->startBackendMatch();
    EXPECT_FALSE(context->done());
    backend->completeMatch(0);
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, OnDoneRunsOnceWhenCommittedTransferCompletes) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    TransferDescriptor descriptor;
    descriptor.source_tier = Tier::HOST;
    auto context = coordinator->create({descriptor}, {false}, 1);
    ASSERT_TRUE(coordinator->registerContext(context));

    size_t callback_count = 0;
    context->onDone([&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    ASSERT_TRUE(context->commit());
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeOne(true));
    EXPECT_TRUE(context->done());
    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, BackendMatchFailureAbortsWithoutRunningAllocatorCallback) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));
    size_t callbacks = 0;
    context->setMatchCallback([&](LoadAsyncContext&, size_t) {
        ++callbacks;
        return true;
    });

    backend->failNextMatch();
    context->startBackendMatch();
    backend->completeMatch(1);

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    EXPECT_EQ(callbacks, 0u);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, AllocatorCallbackPreservesRetryableCapacityStatus) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([](LoadAsyncContext&, size_t) {
        return LoadMatchResult{false, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED};
    });

    context->startBackendMatch();
    backend->completeMatch(1);

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, CoordinatorCommitFailurePublishesInternalStatusBeforeTerminalState) {
    size_t aborts = 0;
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return false; }, [&](LoadAsyncContext&) { ++aborts; });
    auto context = coordinator->create({}, {}, 0);
    ASSERT_TRUE(coordinator->registerContext(context));

    EXPECT_FALSE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    EXPECT_EQ(aborts, 0u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, ConcurrentCommitRunsCoordinatorCallbackOnceAndKeepsWinnerState) {
    using namespace std::chrono_literals;

    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered  = false;
    bool                    released = false;
    size_t                  commits  = 0;
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>&) {
            std::unique_lock<std::mutex> lock(mutex);
            ++commits;
            entered = true;
            cv.notify_all();
            cv.wait(lock, [&] { return released; });
            return true;
        },
        [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors;
    descriptors.emplace_back(nullptr,
                             /*group_set_id=*/0,
                             /*path_index=*/0,
                             Tier::HOST,
                             Tier::DEVICE,
                             BlockIndicesType{1});
    auto context = coordinator->create(std::move(descriptors), {false}, 1);
    ASSERT_TRUE(coordinator->registerContext(context));

    auto winner = std::async(std::launch::async, [&] { return context->commit(); });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return entered; });
    }
    auto loser = std::async(std::launch::async, [&] { return context->commit(); });
    ASSERT_EQ(loser.wait_for(100ms), std::future_status::ready);
    EXPECT_FALSE(loser.get());
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
    }
    cv.notify_all();

    EXPECT_TRUE(winner.get());
    EXPECT_EQ(commits, 1u);
    EXPECT_TRUE(context->completeOne(true));
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, ImmediateTransferFailureAfterSuccessfulCommitKeepsFallbackStatus) {
    size_t commits = 0;
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>& context) {
            ++commits;
            EXPECT_TRUE(context->completeOne(false));
            return true;
        },
        [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors;
    descriptors.emplace_back(nullptr,
                             /*group_set_id=*/0,
                             /*path_index=*/0,
                             Tier::HOST,
                             Tier::DEVICE,
                             BlockIndicesType{1});
    auto context = coordinator->create(std::move(descriptors), {false}, 1);
    ASSERT_TRUE(coordinator->registerContext(context));

    EXPECT_TRUE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    EXPECT_EQ(commits, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, BackendReadFailureMarksCommittedContextFailedAndReleasesPin) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });

    context->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    EXPECT_EQ(pool->refCount(block), 2u);
    backend->failNextRead();
    backend->completeRead();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, MatchedKeyExposesAllOfItsHandles) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1, 2}),
                           {{{0, NULL_BLOCK_IDX}, {0, NULL_BLOCK_IDX}}, {{0, NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, 0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 1u);
        EXPECT_EQ(current.backendHandles().size(), 1u);
        EXPECT_EQ(current.backendHandles().front().size(), 2u);
        current.setBackendTargetBlock(0, 0, block);
        current.setBackendTargetBlock(0, 1, block);
        return current.commit();
    });

    context->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    backend->completeRead();
    EXPECT_EQ(backend->readKeys(), (CacheKeysType{1}));
    EXPECT_EQ(backend->readHandleCounts(), (std::vector<size_t>{2}));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, MatchedKeysKeepReuseShapeAndForwardDerivedMatchMeta) {
    size_t                                      commits     = 0;
    size_t                                      aborts      = 0;
    auto                                        coordinator = makeCoordinator(commits, aborts);
    auto                                        backend     = std::make_shared<ManualBackend>();
    std::vector<std::shared_ptr<TestBlockPool>> pools;
    std::vector<DeviceBlockPoolPtr>             bound_pools;
    std::vector<BlockIdxType>                   blocks;
    for (size_t group_id = 0; group_id < 3; ++group_id) {
        auto pool  = std::make_shared<TestBlockPool>();
        auto block = pool->malloc().value();
        pool->incRef(block);
        pools.push_back(pool);
        bound_pools.push_back(pool);
        blocks.push_back(block);
    }
    initBackend(
        *backend, makeTopology({CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::SWA}), bound_pools);

    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1, 2, 3, 4}),
                           std::vector<std::vector<StorageBlockHandle>>(4)};
    for (auto& handles : request.handles) {
        handles = {{0, NULL_BLOCK_IDX}, {1, NULL_BLOCK_IDX}, {2, NULL_BLOCK_IDX}};
    }
    auto context = coordinator->create({}, {}, 0, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 4u);
        const auto& handles = current.backendHandles();
        EXPECT_EQ(handles.size(), 4u);
        EXPECT_EQ(handles[0].size(), 1u);
        EXPECT_EQ(handles[1].size(), 1u);
        EXPECT_EQ(handles[2].size(), 2u);
        EXPECT_EQ(handles[3].size(), 3u);
        for (size_t key_index = 0; key_index < handles.size(); ++key_index) {
            for (size_t handle_index = 0; handle_index < handles[key_index].size(); ++handle_index) {
                current.setBackendTargetBlock(
                    key_index, handle_index, blocks[handles[key_index][handle_index].group_id]);
            }
        }
        return current.commit();
    });

    context->startBackendMatch();
    auto match_meta            = std::make_shared<TestMatchMeta>();
    match_meta->remote_version = 17;
    backend->completeMatch(4, match_meta);
    ASSERT_TRUE(backend->readPending());
    backend->completeRead();
    ASSERT_EQ(backend->readMatchMeta(), match_meta);
    EXPECT_EQ(backend->readMatchMeta()->remote_version, 17u);
    EXPECT_EQ(backend->readHandleCounts(), (std::vector<size_t>{1, 1, 2, 3}));
    EXPECT_EQ(backend->readGroupIds(), (std::vector<std::vector<size_t>>{{0}, {0}, {0, 2}, {0, 1, 2}}));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    for (size_t group_id = 0; group_id < pools.size(); ++group_id) {
        pools[group_id]->decRef(blocks[group_id]);
    }
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, FullKeyMatchKeepsLocalPrefixVisibleAndReadsOnlyRemoteRows) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   first       = pool->malloc().value();
    auto   second      = pool->malloc().value();
    pool->incRef(first);
    pool->incRef(second);
    initBackend(*backend, pool);

    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{10, 20, 30, 40}),
                           std::vector<std::vector<StorageBlockHandle>>(4, {{0, NULL_BLOCK_IDX}}),
                           /*local_matched_blocks_num=*/2};
    auto           context = coordinator->create({}, {}, 2, backend, std::move(request));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 4u);
        const auto& handles = current.backendHandles();
        EXPECT_EQ(handles.size(), 4u);
        EXPECT_TRUE(handles[0].empty());
        EXPECT_TRUE(handles[1].empty());
        EXPECT_EQ(handles[2].size(), 1u);
        EXPECT_EQ(handles[3].size(), 1u);
        current.setBackendTargetBlock(2, 0, first);
        current.setBackendTargetBlock(3, 0, second);
        return current.commit();
    });

    context->startBackendMatch();
    backend->completeMatch(4);
    EXPECT_EQ(backend->matchKeys(), (CacheKeysType{10, 20, 30, 40}));
    EXPECT_EQ(backend->matchLocalBlocks(), 2u);
    ASSERT_TRUE(backend->readPending());
    backend->completeRead();
    EXPECT_EQ(backend->readKeys(), (CacheKeysType{10, 20, 30, 40}));
    EXPECT_EQ(backend->readHandleCounts(), (std::vector<size_t>{0, 0, 1, 1}));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->matchedBlocks(), 4u);
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    pool->decRef(first);
    pool->decRef(second);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, LocalAndStorageReadsMustBothComplete) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);

    TransferDescriptor local;
    local.source_tier = Tier::HOST;
    auto context      = coordinator->create({local}, {false}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 1u);
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });

    context->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_TRUE(context->completeOne(true));
    EXPECT_FALSE(context->done());
    backend->completeRead();
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->matchedBlocks(), 1u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, PendingContextAbortsOnDestruction) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    {
        auto context = coordinator->create({TransferDescriptor{}}, {false}, 1);
        ASSERT_TRUE(coordinator->registerContext(context));
    }
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, AbortWaitsForDeferredAllocatorCallbackAndLosesAfterCommit) {
    using namespace std::chrono_literals;
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));

    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered  = false;
    bool                    released = false;
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });
    context->startBackendMatch();
    auto completion = std::async(std::launch::async, [&] { backend->completeMatch(1); });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return entered; });
    }
    std::future<bool> abort = std::async(std::launch::async, [&] { return context->abortPending(); });
    EXPECT_EQ(abort.wait_for(50ms), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
    }
    cv.notify_all();
    completion.get();
    EXPECT_FALSE(abort.get());
    ASSERT_TRUE(backend->readPending());
    backend->completeRead();
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, AbortBeforeDeferredAllocatorCallbackPreventsCommit) {
    size_t                                  commits     = 0;
    size_t                                  aborts      = 0;
    std::shared_ptr<LoadContextCoordinator> coordinator = makeCoordinator(commits, aborts);
    std::shared_ptr<ManualBackend>          backend     = std::make_shared<ManualBackend>();
    std::shared_ptr<TestBlockPool>          pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    std::shared_ptr<LoadAsyncContext> context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));
    size_t callbacks = 0;
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        ++callbacks;
        return current.commit();
    });

    context->startBackendMatch();
    EXPECT_TRUE(context->abortPending());
    backend->completeMatch(1);

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(callbacks, 0u);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, CoordinatorShutdownWaitsForDeferredAllocatorCallback) {
    using namespace std::chrono_literals;
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, 0, backend, makeRequest(1));
    ASSERT_TRUE(coordinator->registerContext(context));

    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered  = false;
    bool                    released = false;
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
        return current.commit();
    });
    context->startBackendMatch();
    auto completion = std::async(std::launch::async, [&] { backend->completeMatch(1); });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return entered; });
    }
    auto shutdown = std::async(std::launch::async, [&] { coordinator->shutdown(); });
    EXPECT_EQ(shutdown.wait_for(50ms), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
    }
    cv.notify_all();
    completion.get();
    shutdown.get();
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
}

}  // namespace
}  // namespace rtp_llm
