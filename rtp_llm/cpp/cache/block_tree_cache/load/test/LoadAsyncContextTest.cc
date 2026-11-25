#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"

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
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"

namespace rtp_llm {
namespace {

std::shared_ptr<const CacheConfig> wrapTopologyAsConfig(std::shared_ptr<const CacheTopology> topology) {
    auto config                            = std::make_shared<CacheConfig>();
    config->seq_size_per_block             = topology->groups().front().seqSizePerBlock();
    config->setTopology(0, topology->groups(), topology->layers());
    return config;
}

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
    bool submit(Task task, BlockTreeTaskClass = BlockTreeTaskClass::BACKGROUND) override {
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
    explicit ManualBackend(std::chrono::milliseconds timeout = std::chrono::seconds(10)):
        ManualBackend(std::make_shared<ManualExecutor>(), timeout) {}

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
    void timeoutNextRead() {
        timeout_read_ = true;
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
    const std::vector<std::vector<std::string>>& readGroupTags() const {
        return read_group_tags_;
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
    TransferTask::Clock::time_point matchDeadline() const {
        return match_deadline_;
    }
    TransferTask::Clock::time_point readDeadline() const {
        return read_deadline_;
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
        match_deadline_     = request.transfer_task->deadline();
        return pending_match_;
    }
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) override {
        if (timeout_read_) {
            timeout_read_ = false;
            throw StorageOperationTimeout("read timed out", StorageQuarantineReason::RPC_TIMEOUT);
        }
        if (fail_read_) {
            fail_read_ = false;
            throw std::runtime_error("read failed");
        }
        read_match_meta_ = std::dynamic_pointer_cast<TestMatchMeta>(match_meta);
        read_deadline_   = request.transfer_task->deadline();
        read_keys_       = *request.keys;
        for (const auto& key_handles : request.handles) {
            read_handle_counts_.push_back(key_handles.size());
            std::vector<std::string> group_tags;
            for (const auto& handle : key_handles) {
                group_tags.push_back(handle.tag);
            }
            read_group_tags_.push_back(std::move(group_tags));
        }
    }
    void writeImpl(const StorageRequest&) override {}

private:
    ManualBackend(std::shared_ptr<ManualExecutor> executor, std::chrono::milliseconds timeout):
        StorageBackend(executor, nullptr, -1, timeout), executor_(std::move(executor)) {}

    std::shared_ptr<ManualExecutor>       executor_;
    StorageMatchResult                    pending_match_;
    CacheKeysType                         read_keys_;
    CacheKeysType                         match_keys_;
    size_t                                match_local_blocks_{0};
    std::vector<size_t>                   read_handle_counts_;
    std::vector<std::vector<std::string>> read_group_tags_;
    std::shared_ptr<TestMatchMeta>        read_match_meta_;
    TransferTask::Clock::time_point       match_deadline_;
    TransferTask::Clock::time_point       read_deadline_;
    bool                                  fail_match_{false};
    bool                                  fail_read_{false};
    bool                                  timeout_read_{false};
};

std::shared_ptr<const CacheTopology> makeTopology(std::vector<CacheGroupType> types = {CacheGroupType::FULL}) {
    std::vector<GroupBase>   groups;
    std::vector<std::string> tags;
    for (size_t group_id = 0; group_id < types.size(); ++group_id) {
        auto      spec = std::make_shared<MHAKVCacheSpec>("group_" + std::to_string(group_id), 1, 1, 1);
        GroupBase group;
        group.tag                        = spec->tag;
        group.spec                       = std::move(spec);
        group.policy                     = defaultCacheGroupPolicy(types[group_id]);
        group.policy.enable_prefix_reuse = true;
        if (types[group_id] == CacheGroupType::SWA) {
            group.policy.sliding_window_size = 2;
        }
        tags.push_back(group.tag);
        groups.push_back(std::move(group));
    }
    return CacheTopology::create(std::move(groups), {{0, std::move(tags)}});
}

void initBackend(ManualBackend& backend, const DeviceBlockPoolPtr& pool) {
    RTP_LLM_CHECK(backend.init(wrapTopologyAsConfig(makeTopology()),
                               {{"group_0", pool}},
                               [](int, const std::string&, int) { return std::vector<BlockInfo>{}; }));
}

void initBackend(ManualBackend&                         backend,
                 std::shared_ptr<const CacheTopology>   topology,
                 const std::vector<DeviceBlockPoolPtr>& pools) {
    StorageBackend::PoolsByTag pools_by_tag;
    RTP_LLM_CHECK(topology->groupTags().size() == pools.size());
    for (size_t i = 0; i < pools.size(); ++i) {
        RTP_LLM_CHECK(pools_by_tag.emplace(topology->groupTags()[i], pools[i]).second);
    }
    RTP_LLM_CHECK(backend.init(wrapTopologyAsConfig(topology),
                               std::move(pools_by_tag),
                               [](int, const std::string&, int) { return std::vector<BlockInfo>{}; }));
}

StorageRequest makeRequest(size_t key_count) {
    CacheKeysType keys;
    keys.reserve(key_count);
    std::vector<std::vector<StorageBlockHandle>> handles;
    handles.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        keys.push_back(i + 1);
        handles.push_back({{"group_0", NULL_BLOCK_IDX}});
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
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
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
    size_t             commits     = 0;
    size_t             aborts      = 0;
    auto               coordinator = makeCoordinator(commits, aborts);
    TransferDescriptor descriptor;
    descriptor.source_tier = Tier::HOST;
    auto context           = coordinator->create({descriptor}, {false}, {}, 1);
    ASSERT_NE(context, nullptr);

    size_t callback_count = 0;
    context->onDone([&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    ASSERT_TRUE(context->commit());
    EXPECT_FALSE(context->done());
    EXPECT_EQ(context->completionTimeUs(), 0);
    EXPECT_TRUE(context->completeTransfers(1, true));
    EXPECT_TRUE(context->done());
    EXPECT_GT(context->completionTimeUs(), 0);
    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, ThrowingDoneCallbacksDoNotInterruptOtherSubscribers) {
    for (bool success : {false, true}) {
        SCOPED_TRACE(success);
        size_t             commits     = 0;
        size_t             aborts      = 0;
        auto               coordinator = makeCoordinator(commits, aborts);
        TransferDescriptor descriptor;
        descriptor.source_tier = Tier::HOST;
        auto context           = coordinator->create({descriptor}, {false}, {}, 1);
        ASSERT_NE(context, nullptr);
        size_t notifications = 0;
        auto   observe       = [&](ErrorInfo error) {
            EXPECT_EQ(error.ok(), success);
            EXPECT_TRUE(context->done());
            ++notifications;
        };
        auto throw_exception = [](ErrorInfo) { throw std::runtime_error("onDone failed"); };
        auto throw_unknown   = [](ErrorInfo) { throw 1; };
        context->onDone(throw_exception);
        context->onDone(observe);
        context->onDone(throw_unknown);
        context->onDone(observe);
        ASSERT_TRUE(context->commit());
        EXPECT_NO_THROW(EXPECT_TRUE(context->completeTransfers(1, success)));
        EXPECT_EQ(notifications, 2u);
        EXPECT_NO_THROW(context->onDone(throw_exception));
        EXPECT_NO_THROW(context->onDone(throw_unknown));
        context->onDone(observe);
        EXPECT_EQ(notifications, 3u);
        EXPECT_EQ(context->success(), success);
        coordinator->shutdown();
    }
}

TEST(LoadAsyncContextTest, EmptyAndDeviceOnlyContextsHaveNoNewLoads) {
    auto coordinator = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    auto empty       = coordinator->create({}, {}, {}, 0);
    EXPECT_TRUE(empty->newLoadIndices().empty());
    TransferDescriptor device_desc;
    device_desc.source_tier = Tier::DEVICE;
    auto device_only        = coordinator->create({device_desc}, {false}, {}, 1, 1, 1);
    EXPECT_TRUE(device_only->newLoadIndices().empty());
}

TEST(LoadAsyncContextTest, NewLoadIndicesExcludeJoinedLoadsAndSurviveTheirCompletion) {
    auto coordinator = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    auto joined      = block_tree_cache_test::makeLoadOperationForTest();
    TransferDescriptor device_desc;
    device_desc.source_tier = Tier::DEVICE;
    auto context            = coordinator->create({device_desc,
                                                   joined->descriptor(),
                                                   TransferDescriptor::hostToDevice(0, 2, {}),
                                                   TransferDescriptor::diskToDevice(0, 3, {})},
                                                  {false, true, false, false},
                                                  {nullptr, joined},
                                       4,
                                       1,
                                       3);
    EXPECT_EQ(context->newLoadIndices(), (std::vector<size_t>{2, 3}));
    auto join_only = coordinator->create({joined->descriptor()}, {true}, {joined}, 1);
    EXPECT_TRUE(join_only->newLoadIndices().empty());
    joined->complete(true);
    context->setTargetBlocks(2, {4});
    EXPECT_EQ(context->newLoadIndices(), (std::vector<size_t>{2, 3}));
    EXPECT_FALSE(context->done());
}

TEST(LoadAsyncContextTest, JoinWaitCompletesOnLastJoinedDependency) {
    size_t                                  commits     = 0;
    size_t                                  aborts      = 0;
    std::shared_ptr<LoadContextCoordinator> coordinator = makeCoordinator(commits, aborts);
    TransferDescriptor                      first;
    first.source_tier = Tier::HOST;
    TransferDescriptor second;
    second.source_tier = Tier::HOST;
    auto first_load    = block_tree_cache_test::makeLoadOperationForTest(first.source_tier, 1);
    auto second_load   = block_tree_cache_test::makeLoadOperationForTest(second.source_tier, 2);
    auto context       = coordinator->create({first, second}, {true, true}, {first_load, second_load}, 1);
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(context->commit());

    first_load->complete(true);
    EXPECT_FALSE(context->done());
    second_load->complete(true);
    EXPECT_TRUE(context->success());
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, SettlementBarrierDefersTerminalNotificationUntilSettled) {
    size_t                            ready_count    = 0;
    size_t                            callback_count = 0;
    std::shared_ptr<LoadAsyncContext> ready_context;
    auto                              coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>& context) {
            context->setSettlementReadyCallback([&](const std::shared_ptr<LoadAsyncContext>& ready) {
                ++ready_count;
                ready_context = ready;
            });
            return true;
        },
        [](LoadAsyncContext&) {});
    TransferDescriptor descriptor;
    descriptor.source_tier = Tier::HOST;
    auto context           = coordinator->create({descriptor}, {false}, {}, 1);
    ASSERT_NE(context, nullptr);
    context->onDone([&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    ASSERT_TRUE(context->commit());
    ASSERT_TRUE(context->completeTransfers(1, true));
    EXPECT_EQ(ready_count, 1u);
    EXPECT_EQ(ready_context, context);
    EXPECT_TRUE(context->aggregateSuccess());
    EXPECT_FALSE(context->done());
    EXPECT_EQ(callback_count, 0u);

    EXPECT_TRUE(context->settle(true));
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(context->settle(true));
    EXPECT_EQ(ready_count, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, CompletionBeforeSettlementRegistrationWaitsForCommit) {
    for (bool success : {false, true}) {
        SCOPED_TRACE(success);
        size_t ready_count    = 0;
        size_t callback_count = 0;
        auto   coordinator    = std::make_shared<LoadContextCoordinator>(
            [&](const std::shared_ptr<LoadAsyncContext>& context) {
                EXPECT_TRUE(context->completeTransfers(1, success));
                EXPECT_FALSE(context->done());
                EXPECT_EQ(callback_count, 0u);
                context->setSettlementReadyCallback([&](const auto& ready) {
                    ++ready_count;
                    EXPECT_TRUE(ready->settle(success));
                });
                EXPECT_EQ(ready_count, 0u);
                return true;
            },
            [](LoadAsyncContext&) {});
        TransferDescriptor descriptor;
        descriptor.source_tier = Tier::HOST;
        auto context           = coordinator->create({descriptor}, {false}, {}, 1);
        context->onDone([&](ErrorInfo error) {
            EXPECT_EQ(error.ok(), success);
            ++callback_count;
        });

        ASSERT_TRUE(context->commit());
        EXPECT_TRUE(context->done());
        EXPECT_EQ(context->success(), success);
        EXPECT_EQ(ready_count, 1u);
        EXPECT_EQ(callback_count, 1u);
        EXPECT_FALSE(context->settle(success));
        coordinator->shutdown();
    }
}

TEST(LoadAsyncContextTest, BatchTransferCompletionReachesSettlementBarrierOnce) {
    size_t ready_count = 0;
    auto   coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>& context) {
            context->setSettlementReadyCallback([&](const std::shared_ptr<LoadAsyncContext>&) { ++ready_count; });
            return true;
        },
        [](LoadAsyncContext&) {});
    TransferDescriptor first;
    first.source_tier = Tier::HOST;
    TransferDescriptor second;
    second.source_tier = Tier::DISK;
    auto context       = coordinator->create({first, second}, {false, false}, {}, 1);
    ASSERT_NE(context, nullptr);

    ASSERT_TRUE(context->commit());
    EXPECT_FALSE(context->completeTransfers(3, true));
    EXPECT_EQ(ready_count, 0u);
    EXPECT_TRUE(context->completeTransfers(2, true));
    EXPECT_EQ(ready_count, 1u);
    EXPECT_TRUE(context->aggregateSuccess());
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(context->completeTransfers(1, true));
    EXPECT_EQ(ready_count, 1u);

    EXPECT_TRUE(context->settle(true));
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, BackendMatchFailureRunsDeferredAllocationAndKeepsLocalPrefix) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    TransferDescriptor local;
    local.source_tier                = Tier::HOST;
    auto request                     = makeRequest(2);
    request.local_matched_blocks_num = 1;
    auto context = coordinator->create({local}, {false}, {}, 1, 0, 0, {}, backend, std::move(request));
    ASSERT_NE(context, nullptr);
    size_t callbacks = 0;
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        ++callbacks;
        EXPECT_EQ(matched, 1u);
        return current.commit();
    });

    backend->failNextMatch();
    context->startBackendMatch();
    backend->completeMatch(2);

    EXPECT_FALSE(backend->readPending());
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeTransfers(1, true, Tier::HOST));
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(context->errorInfo().ok());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    EXPECT_EQ(context->matchedBlocks(), 1u);
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(commits, 1u);
    EXPECT_EQ(aborts, 0u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, AllocatorCallbackPreservesRetryableCapacityStatus) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback(
        [](LoadAsyncContext&, size_t) { return LoadMatchResult{false, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED}; });

    context->startBackendMatch();
    backend->completeMatch(1);

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, CommitFailureNotifiesWithoutCallingAbort) {
    size_t aborts        = 0;
    size_t notifications = 0;
    auto   coordinator   = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>& context) {
            context->setSettlementReadyCallback([context](const auto&) {
                ADD_FAILURE() << "failed commit must not settle context " << context->contextId();
            });
            return false;
        },
        [&](LoadAsyncContext& context) {
            EXPECT_FALSE(context.done());
            ++aborts;
        });
    auto                            context = coordinator->create({}, {}, {}, 0);
    std::weak_ptr<LoadAsyncContext> weak    = context;
    context->onDone([&](ErrorInfo error) {
        EXPECT_EQ(aborts, 0u);
        EXPECT_FALSE(error.ok());
        EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
        ++notifications;
    });

    EXPECT_FALSE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->abortPending());
    EXPECT_EQ(aborts, 0u);
    EXPECT_EQ(notifications, 1u);
    context.reset();
    EXPECT_TRUE(weak.expired());
    coordinator->shutdown();
    EXPECT_EQ(aborts, 0u);
}

TEST(LoadAsyncContextTest, ConcurrentCommitRunsCoordinatorCallbackOnceAndKeepsWinnerState) {
    using namespace std::chrono_literals;

    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered     = false;
    bool                    released    = false;
    size_t                  commits     = 0;
    auto                    coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>&) {
            std::unique_lock<std::mutex> lock(mutex);
            ++commits;
            entered = true;
            cv.notify_all();
            const bool release_observed = cv.wait_for(lock, 10s, [&] { return released; });
            EXPECT_TRUE(release_observed);
            return release_observed;
        },
        [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors;
    descriptors.emplace_back(nullptr,
                             /*group_set_id=*/0,
                             /*path_index=*/0,
                             Tier::HOST,
                             Tier::DEVICE,
                             BlockIndicesType{1},
                             /*initial_owner=*/false);
    std::shared_ptr<LoadAsyncContext> context = coordinator->create(std::move(descriptors), {false}, {}, 1);
    ASSERT_NE(context, nullptr);

    std::future<bool> winner;
    std::future<bool> loser;
    // Release the callback before either async future is destroyed on an assertion failure.
    block_tree_cache_detail::ScopeRollback release_guard([&] {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        cv.notify_all();
    });
    winner = std::async(std::launch::async, [&] { return context->commit(); });
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return entered; }));
    }
    loser = std::async(std::launch::async, [&] { return context->commit(); });
    ASSERT_EQ(loser.wait_for(100ms), std::future_status::ready);
    EXPECT_FALSE(loser.get());
    release_guard.run();

    ASSERT_EQ(winner.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(winner.get());
    EXPECT_EQ(commits, 1u);
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, ImmediateTransferFailureAfterSuccessfulCommitKeepsFallbackStatus) {
    size_t commits     = 0;
    auto   coordinator = std::make_shared<LoadContextCoordinator>(
        [&](const std::shared_ptr<LoadAsyncContext>& context) {
            ++commits;
            EXPECT_TRUE(context->completeTransfers(1, false));
            return true;
        },
        [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors;
    descriptors.emplace_back(nullptr,
                             /*group_set_id=*/0,
                             /*path_index=*/0,
                             Tier::HOST,
                             Tier::DEVICE,
                             BlockIndicesType{1},
                             /*initial_owner=*/false);
    auto context = coordinator->create(std::move(descriptors), {false}, {}, 1);
    ASSERT_NE(context, nullptr);

    EXPECT_TRUE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    EXPECT_EQ(commits, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, BackendMatchClipsInvalidPrefixBeforeMaterialization) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    auto context = coordinator->create(
        {}, {}, {}, 0, 0, 0, [](size_t blocks) { return blocks <= 1 || blocks >= 4; }, backend, makeRequest(3));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 1u);
        EXPECT_EQ(current.backendHandles().size(), 1u);
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });
    context->startBackendMatch();
    backend->completeMatch(3);
    ASSERT_TRUE(backend->readPending());
    backend->completeRead();
    EXPECT_EQ(backend->readKeys().size(), 1u);
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->matchedBlocks(), 1u);
    EXPECT_EQ(commits, 1u);
    pool->decRef(block);
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
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
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

TEST(LoadAsyncContextTest, BackendTimeoutSurvivesLocalTransferFailureAndSettlement) {
    for (bool backend_first : {false, true}) {
        SCOPED_TRACE(backend_first);
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
        auto context      = coordinator->create({local}, {false}, {}, 0, 0, 0, {}, backend, makeRequest(1));
        ASSERT_NE(context, nullptr);
        context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
            current.setBackendTargetBlock(0, 0, block);
            return current.commit();
        });
        size_t callbacks = 0;
        context->onDone([&](ErrorInfo error) {
            EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
            EXPECT_NE(error.ToString().find("read timed out"), std::string::npos);
            EXPECT_TRUE(pool->isFrozen(block));
            ++callbacks;
        });
        context->setSettlementReadyCallback([](const auto& ready) {
            EXPECT_FALSE(ready->aggregateSuccess());
            EXPECT_TRUE(ready->settle(false));
        });
        context->startBackendMatch();
        backend->completeMatch(1);
        backend->timeoutNextRead();
        if (backend_first) {
            backend->completeRead();
        } else {
            ASSERT_TRUE(context->completeTransfers(1, false));
        }
        EXPECT_FALSE(context->done());
        EXPECT_EQ(callbacks, 0u);
        if (backend_first) {
            ASSERT_TRUE(context->completeTransfers(1, false));
        } else {
            backend->completeRead();
        }

        EXPECT_TRUE(context->done());
        EXPECT_FALSE(context->success());
        EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_EQ(callbacks, 1u);
        context->onDone([&](ErrorInfo error) {
            EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
            EXPECT_NE(error.ToString().find("read timed out"), std::string::npos);
            ++callbacks;
        });
        EXPECT_EQ(callbacks, 2u);
        pool->decRef(block);
        coordinator->shutdown();
    }
}

TEST(LoadAsyncContextTest, DuplicateHandleTagIsRejectedBeforeMatch) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>();
    auto   pool        = std::make_shared<TestBlockPool>();
    auto   block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    StorageRequest request{std::make_shared<CacheKeysType>(CacheKeysType{1, 2}),
                           {{{"group_0", NULL_BLOCK_IDX}, {"group_0", NULL_BLOCK_IDX}}, {{"group_0", NULL_BLOCK_IDX}}}};
    auto           context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, std::move(request));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 1u);
        EXPECT_EQ(current.backendHandles().size(), 1u);
        EXPECT_EQ(current.backendHandles().front().size(), 2u);
        current.setBackendTargetBlock(0, 0, block);
        current.setBackendTargetBlock(0, 1, block);
        return current.commit();
    });

    EXPECT_ANY_THROW(context->startBackendMatch());
    EXPECT_FALSE(backend->readPending());
    EXPECT_EQ(commits, 0u);
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
        handles = {{"group_0", NULL_BLOCK_IDX}, {"group_1", NULL_BLOCK_IDX}, {"group_2", NULL_BLOCK_IDX}};
    }
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, std::move(request));
    ASSERT_NE(context, nullptr);
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
                const auto group_id = static_cast<size_t>(handles[key_index][handle_index].tag.back() - '0');
                current.setBackendTargetBlock(key_index, handle_index, blocks[group_id]);
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
    EXPECT_EQ(backend->readGroupTags(),
              (std::vector<std::vector<std::string>>{
                  {"group_0"}, {"group_0"}, {"group_0", "group_2"}, {"group_0", "group_1", "group_2"}}));
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
                           std::vector<std::vector<StorageBlockHandle>>(4, {{"group_0", NULL_BLOCK_IDX}}),
                           /*local_matched_blocks_num=*/2};
    auto           context = coordinator->create({}, {}, {}, 2, 0, 0, {}, backend, std::move(request));
    ASSERT_NE(context, nullptr);
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
    auto context      = coordinator->create({local}, {false}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 1u);
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });

    context->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_TRUE(context->completeTransfers(1, true));
    EXPECT_FALSE(context->done());
    backend->completeRead();
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->matchedBlocks(), 1u);
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, QueuedMatchExpirationSkipsAllocationAndNotifiesOnce) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   backend     = std::make_shared<ManualBackend>(std::chrono::milliseconds(100));
    auto   pool        = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    size_t allocations = 0;
    context->setMatchCallback([&](LoadAsyncContext&, size_t) {
        ++allocations;
        return true;
    });
    size_t notifications = 0;
    context->onDone([&](ErrorInfo error) {
        ++notifications;
        EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
    });
    context->startBackendMatch();
    ASSERT_TRUE(backend->readPending());
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    EXPECT_TRUE(context->done());
    backend->completeMatch(1);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(allocations, 0u);
    EXPECT_EQ(aborts, 1u);
    EXPECT_EQ(notifications, 1u);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, WaitDoneExitsAtDeadlineBeforeQueuedReadRetires) {
    size_t     commits     = 0;
    size_t     aborts      = 0;
    auto       coordinator = makeCoordinator(commits, aborts);
    auto       backend     = std::make_shared<ManualBackend>(std::chrono::milliseconds(100));
    auto       pool        = std::make_shared<TestBlockPool>();
    const auto block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });
    size_t notifications = 0;
    context->onDone([&](ErrorInfo error) {
        ++notifications;
        EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_FALSE(context->success());
        EXPECT_TRUE(context->deadlineExceeded());
    });
    context->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    block_tree_cache_test::BoundedThread<void> waiter([context] { context->waitDone(); });
    ASSERT_EQ(waiter.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    waiter.get();
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->deadlineExceeded());
    EXPECT_TRUE(backend->readPending());
    EXPECT_EQ(pool->refCount(block), 2u);
    size_t timeout_observations = 0;
    context->onDone([&](ErrorInfo error) {
        ++timeout_observations;
        EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
    });
    EXPECT_EQ(timeout_observations, 1u);
    backend->completeRead();
    EXPECT_TRUE(backend->readKeys().empty());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(notifications, 1u);
    EXPECT_EQ(timeout_observations, 1u);
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_FALSE(pool->isFrozen(block));
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, TimelyRemoteReadDoesNotExpireIndependentHostDiskAndJoin) {
    size_t     commits     = 0;
    size_t     aborts      = 0;
    auto       coordinator = makeCoordinator(commits, aborts);
    auto       backend     = std::make_shared<ManualBackend>(std::chrono::milliseconds(100));
    auto       pool        = std::make_shared<TestBlockPool>();
    const auto block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    TransferDescriptor host;
    host.source_tier = Tier::HOST;
    TransferDescriptor disk;
    disk.source_tier = Tier::DISK;
    auto joined_load = block_tree_cache_test::makeLoadOperationForTest(disk.source_tier, 1);
    auto context =
        coordinator->create({host, disk}, {false, true}, {nullptr, joined_load}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });
    size_t notifications = 0;
    context->onDone([&](ErrorInfo error) {
        ++notifications;
        EXPECT_TRUE(error.ok());
        EXPECT_TRUE(context->success());
    });
    context->startBackendMatch();
    backend->completeMatch(1);
    backend->completeRead();
    EXPECT_EQ(backend->matchDeadline(), backend->readDeadline());
    ASSERT_FALSE(context->done());
    std::this_thread::sleep_until(backend->readDeadline() + std::chrono::milliseconds(1));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeTransfers(1, true));
    joined_load->complete(true);
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(context->errorInfo().ok());
    EXPECT_EQ(notifications, 1u);
    EXPECT_TRUE(context->done());
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, BackendFailureWaitsForLocalTransferBeforePublishingFailure) {
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
    auto context      = coordinator->create({local}, {false}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 1u);
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });

    context->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    EXPECT_EQ(pool->refCount(block), 2u);
    backend->failNextRead();
    backend->completeRead();
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeTransfers(1, true));
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, RemoteTimeoutPreservesOnlySettledLocalReuse) {
    for (bool remote_first : {false, true}) {
        for (bool local_success : {false, true}) {
            for (bool settlement_success : {false, true}) {
                SCOPED_TRACE(testing::Message() << remote_first << ":" << local_success << ":" << settlement_success);
                size_t     commits     = 0;
                size_t     aborts      = 0;
                auto       coordinator = makeCoordinator(commits, aborts);
                auto       backend     = std::make_shared<ManualBackend>();
                auto       pool        = std::make_shared<TestBlockPool>();
                const auto block       = pool->malloc().value();
                pool->incRef(block);
                initBackend(*backend, pool);
                TransferDescriptor host;
                host.source_tier = Tier::HOST;
                host.path_index  = 1;
                TransferDescriptor disk;
                disk.source_tier                 = Tier::DISK;
                disk.path_index                  = 2;
                auto request                     = makeRequest(4);
                request.local_matched_blocks_num = 3;
                auto joined_load                 = block_tree_cache_test::makeLoadOperationForTest(disk.source_tier, 1);
                auto context                     = coordinator->create(
                    {host, disk}, {false, true}, {nullptr, joined_load}, 3, 0, 0, {}, backend, std::move(request));
                ASSERT_NE(context, nullptr);
                context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
                    EXPECT_EQ(matched, 4u);
                    current.setBackendTargetBlock(3, 0, block);
                    return current.commit();
                });
                size_t settlements = 0;
                context->setSettlementReadyCallback([&](const auto& current) {
                    ++settlements;
                    EXPECT_FALSE(current->aggregateSuccess());
                    EXPECT_EQ(current->localTransfersSucceeded(), local_success);
                    EXPECT_FALSE(current->localLoadSucceeded());
                    EXPECT_TRUE(current->settle(current->localTransfersSucceeded() && settlement_success));
                });
                context->startBackendMatch();
                backend->completeMatch(4);
                backend->timeoutNextRead();
                const auto finish_local = [&] {
                    EXPECT_TRUE(context->completeTransfers(1, true));
                    joined_load->complete(local_success);
                };
                if (remote_first) {
                    backend->completeRead();
                    EXPECT_FALSE(context->done());
                    EXPECT_FALSE(context->localLoadSucceeded());
                    finish_local();
                } else {
                    finish_local();
                    EXPECT_FALSE(context->done());
                    EXPECT_FALSE(context->localLoadSucceeded());
                    backend->completeRead();
                }
                EXPECT_EQ(settlements, 1u);
                EXPECT_TRUE(context->done());
                EXPECT_FALSE(context->success());
                EXPECT_TRUE(context->backendTimedOut());
                EXPECT_EQ(context->localLoadSucceeded(), local_success && settlement_success);
                EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
                EXPECT_EQ(context->localMatchedBlocks(), 3u);
                EXPECT_EQ(context->matchedBlocks(Tier::HOST), 1u);
                EXPECT_EQ(context->matchedBlocks(Tier::DISK), 1u);
                EXPECT_TRUE(pool->isFrozen(block));
                pool->decRef(block);
                coordinator->shutdown();
            }
        }
    }
}

TEST(LoadAsyncContextTest, RemoteTimeoutDoesNotFailLocalJoiners) {
    size_t     commits     = 0;
    size_t     aborts      = 0;
    auto       coordinator = makeCoordinator(commits, aborts);
    auto       backend     = std::make_shared<ManualBackend>();
    auto       pool        = std::make_shared<TestBlockPool>();
    const auto block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    TransferDescriptor host;
    host.source_tier                 = Tier::HOST;
    auto request                     = makeRequest(2);
    request.local_matched_blocks_num = 1;
    auto owner = coordinator->create({host}, {false}, {}, 1, 0, 0, {}, backend, std::move(request));
    ASSERT_NE(owner, nullptr);
    owner->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(1, 0, block);
        return current.commit();
    });
    auto shared_load = block_tree_cache_test::makeLoadOperationForTest(host.source_tier, 1);
    std::vector<std::shared_ptr<LoadAsyncContext>> joiners;
    for (int i = 0; i < 2; ++i) {
        auto joined = coordinator->create({host}, {true}, {shared_load}, 1);
        ASSERT_NE(joined, nullptr);
        ASSERT_TRUE(joined->commit());
        joiners.push_back(joined);
    }
    owner->setSettlementReadyCallback([&](const auto& current) {
        const bool local_success = current->localTransfersSucceeded();
        shared_load->complete(local_success);
        EXPECT_TRUE(current->settle(local_success));
    });
    owner->startBackendMatch();
    backend->completeMatch(2);
    ASSERT_TRUE(owner->completeTransfers(1, true));
    backend->timeoutNextRead();
    backend->completeRead();
    EXPECT_FALSE(owner->success());
    EXPECT_TRUE(owner->localLoadSucceeded());
    for (const auto& joined : joiners) {
        EXPECT_TRUE(joined->success());
        EXPECT_TRUE(joined->localLoadSucceeded());
        EXPECT_TRUE(joined->errorInfo().ok());
    }
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, PendingContextAbortsOnDestruction) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    {
        auto context = coordinator->create({TransferDescriptor{}}, {false}, {}, 1);
        ASSERT_NE(context, nullptr);
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
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);

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
    std::shared_ptr<LoadAsyncContext> context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
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
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);

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

TEST(LoadAsyncContextTest, FailureWaitsForBackendAndPropagatesToTwoJoiners) {
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
    auto owner        = coordinator->create({local, local}, {false, false}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(owner, nullptr);
    owner->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(0, 0, block);
        return current.commit();
    });
    auto shared_load = block_tree_cache_test::makeLoadOperationForTest(local.source_tier, 1);
    std::vector<std::shared_ptr<LoadAsyncContext>> joiners;
    size_t                                         joined_callbacks = 0;
    for (int i = 0; i < 2; ++i) {
        auto joined = coordinator->create({local}, {true}, {shared_load}, 0);
        ASSERT_NE(joined, nullptr);
        ASSERT_TRUE(joined->commit());
        joined->onDone([&](ErrorInfo error) {
            EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION);
            ++joined_callbacks;
        });
        joiners.push_back(joined);
    }
    size_t settlements = 0;
    owner->setSettlementReadyCallback([&](const std::shared_ptr<LoadAsyncContext>& current) {
        ++settlements;
        EXPECT_FALSE(backend->readPending());
        EXPECT_EQ(pool->refCount(block), 1u);
        shared_load->complete(current->aggregateSuccess());
        EXPECT_TRUE(current->settle(false));
    });
    owner->startBackendMatch();
    backend->completeMatch(1);
    ASSERT_TRUE(backend->readPending());
    ASSERT_EQ(pool->refCount(block), 2u);
    EXPECT_TRUE(owner->completeTransfers(1, false));
    EXPECT_TRUE(owner->completeTransfers(1, false));
    EXPECT_FALSE(owner->done());
    EXPECT_EQ(settlements, 0u);
    EXPECT_EQ(joined_callbacks, 0u);
    EXPECT_EQ(pool->refCount(block), 2u);
    for (const auto& joined : joiners)
        EXPECT_FALSE(joined->done());
    backend->completeRead();
    EXPECT_EQ(settlements, 1u);
    EXPECT_EQ(joined_callbacks, 2u);
    EXPECT_EQ(owner->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_TRUE(owner->done());
    EXPECT_FALSE(owner->success());
    size_t late_callbacks = 0;
    owner->onDone([&](ErrorInfo error) {
        EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION);
        ++late_callbacks;
    });
    EXPECT_EQ(late_callbacks, 1u);
    for (const auto& joined : joiners) {
        EXPECT_FALSE(joined->success());
        EXPECT_EQ(joined->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    }
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, LoadFailureWaitsForOtherTransfers) {
    size_t             commits     = 0;
    size_t             aborts      = 0;
    auto               coordinator = makeCoordinator(commits, aborts);
    TransferDescriptor local;
    local.source_tier = Tier::HOST;
    auto context      = coordinator->create({local, local}, {false, false}, {}, 0);
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(context->commit());
    EXPECT_TRUE(context->completeTransfers(1, false));
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->completeTransfers(1, true));
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_FALSE(context->success());
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, BackendFailureBeforeCommitIsPreserved) {
    size_t     commits     = 0;
    size_t     aborts      = 0;
    auto       coordinator = makeCoordinator(commits, aborts);
    auto       backend     = std::make_shared<ManualBackend>();
    auto       pool        = std::make_shared<TestBlockPool>();
    const auto block       = pool->malloc().value();
    pool->incRef(block);
    initBackend(*backend, pool);
    auto context = coordinator->create({}, {}, {}, 0, 0, 0, {}, backend, makeRequest(1));
    ASSERT_NE(context, nullptr);
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(0, 0, block);
        return true;  // Deliberately postpone commit until after backend completion.
    });
    context->startBackendMatch();
    backend->completeMatch(1);
    backend->failNextRead();
    backend->completeRead();
    EXPECT_FALSE(context->done());
    ASSERT_TRUE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    context->onDone([](ErrorInfo error) { EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION); });
    pool->decRef(block);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, SettlementFailurePublishesError) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   context     = coordinator->create({}, {}, {}, 0);
    context->setSettlementReadyCallback([](const auto& ready) { EXPECT_TRUE(ready->settle(false)); });
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(context->commit());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    context->onDone([](ErrorInfo error) { EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION); });
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, CancelledCommitKeepsDevicePrefixBeforeJoinFailureNotification) {
    for (const bool explicit_settlement : {false, true}) {
        for (const bool join_notified_first : {false, true}) {
            SCOPED_TRACE(explicit_settlement);
            SCOPED_TRACE(join_notified_first);
            auto coordinator = std::make_shared<LoadContextCoordinator>(
                [explicit_settlement](const auto& context) {
                    if (explicit_settlement) {
                        context->setSettlementReadyCallback([](const auto& ready) {
                            EXPECT_FALSE(ready->hostTransfersSucceeded());
                            EXPECT_TRUE(
                                ready->settle(ready->localTransfersSucceeded(), ready->hostTransfersSucceeded()));
                        });
                    }
                    context->cancelUnsubmittedTransfers();
                    return true;
                },
                [](LoadAsyncContext&) {});
            TransferDescriptor device, joined_host, pending_host;
            device.source_tier      = Tier::DEVICE;
            joined_host.source_tier = pending_host.source_tier = Tier::HOST;
            auto joined_load = block_tree_cache_test::makeLoadOperationForTest(joined_host.source_tier, 1);
            auto context     = coordinator->create(
                {device, joined_host, pending_host}, {false, true, false}, {nullptr, joined_load, nullptr}, 3, 1, 3);
            ASSERT_NE(context, nullptr);
            if (join_notified_first) {
                joined_load->complete(false);
            }
            ASSERT_TRUE(context->commit());
            ASSERT_TRUE(context->done());
            EXPECT_FALSE(context->success());
            EXPECT_FALSE(context->localLoadSucceeded());
            EXPECT_EQ(context->deviceReadyBlocks(), 1u);
            if (!join_notified_first) {
                joined_load->complete(false);
                EXPECT_EQ(context->deviceReadyBlocks(), 1u);
            }
            coordinator->shutdown();
        }
    }
}

TEST(LoadAsyncContextTest, CancelledCommitRollsBackOnceAndStillWaitsForBackend) {
    size_t aborts        = 0;
    size_t notifications = 0;
    auto   coordinator   = std::make_shared<LoadContextCoordinator>(
        [&](const auto& context) {
            context->setSettlementReadyCallback([&](const auto& ready) {
                EXPECT_EQ(aborts, 1u);
                EXPECT_TRUE(ready->settle(ready->localTransfersSucceeded(), ready->hostTransfersSucceeded()));
            });
            context->cancelUnsubmittedTransfers();
            return true;
        },
        [&](LoadAsyncContext& context) {
            EXPECT_FALSE(context.done());
            ++aborts;
        });
    auto backend = std::make_shared<ManualBackend>();
    auto pool    = std::make_shared<TestBlockPool>();
    initBackend(*backend, pool);
    const auto block = pool->malloc().value();
    pool->incRef(block);
    TransferDescriptor local;
    local.source_tier                = Tier::HOST;
    auto request                     = makeRequest(2);
    request.local_matched_blocks_num = 1;
    auto context = coordinator->create({local}, {false}, {}, 1, 0, 0, {}, backend, std::move(request));
    context->setMatchCallback([&](LoadAsyncContext& current, size_t) {
        current.setBackendTargetBlock(1, 0, block);
        return current.commit();
    });
    context->onDone([&](ErrorInfo error) {
        EXPECT_FALSE(error.ok());
        EXPECT_EQ(aborts, 1u);
        ++notifications;
    });
    context->startBackendMatch();
    backend->completeMatch(2);

    EXPECT_EQ(aborts, 1u);
    EXPECT_EQ(notifications, 0u);
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(context->abortPending());
    ASSERT_TRUE(backend->readPending());
    backend->completeRead();
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::NONE);
    EXPECT_EQ(notifications, 1u);
    context.reset();
    coordinator->shutdown();
    EXPECT_EQ(aborts, 1u);
    pool->decRef(block);
}

TEST(LoadAsyncContextTest, ShutdownWaitsForFailedCommitBeforeClearingCallbacks) {
    using namespace std::chrono_literals;
    auto              barrier     = std::make_shared<block_tree_cache_test::CallbackBarrier>();
    auto              coordinator = std::make_shared<LoadContextCoordinator>(
        [barrier](const auto&) {
            barrier->enterAndWait();
            return false;
        },
        [](LoadAsyncContext&) { ADD_FAILURE() << "failed commit must already be rolled back"; });
    auto              context     = coordinator->create({}, {}, {}, 0);
    std::future<bool> commit;
    std::future<void> shutdown;
    block_tree_cache_detail::ScopeRollback release([&] { barrier->release(); });
    commit = std::async(std::launch::async, [&] { return context->commit(); });
    ASSERT_TRUE(barrier->waitUntilEnteredFor(1, 5s));
    EXPECT_FALSE(context->done());
    shutdown = std::async(std::launch::async, [&] { coordinator->shutdown(); });
    EXPECT_EQ(shutdown.wait_for(50ms), std::future_status::timeout);
    release.run();
    ASSERT_EQ(commit.wait_for(5s), std::future_status::ready);
    EXPECT_FALSE(commit.get());
    ASSERT_EQ(shutdown.wait_for(5s), std::future_status::ready);
    shutdown.get();
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    context.reset();
}

TEST(LoadAsyncContextTest, CommitRejectsPendingContextDuringAbort) {
    std::promise<void> entered;
    std::promise<void> release;
    auto               released    = release.get_future().share();
    auto               coordinator = std::make_shared<LoadContextCoordinator>(
        [](const auto&) {
            ADD_FAILURE() << "cancelled load was submitted";
            return false;
        },
        [&](auto&) {
            entered.set_value();
            released.wait();
        });
    auto context = coordinator->create({}, {}, {}, 0);
    ASSERT_NE(context, nullptr);
    auto abort = std::async(std::launch::async, [&] { return context->abortPending(); });
    entered.get_future().wait();
    EXPECT_FALSE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    release.set_value();
    EXPECT_TRUE(abort.get());
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    coordinator->shutdown();
}

TEST(LoadAsyncContextTest, ShutdownRejectsCommitOfAbortedPendingContext) {
    size_t commits     = 0;
    size_t aborts      = 0;
    auto   coordinator = makeCoordinator(commits, aborts);
    auto   context     = coordinator->create({}, {}, {}, 0);
    ASSERT_NE(context, nullptr);

    coordinator->shutdown();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->commit());
    EXPECT_FALSE(context->commit());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 1u);
}

TEST(LoadAsyncContextTest, AbortedContextRejectsCommitWhetherCoordinatorIsOpenOrClosed) {
    size_t commits        = 0;
    size_t aborts         = 0;
    auto   coordinator    = makeCoordinator(commits, aborts);
    auto   live_context   = coordinator->create({}, {}, {}, 0);
    auto   closed_context = coordinator->create({}, {}, {}, 0);
    ASSERT_NE(live_context, nullptr);
    ASSERT_NE(closed_context, nullptr);
    ASSERT_TRUE(live_context->abortPending());
    ASSERT_TRUE(closed_context->abortPending());

    EXPECT_FALSE(live_context->commit());
    EXPECT_TRUE(live_context->done());
    EXPECT_FALSE(live_context->success());
    EXPECT_EQ(live_context->mallocStatus(), MallocStatus::INTERNAL_ERROR);

    coordinator->shutdown();

    EXPECT_FALSE(closed_context->commit());
    EXPECT_TRUE(closed_context->done());
    EXPECT_FALSE(closed_context->success());
    EXPECT_EQ(closed_context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    EXPECT_EQ(commits, 0u);
    EXPECT_EQ(aborts, 2u);
}

TEST(LoadAsyncContextTest, ShutdownRejectsCommitBeforeAbortCleanupFinishes) {
    using namespace std::chrono_literals;
    std::promise<void> entered;
    std::promise<void> release;
    auto               released    = release.get_future().share();
    auto               coordinator = std::make_shared<LoadContextCoordinator>(
        [](const auto&) {
            ADD_FAILURE() << "shutdown load was submitted";
            return false;
        },
        [&](auto&) {
            entered.set_value();
            released.wait();
        });
    auto context = coordinator->create({}, {}, {}, 0);
    ASSERT_NE(context, nullptr);

    std::future<void>                      shutdown;
    block_tree_cache_detail::ScopeRollback release_guard([&] { release.set_value(); });
    shutdown = std::async(std::launch::async, [&] { coordinator->shutdown(); });
    ASSERT_EQ(entered.get_future().wait_for(5s), std::future_status::ready);

    EXPECT_FALSE(context->commit());
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
    release_guard.run();
    shutdown.get();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->mallocStatus(), MallocStatus::INTERNAL_ERROR);
}

}  // namespace
}  // namespace rtp_llm
