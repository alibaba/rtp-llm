#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/LegacyTransferEngine.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <pthread.h>
#include <set>
#include <stdexcept>
#include <thread>

#include <sys/resource.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include "autil/EnvUtil.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AsyncProfileTrace.h"

namespace rtp_llm {
namespace {
using namespace block_transfer_engine_test;
using namespace std::chrono_literals;

// Physical disk reads are exercised directly below; this is not an admitted logical route.
TransferDescriptor physicalDiskRead(size_t group, BlockIdxType disk, BlockIdxType host) {
    auto desc = TransferDescriptor::hostToDisk(group, host, disk);
    std::swap(desc.source_tier, desc.target_tier);
    std::swap(desc.source_blocks, desc.target_blocks);
    return desc;
}

size_t seriesCount(kmonitor::MutableMetric* metric) {
    return metric->metric_data_->Size();
}

double snapshot(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags& tags) {
    auto* series = metric->DeclareMetric(&tags);
    if (!series) {
        ADD_FAILURE() << "missing series: " << tags.ToString();
        return -1;
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    series->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(series));
    if (record.Values().size() != 1) {
        ADD_FAILURE() << "missing value: " << tags.ToString();
        return -1;
    }
    return std::stod(record.Values().front()->Value());
}

// Uses the real DiskIOExecutor queue/callback implementation, with bounded barriers
// at physical I/O. These tests verify scheduling, not SSD or CUDA behavior.
class ControlledDiskIO: public DiskBlockIO {
public:
    ~ControlledDiskIO() override {
        close();
    }
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t bytes, bool) override {
        data.resize(bytes);
        return DiskBlockIOStatus::OK;
    }
    void        close() override {}
    std::string debugString() const override {
        return "ControlledDiskIO";
    }
    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override {
        return access(false, offset, dst, bytes);
    }
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        return access(true, offset, const_cast<void*>(src), bytes);
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& items) override {
        for (const auto& item : items) {
            auto status = read(item.offset, item.buffer, item.bytes);
            if (status != DiskBlockIOStatus::OK)
                return status;
        }
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& items) override {
        for (const auto& item : items) {
            auto status = write(item.offset, item.buffer, item.bytes);
            if (status != DiskBlockIOStatus::OK)
                return status;
        }
        return DiskBlockIOStatus::OK;
    }
    bool waitWrites(size_t count = 1) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, 5s, [&] { return writes >= count; });
    }
    bool waitReads(size_t count = 1) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, 5s, [&] { return reads >= count; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        blocked = false;
        cv.notify_all();
    }
    DiskBlockIOStatus access(bool write, uint64_t offset, void* buffer, size_t bytes) {
        std::unique_lock<std::mutex> lock(mutex);
        worker = std::this_thread::get_id();
        if (write)
            ++writes;
        else
            ++reads;
        cv.notify_all();
        if (!cv.wait_for(lock, 10s, [&] { return !blocked; }))
            return DiskBlockIOStatus::IO_ERROR;
        if (write && writes == fail_write)
            return DiskBlockIOStatus::PARTIAL_FAILURE;
        if (write)
            std::memcpy(data.data() + offset, buffer, bytes);
        else
            std::memcpy(buffer, data.data() + offset, bytes);
        return DiskBlockIOStatus::OK;
    }
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    blocked{false};
    size_t                  writes{0}, reads{0}, fail_write{0};
    std::thread::id         worker;
    std::vector<uint8_t>    data;
};

DiskIOExecutorConfig diskExecutorOptions() {
    DiskIOExecutorConfig config;
    config.read_thread_count = config.write_thread_count = 1;
    config.read_queue_size = config.write_queue_size = 16;
    return config;
}

bool await(const std::shared_ptr<TransferAsyncContext>& context) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future  = promise->get_future();
    context->onDone([promise](TransferResult) { promise->set_value(); });
    return future.wait_for(5s) == std::future_status::ready;
}

void checkWorkLoopTrace(const std::vector<AsyncProfileEvent>& events) {
    for (const auto& event : events) {
        const auto& r = event.record;
        if (r.stage == "dropped")
            continue;
        if (r.stage == "transfer_queue_wait" || r.stage == "staging_wait") {
            EXPECT_EQ(r.kind, AsyncProfileKind::Wait) << r.stage;
        } else {
            EXPECT_TRUE(r.stage == "worker_task_total" || r.stage == "worker_batch_total" || r.stage == "h2d"
                        || r.stage == "d2h" || r.stage == "disk_read" || r.stage == "disk_write")
                << r.stage;
            EXPECT_EQ(r.kind, AsyncProfileKind::Lifecycle) << r.stage;
        }
    }
}

void prepareOomDeathTest(const char* marker) {
    const rlimit limit{0, 0};
    if (setrlimit(RLIMIT_CORE, &limit) != 0)
        _exit(2);
    std::fprintf(stderr, "%s\n", marker);
}

class TransferWorkLoopTest: public ::testing::Test {
protected:
    void SetUp() override {
        direct_disk_executor = std::make_unique<DiskIOExecutor>(diskOptions());
        direct_disk_executor->start();
        host                 = makeHostPool(384, 16);
        auto controlled      = std::make_unique<ControlledDiskIO>();
        io                   = controlled.get();
        disk                 = makeDiskPool(384, 16, temp.path, std::move(controlled));
        group                = makeTestGroupSet(
            0,
            makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 384)}),
            {"group0"},
            {},
            host,
            disk);
        for (size_t i = 0; i < 8; ++i) {
            hosts.push_back(poolMalloc(*host));
            disks.push_back(poolMalloc(*disk));
            std::memset(host->blockBuffer(hosts.back()).addr, int(i + 17), 384);
        }
    }
    virtual DiskIOExecutorConfig diskOptions() const {
        return diskExecutorOptions();
    }
    void TearDown() override {
        io->release();
        request_dispatcher.reset();
        engine.reset();
        direct_disk_executor.reset();
        for (auto block : hosts)
            releasePoolBlock(*host, block);
        for (auto block : disks)
            releasePoolBlock(*disk, block);
    }
    void initialize(
        size_t q = 8, size_t m = 2, size_t c = 1, size_t workers = 1, size_t staging_blocks = 2, bool async = true) {
        engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                              true,
                                                              DeviceHostCopyOptions{},
                                                              staging_blocks,
                                                              m,
                                                              workers,
                                                              q,
                                                              reporter,
                                                              3,
                                                              c,
                                                              async,
                                                              diskOptions());
        EXPECT_EQ(engine->transferWorkerCount(), workers);
        EXPECT_EQ(engine->asyncEnabled(), async);
    }
    std::shared_ptr<TransferAsyncContext> executeRequest(TransferTask task) {
        if (!request_dispatcher || request_dispatcher->per_rank_engine_ != engine)
            request_dispatcher = std::make_unique<BlockTransferDispatcher>(engine);
        return request_dispatcher->executePerRank(std::move(task));
    }
    void enableMetrics(bool stages = true) {
        raw      = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
        reporter = std::make_shared<BlockTreeCacheMetricsReporter>(raw, stages, 1h);
    }
    RtpLLMCacheTransferMetrics* metrics() {
        return raw->getMetricsGroup<RtpLLMCacheTransferMetrics>();
    }
    void checkStage(const char* stage,
                    size_t      count,
                    bool        success = true,
                    Tier        source  = Tier::HOST,
                    Tier        target  = Tier::DISK,
                    const char* scope   = "worker_batch") {
        kmonitor::MetricsTags tags("operation", target == Tier::DEVICE ? "load" : "evict");
        tags.AddTag("source_tier", tierName(source));
        tags.AddTag("target_tier", tierName(target));
        tags.AddTag("stage", stage);
        tags.AddTag("scope", scope);
        tags.AddTag("success", success ? "true" : "false");
        tags.AddTag("transfer_rank", "3");
        EXPECT_DOUBLE_EQ(snapshot(metrics()->transfer_stage_qps_metric, tags), count) << stage;
        EXPECT_GE(snapshot(metrics()->transfer_stage_latency_sum_us_metric, tags), 0) << stage;
    }
    void checkQueueWait(size_t count, Tier source = Tier::HOST, Tier target = Tier::DISK) {
        kmonitor::MetricsTags tags("pool_type", "transfer");
        tags.AddTag("source_tier", tierName(source));
        tags.AddTag("target_tier", tierName(target));
        EXPECT_DOUBLE_EQ(snapshot(metrics()->task_queue_waiting_tasks_metric, tags), count);
        EXPECT_GE(snapshot(metrics()->transfer_task_queue_wait_latency_us_metric, tags), 0);
    }
    TransferTask writes(size_t count = 1) {
        std::vector<TransferDescriptor> descriptors;
        for (size_t i = 0; i < count; ++i)
            descriptors.push_back(TransferDescriptor::hostToDisk(0, hosts[i], disks[i]));
        return makeTransferTask(std::move(descriptors));
    }
    TempDirGuard                                   temp{"transfer_workloop"};
    std::shared_ptr<HostBlockPool>                 host;
    DiskBlockPoolPtr                               disk;
    GroupSetPtr                                    group;
    std::shared_ptr<PerRankBlockTransferEngine>    engine;
    std::unique_ptr<BlockTransferDispatcher>       request_dispatcher;
    std::shared_ptr<kmonitor::MetricsReporter>     raw;
    std::shared_ptr<BlockTreeCacheMetricsReporter> reporter;
    std::unique_ptr<DiskIOExecutor>                direct_disk_executor;
    ControlledDiskIO*                              io{nullptr};
    std::vector<BlockIdxType>                      hosts, disks;
};

class TransferWorkLoopDeathTest: public TransferWorkLoopTest {
protected:
    void SetUp() override {
        ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        TransferWorkLoopTest::SetUp();
    }
};

// Opt-in CPU hot-path benchmark. The same bounded in-memory Disk backend is
// used in every mode, so SSD/CUDA latency cannot hide observation overhead.
class TransferWorkLoopProfileBenchmark: public TransferWorkLoopTest, public ::testing::WithParamInterface<size_t> {
protected:
    DiskIOExecutorConfig diskOptions() const override {
        auto config              = diskExecutorOptions();
        config.read_thread_count = config.write_thread_count = GetParam();
        config.read_queue_size = config.write_queue_size = 512;
        return config;
    }
};

TEST_P(TransferWorkLoopProfileBenchmark, DISABLED_ProfilingOverhead) {
    constexpr size_t tasks = 16384, wave = 16;
    for (size_t workers : {1u, 2u}) {
        for (size_t batch : {1u, 8u}) {
            for (const char* mode : {"off", "metrics", "trace", "both"}) {
                const bool metrics_enabled = std::strcmp(mode, "metrics") == 0 || std::strcmp(mode, "both") == 0;
                const bool trace_enabled   = std::strcmp(mode, "trace") == 0 || std::strcmp(mode, "both") == 0;
                reporter.reset();
                raw.reset();
                if (metrics_enabled)
                    enableMetrics();
                initialize(wave, batch, 1, workers);
                const auto                                         task = writes(batch);
                std::vector<std::shared_ptr<TransferAsyncContext>> contexts;
                contexts.reserve(wave);
                auto run = [&](size_t count) {
                    for (size_t offset = 0; offset < count; offset += wave) {
                        contexts.clear();
                        for (size_t i = 0; i < wave; ++i)
                            contexts.push_back(engine->execute(task));
                        engine->waitForIdle();
                        for (const auto& context : contexts)
                            ASSERT_TRUE(context->success()) << context->errorInfo().ToString();
                    }
                };
                ASSERT_NO_FATAL_FAILURE(run(512));
                auto       session = trace_enabled ? AsyncProfileSession::start(400000) : nullptr;
                const auto begin   = std::chrono::steady_clock::now();
                ASSERT_NO_FATAL_FAILURE(run(tasks));
                const auto elapsed =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin);
                std::printf("WORKLOOP_BENCH mode=%s disk_workers=%zu owners=%zu batch=%zu tasks=%zu us_per_task=%.6f\n",
                            mode,
                            GetParam(),
                            workers,
                            batch,
                            tasks,
                            elapsed.count() / tasks);
                engine->shutdown();
                engine.reset();
                if (session) {
                    EXPECT_FALSE(session->finish().empty());
                }
            }
        }
    }
}

TEST_P(TransferWorkLoopProfileBenchmark, TraceCardinalityFollowsBatches) {
    initialize(8, 8, 1, 2);
    auto session = AsyncProfileSession::start(100000);
    ASSERT_TRUE(await(engine->execute(writes(8))));
    ASSERT_TRUE(await(engine->execute(writes(8))));
    engine->shutdown();
    const auto events = session->finish();
    checkWorkLoopTrace(events);
    EXPECT_EQ(events.size(), 6u);  // Three intervals per batch, for eight blocks and any Disk worker count.
    EXPECT_EQ(io->writes, 16u);
    EXPECT_EQ(io->reads, 0u);
}

INSTANTIATE_TEST_SUITE_P(DiskWorkers, TransferWorkLoopProfileBenchmark, ::testing::Values(1u, 8u, 16u));

TEST_F(TransferWorkLoopTest, MultipleOwnersRoundRobinWithStaticDirectionCapacityAndThreadNames) {
    initialize(2, 1, 1, 2);
    io->blocked = true;
    std::array<std::promise<std::string>, 2>           names;
    std::array<std::future<std::string>, 2>            observed{names[0].get_future(), names[1].get_future()};
    std::vector<std::shared_ptr<TransferAsyncContext>> contexts;
    block_tree_cache_detail::ScopeRollback             cleanup([&] {
        io->release();
        engine->shutdown();
    });
    ASSERT_EQ(engine->load_loops_.size(), 2u);
    ASSERT_EQ(engine->evict_loops_.size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        auto context = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[i], disks[i])}));
        context->onDone([&, i](TransferResult result) {
            EXPECT_TRUE(result.ok());
            char name[16]{};
            EXPECT_EQ(pthread_getname_np(pthread_self(), name, sizeof(name)), 0);
            names[i].set_value(name);
        });
        contexts.push_back(std::move(context));
    }
    EXPECT_EQ(engine->queueSizes().background, 2u);
    EXPECT_EQ(engine->evict_loops_[0]->outstandingBatches(), 1u);
    EXPECT_EQ(engine->evict_loops_[1]->outstandingBatches(), 1u);
    auto rejected = engine->execute(writes());
    ASSERT_TRUE(rejected->done());
    EXPECT_EQ(rejected->result().copy_error, CacheCopyError::NOT_EXECUTED);
    io->release();
    engine->waitForIdle();
    for (size_t i = 0; i < 2; ++i) {
        ASSERT_EQ(observed[i].wait_for(5s), std::future_status::ready);
        EXPECT_EQ(observed[i].get(), "btc_evict_" + std::to_string(i));
    }
    EXPECT_EQ(engine->queueSizes().background, 0u);
    auto reused = engine->execute(writes());
    ASSERT_TRUE(await(reused));
    EXPECT_TRUE(reused->success());
    cleanup.run();
    for (const auto& loop : engine->evict_loops_)
        EXPECT_FALSE(loop->accepting_);
    for (const auto& loop : engine->evict_loops_) {
        EXPECT_EQ(loop->capacity_, 1u);
        EXPECT_EQ(loop->outstandingBatches(), 0u);
    }
}

TEST_F(TransferWorkLoopTest, AdmissionRetriesOnlyFullOwnersAndPreservesUnacceptedArguments) {
    initialize(1, 1);
    io->blocked = true;
    block_tree_cache_detail::ScopeRollback cleanup([&] {
        io->release();
        engine->shutdown();
    });
    auto&                                  owner  = *engine->evict_loops_.front();
    auto                                   active = owner.execute(writes());
    ASSERT_TRUE(io->waitWrites());

    auto                                     task      = writes();
    const auto                               trace_id  = task.traceId();
    size_t                                   callbacks = 0;
    TransferWorkLoop::BeforeCompleteCallback callback  = [&](const TransferResult& result) {
        EXPECT_TRUE(result.ok());
        ++callbacks;
    };
    bool retryable = false;
    EXPECT_EQ(owner.execute(std::move(task), std::move(callback), &retryable), nullptr);
    EXPECT_TRUE(retryable);
    ASSERT_EQ(task.descriptors().size(), 1u);
    EXPECT_EQ(task.traceId(), trace_id);
    EXPECT_TRUE(callback);
    EXPECT_EQ(callbacks, 0u);

    auto invalid  = makeTransferTask({});
    auto rejected = engine->execute(std::move(invalid));
    ASSERT_NE(rejected, nullptr);
    EXPECT_EQ(rejected->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    auto expired      = writes();
    expired.deadline_ = TransferTask::Clock::now() - 1ms;
    rejected          = owner.execute(std::move(expired), {}, &retryable);
    ASSERT_NE(rejected, nullptr);
    EXPECT_FALSE(retryable);
    EXPECT_EQ(rejected->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);

    io->release();
    ASSERT_TRUE(await(active));
    engine->waitForIdle();
    auto accepted = owner.execute(std::move(task), std::move(callback), &retryable);
    ASSERT_NE(accepted, nullptr);
    EXPECT_FALSE(retryable);
    ASSERT_TRUE(await(accepted));
    EXPECT_TRUE(accepted->success());
    engine->waitForIdle();
    EXPECT_EQ(callbacks, 1u);

    owner.stopAdmission();
    rejected = owner.execute(writes(), {}, &retryable);
    ASSERT_NE(rejected, nullptr);
    EXPECT_FALSE(retryable);
    EXPECT_FALSE(rejected->success());
}

TEST_F(TransferWorkLoopTest, FullOwnerRetriesOtherOwnersUntilDirectionCapacityIsExhausted) {
    initialize(5, 1, 1, 3);
    for (size_t worker = 0; worker < 3; ++worker) {
        const auto expected = worker < 2 ? 2u : 1u;
        EXPECT_EQ(engine->load_loops_[worker]->capacity_, expected);
        EXPECT_EQ(engine->evict_loops_[worker]->capacity_, expected);
    }
    io->blocked = true;
    std::vector<std::shared_ptr<TransferAsyncContext>> contexts;
    block_tree_cache_detail::ScopeRollback             cleanup([&] {
        io->release();
        engine->shutdown();
    });
    // Fill the last owner's single slot while the other owners still have room.
    contexts.push_back(engine->evict_loops_[2]->execute(writes()));
    for (size_t i = 1; i < 3; ++i)
        contexts.push_back(engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[i], disks[i])})));
    EXPECT_EQ(engine->queueSizes().background, 3u);
    contexts.push_back(engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[3], disks[3])})));
    EXPECT_FALSE(contexts.back()->done());
    EXPECT_EQ(engine->evict_loops_[0]->outstandingBatches(), 2u);
    EXPECT_EQ(engine->evict_loops_[1]->outstandingBatches(), 1u);
    EXPECT_EQ(engine->evict_loops_[2]->outstandingBatches(), 1u);
    contexts.push_back(engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[4], disks[4])})));
    EXPECT_FALSE(contexts.back()->done());
    EXPECT_EQ(engine->queueSizes().background, 5u);
    auto rejected = engine->execute(writes());
    ASSERT_TRUE(rejected->done());
    EXPECT_EQ(rejected->result().copy_error, CacheCopyError::NOT_EXECUTED);
    cleanup.run();
    for (const auto& context : contexts)
        EXPECT_TRUE(context->success());
    EXPECT_EQ(io->writes, 5u);
}

TEST_F(TransferWorkLoopTest, ConcurrentSubmittersRespectStaticDirectionCapacity) {
    initialize(5, 1, 1, 3);
    io->blocked = true;
    std::promise<void>                                              start;
    auto                                                            ready = start.get_future().share();
    std::vector<std::future<std::shared_ptr<TransferAsyncContext>>> submitters;
    block_tree_cache_detail::ScopeRollback                          release_start([&] { start.set_value(); });
    block_tree_cache_detail::ScopeRollback                          cleanup([&] {
        release_start.run();
        for (auto& submitter : submitters)
            if (submitter.valid())
                submitter.wait();
        io->release();
        engine->shutdown();
    });
    for (size_t i = 0; i < hosts.size(); ++i)
        submitters.push_back(std::async(std::launch::async, [&, i] {
            ready.wait();
            return engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[i], disks[i])}));
        }));
    release_start.run();
    size_t accepted = 0;
    for (auto& submitter : submitters) {
        auto context = submitter.get();
        if (context->done()) {
            EXPECT_EQ(context->result().copy_error, CacheCopyError::NOT_EXECUTED);
        } else {
            ++accepted;
        }
    }
    EXPECT_EQ(accepted, 5u);
    EXPECT_EQ(engine->queueSizes().background, 5u);
    cleanup.run();
    EXPECT_EQ(engine->queueSizes().background, 0u);
    EXPECT_EQ(io->writes, 5u);
}

TEST_F(TransferWorkLoopTest, EngineIdleWaitIncludesCallbackSuccessorOnAnEarlierOwner) {
    initialize(4, 1, 1, 2);
    engine->next_evict_.store(1);
    io->blocked = true;
    std::promise<void>                                  parent_entered, release_parent, wait_started;
    auto                                                entered         = parent_entered.get_future();
    auto                                                parent_released = release_parent.get_future();
    auto                                                started         = wait_started.get_future();
    std::promise<std::shared_ptr<TransferAsyncContext>> child_submitted;
    auto                                                child = child_submitted.get_future();
    std::future<void>                                   idle;
    block_tree_cache_detail::ScopeRollback              unblock_parent([&] { release_parent.set_value(); });
    block_tree_cache_detail::ScopeRollback              cleanup([&] {
        unblock_parent.run();
        child.wait_for(5s);
        io->release();
        engine->shutdown();
        if (idle.valid())
            idle.wait();
    });
    auto                                                parent = engine->execute(writes());
    parent->onDone([&](TransferResult result) {
        EXPECT_TRUE(result.ok());
        parent_entered.set_value();
        EXPECT_EQ(parent_released.wait_for(5s), std::future_status::ready);
        {
            std::lock_guard<std::mutex> lock(io->mutex);
            io->blocked = true;
        }
        child_submitted.set_value(
            engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[1], disks[1])})));
    });
    io->release();
    ASSERT_EQ(entered.wait_for(5s), std::future_status::ready);
    idle = std::async(std::launch::async, [&] {
        wait_started.set_value();
        engine->waitForIdle();
    });
    ASSERT_EQ(started.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(idle.wait_for(20ms), std::future_status::timeout);
    unblock_parent.run();
    ASSERT_EQ(child.wait_for(5s), std::future_status::ready);
    ASSERT_TRUE(io->waitWrites(2));
    EXPECT_EQ(idle.wait_for(20ms), std::future_status::timeout);
    io->release();
    ASSERT_EQ(idle.wait_for(5s), std::future_status::ready);
    idle.get();
    EXPECT_TRUE(child.get()->success());
    EXPECT_EQ(engine->queueSizes().background, 0u);
    // The future was consumed above; cleanup only needs to drain the owners.
    cleanup.dismiss();
    engine->shutdown();
}

TEST_F(TransferWorkLoopDeathTest, EntryBadAllocIsFatal) {
    EXPECT_EXIT(
        {
            TransferWorkLoop owner({group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 1, 8, {});
            TransferWorkLoop::Entry entry(writes(), {});
            prepareOomDeathTest("exercising Entry OOM boundary");
            owner.withEntry(entry, [] { throw std::bad_alloc(); });
            _exit(0);
        },
        ::testing::KilledBySignal(SIGABRT),
        "exercising Entry OOM boundary");
}

TEST_F(TransferWorkLoopDeathTest, BeforeCompleteCallbackBadAllocIsFatal) {
    EXPECT_EXIT(
        {
            TransferWorkLoop owner({group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 1, 8, {});
            owner.start();
            prepareOomDeathTest("exercising before_complete OOM boundary");
            auto context = owner.execute(writes(), [](const TransferResult&) { throw std::bad_alloc(); });
            if (!await(context))
                _exit(2);
            owner.shutdown();
            _exit(0);
        },
        ::testing::KilledBySignal(SIGABRT),
        "exercising before_complete OOM boundary");
}

TEST_F(TransferWorkLoopTest, OrdinaryEntryExceptionCompletesOnlyThatTask) {
    TransferWorkLoop owner({group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 1, 8, {});
    auto             context = owner.execute(writes());
    owner.entries_.splice(owner.entries_.end(), owner.incoming_);
    auto& entry    = **owner.entries_.begin();
    entry.position = owner.entries_.begin();
    ASSERT_TRUE(owner.plan(entry));
    owner.withEntry(entry, [] { throw std::runtime_error("batch preparation failed"); });
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().ToString(), "batch preparation failed");
    EXPECT_EQ(owner.outstandingBatches(), 0u);
}

TEST_F(TransferWorkLoopTest, SuppliedBatchLimitStaysFixedWhileStagingIsLeased) {
    struct Limits {
        size_t blocks, expected;
    };
    for (const auto limits : {Limits{0, 8}, Limits{2, 2}, Limits{3, 3}, Limits{32, 8}}) {
        SCOPED_TRACE(limits.blocks);
        auto pool =
            limits.blocks ? std::make_shared<HostStagingBlockPool>(limits.blocks, disk->strideBytes()) : nullptr;
        TransferWorkLoop owner(
            {group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, limits.expected, 8, pool);
        EXPECT_EQ(owner.batchLimit(), limits.expected);
        // A fully occupied pool does not change the cached admission geometry.
        const auto leases = pool ? pool->tryMallocBatch(limits.blocks) : std::nullopt;
        EXPECT_EQ(leases.has_value(), bool(pool));
        EXPECT_EQ(owner.batchLimit(), limits.expected);
    }
}

TEST_F(TransferWorkLoopTest, FixedBatchLimitAlsoBoundsRoutesWithoutStaging) {
    initialize(3, 2);
    io->blocked = true;
    block_tree_cache_detail::ScopeRollback cleanup([&] {
        io->release();
        engine->shutdown();
    });
    auto                                   context = engine->execute(writes(3));
    ASSERT_TRUE(io->waitWrites());
    // PerRank splits the request before either physical batch reaches its owner.
    EXPECT_EQ(engine->evict_loops_.front()->outstandingBatches(), 2u);
    EXPECT_FALSE(context->done());
    cleanup.run();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(io->writes, 3u);
}

TEST_F(TransferWorkLoopTest, LegacyLocalRequestGroupsBeforeIndependentBatchExecution) {
    auto second = makeTestGroupSet(1, group->topologyPtr(), {"group0"}, {}, host, disk);
    engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, second},
                                                          true,
                                                          DeviceHostCopyOptions{},
                                                          128,
                                                          2,
                                                          2,
                                                          8,
                                                          nullptr,
                                                          0,
                                                          1,
                                                          false,
                                                          diskOptions());
    ASSERT_FALSE(engine->asyncEnabled());
    auto context = executeRequest(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[4], disks[0]),
                                                    TransferDescriptor::hostToDisk(1, hosts[3], disks[1]),
                                                    TransferDescriptor::hostToDisk(0, hosts[2], disks[2]),
                                                    TransferDescriptor::hostToDisk(1, hosts[1], disks[3]),
                                                    TransferDescriptor::hostToDisk(0, hosts[0], disks[4])}));
    ASSERT_TRUE(await(context));
    request_dispatcher->shutdown();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(context->result().block_errors, std::vector<CacheCopyError>(5, CacheCopyError::NONE));
    EXPECT_EQ(io->writes, 5u);
    for (size_t i = 0; i < 5; ++i)
        EXPECT_EQ(io->data[disk->blockOffset(disks[i])], 21u - i);
}

TEST_F(TransferWorkLoopTest, PureDiskResultsAndCallbacksBelongToOwner) {
    initialize();
    auto* source = static_cast<uint8_t*>(host->blockBuffer(hosts[2]).addr);
    std::memset(source + group->storageBytes(), 0xa5, disk->strideBytes() - group->storageBytes());
    io->blocked  = true;
    auto context = executeRequest(writes(3));
    ASSERT_TRUE(io->waitWrites());
    std::thread::id callback_thread;
    context->onDone([&](TransferResult result) {
        EXPECT_TRUE(result.ok());
        callback_thread = std::this_thread::get_id();
    });
    io->release();
    ASSERT_TRUE(await(context));
    // Inspect the physical record after the accepted eviction has completed.
    std::memset(host->blockBuffer(hosts[3]).addr, 0, 384);
    ASSERT_EQ(disk->read(disks[2], host->blockBuffer(hosts[3]).addr, disk->strideBytes()), BlockIOStatus::OK);
    engine->shutdown();
    EXPECT_NE(callback_thread, io->worker);
    EXPECT_NE(callback_thread, std::this_thread::get_id());
    EXPECT_EQ(std::memcmp(host->blockBuffer(hosts[2]).addr, host->blockBuffer(hosts[3]).addr, 384), 0);
    for (size_t i = group->storageBytes(); i < disk->strideBytes(); ++i)
        EXPECT_EQ(io->data[disk->blockOffset(disks[2]) + i], 0xa5u);
}

TEST_F(TransferWorkLoopTest, DiskResultPublicationWaitsForNotificationMutex) {
    TransferWorkLoop        owner({group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 2, 8, {});
    TransferWorkLoop::Entry entry(writes(2), {});
    entry.step = TransferWorkLoop::Step::DISK_WRITE;
    std::promise<void> started;
    auto               started_future = started.get_future();
    std::future<void>  callback;
    {
        std::unique_lock<std::mutex> lock(owner.mutex_);
        callback = std::async(std::launch::async, [&] {
            started.set_value();
            owner.publishDisk(entry, BlockIOStatus::IO_ERROR);
        });
        ASSERT_EQ(started_future.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(callback.wait_for(20ms), std::future_status::timeout);
        EXPECT_TRUE(owner.disk_completed_.empty());
    }
    ASSERT_EQ(callback.wait_for(5s), std::future_status::ready);
    callback.get();
    EXPECT_EQ(owner.disk_completed_.head, &entry);
    EXPECT_EQ(entry.result.status.code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_TRUE(entry.result.status.ToString().empty());  // The owner adds the error text.
    EXPECT_EQ(entry.result.copy_error, CacheCopyError::IO_FAILED);
    EXPECT_EQ(entry.result.block_errors, std::vector<CacheCopyError>(2, CacheCopyError::IO_FAILED));
    EXPECT_FALSE(owner.disk_completed_.empty());
    EXPECT_EQ(owner.disk_completed_.pop(), &entry);
}

TEST_F(TransferWorkLoopTest, HostToDiskEvictionRunsOnEvictOwner) {
    initialize();
    io->blocked  = true;
    auto context = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[0], disks[0])}));
    ASSERT_TRUE(io->waitWrites());
    EXPECT_EQ(engine->queueSizes().load, 0u);
    EXPECT_EQ(engine->queueSizes().background, 1u);
    std::thread::id completing_thread;
    context->onDone([&](TransferResult) { completing_thread = std::this_thread::get_id(); });
    const auto owner_thread = engine->evict_loops_.front()->thread_.get_id();
    io->release();
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(completing_thread, owner_thread);
}

TEST_F(TransferWorkLoopTest, DiskQueuePreservesCompletionsPublishedDuringConsumption) {
    TransferWorkLoop owner({group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 2, 8, {});
    auto             first     = owner.execute(writes());
    auto             second    = owner.execute(writes());
    size_t           callbacks = 0;
    // Drive submission directly so the second completion can be published while
    // the owner is consuming the first result, after its Disk queue has been taken.
    owner.entries_.splice(owner.entries_.end(), owner.incoming_);
    for (auto position = owner.entries_.begin(); position != owner.entries_.end(); ++position) {
        auto& entry    = **position;
        entry.position = position;
        ASSERT_TRUE(owner.plan(entry));
    }
    auto* second_entry = owner.entries_.back().get();
    first->onDone([&](TransferResult) {
        ++callbacks;
        owner.publishDisk(*second_entry, BlockIOStatus::IO_ERROR);
    });
    second->onDone([&](TransferResult) { ++callbacks; });
    owner.pollOnce();
    EXPECT_FALSE(first->done());
    EXPECT_FALSE(second->done());
    owner.publishDisk(*owner.entries_.front(), BlockIOStatus::OK);
    owner.pollOnce();
    EXPECT_TRUE(first->success());
    EXPECT_FALSE(second->done());
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(owner.outstandingBatches(), 1u);
    EXPECT_FALSE(owner.disk_completed_.empty());
    // The notification published during completion must survive into the next
    // turn, including when admission closes and the owner drains.
    owner.start();
    auto drain = std::async(std::launch::async, [&] { owner.shutdown(); });
    ASSERT_EQ(drain.wait_for(5s), std::future_status::ready);
    drain.get();
    EXPECT_EQ(second->result().copy_error, CacheCopyError::IO_FAILED);
    EXPECT_EQ(second->result().status.ToString(), "disk block transfer failed");
    EXPECT_EQ(owner.outstandingBatches(), 0u);
    EXPECT_TRUE(owner.disk_completed_.empty());
    EXPECT_EQ(callbacks, 2u);
}

TEST_F(TransferWorkLoopTest, ResidentDiskBatchesFitDerivedWorkerQueues) {
    constexpr size_t q = 4, batch_size = 2;
    for (bool write : {false, true}) {
        SCOPED_TRACE(write ? "write" : "read");
        enableMetrics();
        auto config               = diskOptions();
        config.read_thread_count  = 3;
        config.write_thread_count = 2;
        config.read_queue_size = config.write_queue_size = 1;
        engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                              true,
                                                              DeviceHostCopyOptions{},
                                                              2,
                                                              batch_size,
                                                              2,
                                                              q,
                                                              nullptr,
                                                              0,
                                                              1,
                                                              true,
                                                              config);
        EXPECT_EQ(engine->diskIOExecutor()->config().read_queue_size, 8u);
        EXPECT_EQ(engine->diskIOExecutor()->config().write_queue_size, 4u);
        const auto source = write ? Tier::HOST : Tier::DISK;
        const auto target = write ? Tier::DISK : Tier::HOST;
        const auto task   = [&](size_t offset, size_t count) {
            std::vector<TransferDescriptor> descriptors;
            for (size_t i = offset; i < offset + count; ++i)
                descriptors.push_back(write ? TransferDescriptor::hostToDisk(0, hosts[i], disks[i]) :
                                              physicalDiskRead(0, disks[i], hosts[i]));
            return makeTransferTask(std::move(descriptors));
        };
        TransferWorkLoop                       owner({group},
                               nullptr,
                               engine->disk_io_executor_.get(),
                               write ? TransferDirection::EVICT : TransferDirection::LOAD,
                               batch_size,
                               q,
                                                     {},
                               reporter,
                               3);
        bool                                   started = false;
        block_tree_cache_detail::ScopeRollback finish([&] {
            io->release();
            if (!started) {
                owner.start();
                started = true;
            }
            owner.shutdown();
            engine->shutdown();
        });
        for (size_t i = 0; i < hosts.size(); ++i) {
            std::memset(io->data.data() + disk->blockOffset(disks[i]), write ? 0 : int(i + 17), 384);
            std::memset(host->blockBuffer(hosts[i]).addr, write ? int(i + 17) : 0, 384);
        }
        io->blocked = true;
        std::vector<std::shared_ptr<TransferAsyncContext>> contexts;
        for (size_t i = 0; i < q; ++i)
            contexts.push_back(owner.execute(task(i * batch_size, batch_size)));
        EXPECT_EQ(owner.outstandingBatches(), q);
        // Use the engine's derived budget. Drive all submissions before starting
        // this owner so blocked disk workers cannot hide an undersized queue.
        owner.entries_.splice(owner.entries_.end(), owner.incoming_);
        for (auto position = owner.entries_.begin(); position != owner.entries_.end(); ++position) {
            auto& entry    = **position;
            entry.position = position;
            ASSERT_TRUE(owner.plan(entry));
            owner.submitDisk(entry);
        }
        ASSERT_TRUE(write ? io->waitWrites() : io->waitReads());
        EXPECT_EQ(owner.entries_.size(), q);
        EXPECT_EQ(owner.outstandingBatches(), q);
        EXPECT_EQ(owner.execute(task(0, 1))->result().copy_error, CacheCopyError::NOT_EXECUTED);
        finish.run();
        EXPECT_EQ(write ? io->writes : io->reads, q * batch_size);
        for (const auto& context : contexts)
            EXPECT_TRUE(context->success());
        for (size_t i = 0; i < hosts.size(); ++i) {
            EXPECT_EQ(io->data[disk->blockOffset(disks[i])], i + 17);
            EXPECT_EQ(static_cast<const uint8_t*>(host->blockBuffer(hosts[i]).addr)[0], i + 17);
        }
        EXPECT_EQ(owner.execute(task(0, 1))->result().copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_EQ(owner.execute(task(0, 8))->result().copy_error, CacheCopyError::NOT_EXECUTED);
        reporter->flushStageMetrics();
        EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 5u);
        checkStage("worker_task_total", q, true, source, target, "worker_task");
        checkStage("worker_task_total", 3, false, source, target, "worker_task");
        checkStage("transfer_queue_wait", q, true, source, target, "worker_task");
        checkStage("transfer_queue_wait", 3, false, source, target, "worker_task");
        checkStage(write ? "disk_write" : "disk_read", q, true, source, target);
        checkQueueWait(q, source, target);
    }
}

TEST_F(TransferWorkLoopTest, ResidentBatchesIncludeInflightCallbackAndEntryCleanup) {
    enableMetrics();
    initialize(2, 2);
    BlockTransferDispatcher dispatcher(engine);
    const auto              check_backlog = [&](size_t expected) {
        const auto sizes = dispatcher.queueSizes();
        EXPECT_EQ(sizes.background, expected);
        EXPECT_EQ(sizes.load, 0u);
        EXPECT_EQ(sizes.completion, 0u);
        reporter->reportQueueBacklog(sizes, "transfer");
        kmonitor::MetricsTags tags("pool_type", "transfer");
        tags.AddTag("queue_type", "background");
        EXPECT_DOUBLE_EQ(snapshot(metrics()->task_queue_backlog_metric, tags), expected);
        EXPECT_EQ(seriesCount(metrics()->task_queue_backlog_metric), 3u);
    };
    io->blocked  = true;
    auto context = engine->execute(writes(2));
    ASSERT_TRUE(io->waitWrites());
    check_backlog(1);  // In-flight work remains in the existing backlog series.
    auto invalid  = TransferDescriptor::hostToDisk(0, std::numeric_limits<BlockIdxType>::max(), disks[0]);
    auto rejected = engine->execute(makeTransferTask({invalid}));
    EXPECT_TRUE(rejected->done());
    EXPECT_EQ(rejected->result().copy_error, CacheCopyError::INVALID_REQUEST);
    std::promise<void>                     entered, release;
    auto                                   entered_future = entered.get_future();
    auto                                   released       = release.get_future();
    block_tree_cache_detail::ScopeRollback unblock([&] { release.set_value(); });
    context->onDone([&](TransferResult result) {
        EXPECT_TRUE(result.ok());
        entered.set_value();
        EXPECT_EQ(released.wait_for(5s), std::future_status::ready);
    });
    io->release();
    ASSERT_EQ(entered_future.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(context->done());
    auto waiting = engine->execute(writes());
    EXPECT_FALSE(waiting->done());
    check_backlog(2);  // Callback and incoming Entry each retain one resident slot.
    EXPECT_EQ(engine->execute(writes())->result().copy_error, CacheCopyError::NOT_EXECUTED);
    auto drain = std::async(std::launch::async, [&] { engine->shutdown(); });
    EXPECT_EQ(drain.wait_for(20ms), std::future_status::timeout);
    unblock.run();
    ASSERT_EQ(drain.wait_for(5s), std::future_status::ready);
    drain.get();
    check_backlog(0);
    EXPECT_TRUE(waiting->success());
}

TEST_F(TransferWorkLoopTest, AdmissionAndQueueTimeoutsDoNotStartIo) {
    enableMetrics();
    size_t           callbacks = 0;
    TransferWorkLoop owner(
        {group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 5, 3, {}, reporter, 3);
    const auto expect_timeout = [](const std::shared_ptr<TransferAsyncContext>& context) {
        EXPECT_TRUE(context->done());
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_EQ(context->result().block_errors, std::vector<CacheCopyError>(5, CacheCopyError::NOT_EXECUTED));
    };
    auto task      = writes(5);
    task.deadline_ = TransferTask::Clock::now() - 1s;
    expect_timeout(owner.execute(std::move(task)));
    EXPECT_EQ(owner.entry_count_, 0u);

    task           = writes(5);
    task.deadline_ = TransferTask::Clock::now() + 1h;
    auto context   = owner.execute(std::move(task));
    ASSERT_FALSE(context->done());
    ASSERT_EQ(owner.entry_count_, 1u);
    // The owner has not started: expire an accepted task without a wall-clock race.
    owner.incoming_.front()->task.deadline_ = TransferTask::Clock::now() - 1s;
    context->onDone([&](TransferResult result) {
        EXPECT_EQ(result.status.code(), ErrorCode::DEADLINE_EXCEEDED);
        ++callbacks;
    });
    owner.start();
    ASSERT_TRUE(await(context));
    {
        std::unique_lock<std::mutex> lock(owner.mutex_);
        ASSERT_TRUE(owner.cv_.wait_for(lock, 5s, [&] { return owner.entry_count_ == 0; }));
    }
    expect_timeout(context);
    EXPECT_EQ(io->writes, 0u);
    // The whole resident budget must be reusable after timeout cleanup.
    auto successor = owner.execute(writes(5));
    ASSERT_TRUE(await(successor));
    EXPECT_TRUE(successor->success());
    owner.shutdown();
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(owner.entry_count_, 0u);
    EXPECT_EQ(io->writes, 5u);  // Only the successor touched Disk.
    EXPECT_EQ(io->reads, 0u);
    reporter->flushStageMetrics();
    EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 5u);
    checkStage("worker_task_total", 2, false, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("worker_task_total", 1, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("transfer_queue_wait", 2, false, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("transfer_queue_wait", 1, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("disk_write", 1);
    checkQueueWait(2);  // The pre-admission timeout is not a consumed queue task.
}

TEST_F(TransferWorkLoopTest, StartedExpiredTaskAndDroppedContextStillFinishAfterStop) {
    TransferWorkLoop owner({group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 5, 8, {});
    block_tree_cache_detail::ScopeRollback finish([&] {
        if (!owner.thread_.joinable())
            owner.start();
        io->release();
        owner.shutdown();
    });
    io->blocked  = true;
    auto context = owner.execute(writes(5));
    // Start the physical batch before expiring it. A started batch must finish
    // even when admission closes and its consumer drops the context.
    owner.entries_.splice(owner.entries_.end(), owner.incoming_);
    auto& entry    = **owner.entries_.begin();
    entry.position = owner.entries_.begin();
    owner.tryStep(entry);
    ASSERT_TRUE(io->waitWrites());
    entry.task.deadline_ = TransferTask::Clock::now() - 1s;
    std::promise<TransferResult> completed;
    auto                         future = completed.get_future();
    context->onDone([&](TransferResult result) { completed.set_value(std::move(result)); });
    context.reset();
    owner.start();
    owner.stopAdmission();
    finish.run();
    ASSERT_EQ(future.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(future.get().ok());
    EXPECT_EQ(io->writes, 5u);
    EXPECT_EQ(io->data[disk->blockOffset(disks[4])], 21u);
}

TEST_F(TransferWorkLoopTest, DispatcherShutdownWaitsForDirectExecuteCallbackAndOwnerCleanup) {
    initialize();
    BlockTransferDispatcher dispatcher(engine);
    io->blocked = true;
    // Direct execute is not registered in the Dispatcher's callback-chain table.
    auto context = dispatcher.executePerRank(writes());
    ASSERT_TRUE(io->waitWrites());
    std::promise<void>                     entered, release;
    auto                                   entered_future = entered.get_future();
    auto                                   released       = release.get_future();
    block_tree_cache_detail::ScopeRollback unblock([&] { release.set_value(); });
    context->onDone([&](TransferResult result) {
        EXPECT_TRUE(result.ok());
        entered.set_value();
        EXPECT_EQ(released.wait_for(5s), std::future_status::ready);
    });
    io->release();
    ASSERT_EQ(entered_future.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(context->done());
    auto shutdown = std::async(std::launch::async, [&] { dispatcher.shutdown(); });
    EXPECT_EQ(shutdown.wait_for(20ms), std::future_status::timeout);
    unblock.run();
    ASSERT_EQ(shutdown.wait_for(5s), std::future_status::ready);
    shutdown.get();
    EXPECT_EQ(dispatcher.queueSizes().background, 0u);
    auto rejected = dispatcher.executePerRank(writes());
    EXPECT_TRUE(rejected->done());
    EXPECT_EQ(rejected->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_NO_THROW(dispatcher.shutdown());
}

TEST_F(TransferWorkLoopTest, CallbackSubmitsSuccessorWithoutReenteringOwner) {
    initialize();
    io->blocked  = true;
    auto context = engine->execute(writes());
    ASSERT_TRUE(io->waitWrites());
    bool               in_callback = false;
    std::promise<void> successor_done;
    auto               future = successor_done.get_future();
    context->onDone([&](TransferResult) {
        in_callback = true;
        auto next   = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[1], disks[1])}));
        EXPECT_FALSE(next->done());
        next->onDone([&](TransferResult result) {
            EXPECT_FALSE(in_callback);
            EXPECT_TRUE(result.ok());
            successor_done.set_value();
        });
        in_callback = false;
    });
    io->release();
    ASSERT_EQ(future.wait_for(5s), std::future_status::ready);
    engine->shutdown();
    EXPECT_EQ(std::memcmp(host->blockBuffer(hosts[1]).addr, io->data.data() + disk->blockOffset(disks[1]), 384), 0);
}

TEST_F(TransferWorkLoopTest, ClosingRejectsCallbackDerivedTransferSafely) {
    initialize();
    io->blocked  = true;
    auto context = engine->execute(writes());
    ASSERT_TRUE(io->waitWrites());
    std::atomic<size_t> callbacks{0};
    context->onDone([&](TransferResult result) {
        EXPECT_TRUE(result.ok());
        auto next = engine->execute(writes());
        EXPECT_TRUE(next->done());
        next->onDone([&](TransferResult rejected) {
            EXPECT_EQ(rejected.copy_error, CacheCopyError::NOT_EXECUTED);
            ++callbacks;
        });
    });
    engine->stopAdmission();
    io->release();
    engine->shutdown();
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(io->writes, 1u);
}

TEST_F(TransferWorkLoopTest, ThrowingCallbackDoesNotLoseOtherCallbacksOrAdmission) {
    initialize();
    io->blocked  = true;
    auto context = engine->execute(writes());
    ASSERT_TRUE(io->waitWrites());
    std::atomic<size_t> calls{0};
    context->onDone([](TransferResult) { throw std::runtime_error("consumer failed"); });
    context->onDone([&](TransferResult) { ++calls; });
    io->release();
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(engine->queueSizes().background, 0u);
}

TEST_F(TransferWorkLoopTest, DiskRejectionHasNoCallbackAndIsNotRetried) {
    enableMetrics();
    initialize();
    engine->disk_io_executor_->stop();
    auto context = engine->execute(writes());
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(context->result().copy_error, CacheCopyError::IO_FAILED);
    EXPECT_EQ(context->result().status.ToString(), "disk block transfer failed");
    EXPECT_EQ(io->writes, 0u);
    reporter->flushStageMetrics();
    EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 3u);
    checkStage("disk_write", 1, false);
    checkStage("worker_task_total", 1, false, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("transfer_queue_wait", 1, true, Tier::HOST, Tier::DISK, "worker_task");
    checkQueueWait(1);
}

TEST_F(TransferWorkLoopTest, RequestJoinsIndependentBatchFailuresInOriginalOrder) {
    enableMetrics();
    initialize();
    io->fail_write = 3;
    auto context   = executeRequest(writes(5));
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NONE,
                                           CacheCopyError::NONE,
                                           CacheCopyError::IO_FAILED,
                                           CacheCopyError::IO_FAILED,
                                           CacheCopyError::NONE}));
    EXPECT_EQ(context->result().status.ToString(), "disk block transfer failed");
    EXPECT_EQ(io->writes, 4u);
    reporter->flushStageMetrics();
    EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 5u);
    checkStage("disk_write", 2);
    checkStage("disk_write", 1, false);
    checkStage("worker_task_total", 2, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("worker_task_total", 1, false, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("transfer_queue_wait", 3, true, Tier::HOST, Tier::DISK, "worker_task");
    EXPECT_EQ(seriesCount(metrics()->transfer_qps_metric), 0u);
    EXPECT_EQ(seriesCount(metrics()->host_cache_copy_error_qps_metric), 0u);
}

TEST_F(TransferWorkLoopTest, PartiallyAdmittedRequestWaitsForAcceptedBatchBeforeCompleting) {
    initialize(1, 2);
    io->blocked  = true;
    auto context = executeRequest(writes(5));
    ASSERT_TRUE(io->waitWrites());
    EXPECT_FALSE(context->done());
    EXPECT_EQ(engine->queueSizes().background, 1u);
    size_t calls = 0;
    context->onDone([&](TransferResult) { ++calls; });
    io->release();
    ASSERT_TRUE(await(context));
    request_dispatcher->shutdown();
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(engine->queueSizes().background, 0u);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NONE,
                                           CacheCopyError::NONE,
                                           CacheCopyError::NOT_EXECUTED,
                                           CacheCopyError::NOT_EXECUTED,
                                           CacheCopyError::NOT_EXECUTED}));
    EXPECT_EQ(io->writes, 2u);
}

TEST_F(TransferWorkLoopTest, RepeatedDiskOnlyIdleWakeupsDoNotNeedPollingTimer) {
    initialize();
    for (size_t i = 0; i < 32; ++i) {
        auto context = engine->execute(writes());
        ASSERT_TRUE(await(context));
        EXPECT_TRUE(context->success());
    }
    engine->shutdown();
    EXPECT_EQ(io->writes, 32u);
}

TEST_F(TransferWorkLoopTest, QueueAndDescriptorCapacityRejectOverflowAndMalformedGeometry) {
    EXPECT_THROW((PerRankBlockTransferEngine({}, false, {}, 4, 2, 1, 0, nullptr, 0, 1, true)), std::invalid_argument);
    EXPECT_THROW((PerRankBlockTransferEngine({}, false, {}, 4, 2, 3, 2, nullptr, 0, 1, true)), std::invalid_argument);
    EXPECT_THROW(
        (PerRankBlockTransferEngine({}, false, {}, 4, 2, 1, std::numeric_limits<size_t>::max(), nullptr, 0, 1, true)),
        std::invalid_argument);
    PerRankBlockTransferEngine legacy({}, false, {}, 4, 2, 3, 2, nullptr, 0, 1, false);
    EXPECT_FALSE(legacy.asyncEnabled());  // Legacy does not statically partition Q.
    initialize(1, 2);
    auto invalid = TransferDescriptor::hostToDisk(0, hosts[0], disks[0]);
    invalid.source_blocks.push_back(hosts[1]);
    EXPECT_EQ(engine->execute(makeTransferTask({invalid}))->result().copy_error, CacheCopyError::INVALID_REQUEST);
    invalid.group_set_id = std::numeric_limits<size_t>::max();
    EXPECT_EQ(engine->execute(makeTransferTask({invalid}))->result().copy_error, CacheCopyError::INVALID_REQUEST);
}

TEST_F(TransferWorkLoopTest, PerRankRejectsInvalidDescriptorsBeforeAdmission) {
    initialize();
    const std::vector<TransferDescriptor> invalid{
        physicalDiskRead(0, disks[0], hosts[0]),
        TransferDescriptor::hostToDevice(0, hosts[0], {1}),
        TransferDescriptor::hostToDisk(0, hosts[0], std::numeric_limits<BlockIdxType>::max()),
        TransferDescriptor::hostToDisk(std::numeric_limits<size_t>::max(), hosts[0], disks[0]),
    };
    for (const auto& descriptor : invalid) {
        auto context = engine->execute(makeTransferTask({descriptor}));
        ASSERT_TRUE(context->done());
        EXPECT_EQ(context->result().copy_error, CacheCopyError::INVALID_REQUEST);
    }
    EXPECT_EQ(engine->queueSizes().load, 0u);
    EXPECT_EQ(engine->queueSizes().background, 0u);
    EXPECT_EQ(io->reads, 0u);
    EXPECT_EQ(io->writes, 0u);
}

TEST_F(TransferWorkLoopTest, LocalRequestGroupsBatchesWithoutChangingResultOrder) {
    auto second  = makeTestGroupSet(1, group->topologyPtr(), {"group0"}, {}, host, disk);
    engine       = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, second},
                                                          true,
                                                          DeviceHostCopyOptions{},
                                                          4,
                                                          2,
                                                          1,
                                                          4,
                                                          nullptr,
                                                          0,
                                                          1,
                                                          true,
                                                          diskOptions());
    auto context = executeRequest(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[2], disks[0]),
                                                    TransferDescriptor::hostToDisk(1, hosts[1], disks[1]),
                                                    TransferDescriptor::hostToDisk(0, hosts[0], disks[2])}));
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(context->result().block_errors, std::vector<CacheCopyError>(3, CacheCopyError::NONE));
    EXPECT_EQ(io->data[disk->blockOffset(disks[0])], 19u);
    EXPECT_EQ(io->data[disk->blockOffset(disks[1])], 18u);
    EXPECT_EQ(io->data[disk->blockOffset(disks[2])], 17u);
}

TEST_F(TransferWorkLoopTest, BeforeCompleteCallbackRunsBeforeReadyAndOutsideInputLock) {
    enableMetrics();
    for (bool with_reporter : {true, false}) {
        SCOPED_TRACE(with_reporter);
        io->blocked              = true;
        const auto writes_before = io->writes;
        auto       session       = AsyncProfileSession::start(100000);
        ASSERT_TRUE(session);
        TransferWorkLoop owner({group},
                               nullptr,
                               direct_disk_executor.get(),
                               TransferDirection::EVICT,
                               2,
                               8,
                               {},
                               with_reporter ? reporter : nullptr,
                               3);
        owner.start();
        std::shared_ptr<TransferAsyncContext> context;
        int64_t                               before_complete_begin = 0;
        context = owner.execute(writes(), [&](const TransferResult& result) {
            before_complete_begin = AsyncProfileSession::nowUs();
            EXPECT_TRUE(result.ok());
            EXPECT_FALSE(context->done());
            const bool unlocked = owner.mutex_.try_lock();
            EXPECT_TRUE(unlocked);
            if (unlocked)
                owner.mutex_.unlock();
        });
        ASSERT_TRUE(io->waitWrites(writes_before + 1));
        context->onDone([&](TransferResult) { EXPECT_NE(before_complete_begin, 0); });
        const auto io_release = AsyncProfileSession::nowUs();
        io->release();
        ASSERT_TRUE(await(context));
        owner.shutdown();
        const auto events = session->finish();
        ASSERT_EQ(events.size(), 3u);  // Task, queue wait and Disk lifecycle only.
        checkWorkLoopTrace(events);
        for (const auto& event : events) {
            EXPECT_EQ(event.name.find("incomplete=1"), std::string::npos);
            EXPECT_LE(event.end_us, before_complete_begin);
            if (event.name.find(".disk_write(") != std::string::npos) {
                EXPECT_LE(event.start_us, io_release);
                EXPECT_GE(event.end_us, io_release);
            }
        }
        const std::string native =
            R"({"traceEvents":[{"ph":"X","name":"native_gpu","cat":"kernel","pid":0,"tid":7,"ts":1,"dur":2}]})";
        const auto json = mergeAsyncProfileTrace(native, events);
        EXPECT_NE(json.find("native_gpu"), std::string::npos);
        EXPECT_EQ(json.find("native_gpu", json.find("native_gpu") + 1), std::string::npos);
        EXPECT_NE(json.find("disk_write"), std::string::npos);
        EXPECT_NE(json.find("KV background"), std::string::npos);
        EXPECT_NE(asyncProfileWorkflowTrace(native, events).find("KV background"), std::string::npos);
    }
    reporter->flushStageMetrics();
    checkStage("worker_task_total", 1, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("disk_write", 1);
}

TEST_F(TransferWorkLoopTest, TraceOnlyMultipleOwnersPreserveBatchWorkflow) {
    initialize(8, 2, 1, 2);
    auto session = AsyncProfileSession::start(100000);
    ASSERT_TRUE(session);
    TransferTask request({}, 30s, 42);
    request.setWorkflow(1234, 5678);
    const auto          transfer_id = request.traceId();
    std::atomic<size_t> callbacks{0};
    for (size_t round = 0; round < 2; ++round) {
        for (size_t i = 0; i < 8; ++i) {
            auto task    = request.subtask({TransferDescriptor::hostToDisk(0, hosts[i], disks[i])});
            auto context = engine->execute(std::move(task));
            context->onDone([&](TransferResult result) {
                EXPECT_TRUE(result.ok());
                ++callbacks;
            });
        }
        engine->waitForIdle();
    }
    engine->shutdown();
    const auto events = session->finish();
    ASSERT_EQ(events.size(), 48u);  // Three intervals per task, regardless of worker count.
    checkWorkLoopTrace(events);
    EXPECT_EQ(callbacks, 16u);
    std::set<uint64_t> spans;
    size_t             reads = 0, writes = 0;
    for (const auto& event : events) {
        EXPECT_EQ(event.record.request_id, 42);
        EXPECT_EQ(event.record.transfer_id, transfer_id);
        EXPECT_EQ(event.record.workflow_id, 1234u);
        EXPECT_EQ(event.record.parent_transfer_id, 5678u);
        EXPECT_EQ(event.record.rank, 3);
        EXPECT_EQ(event.record.begin, 0u);
        EXPECT_EQ(event.record.end, 1u);
        EXPECT_TRUE(spans.insert(event.span_id).second);
        reads += event.record.stage == "disk_read";
        writes += event.record.stage == "disk_write";
    }
    EXPECT_EQ(reads, 0u);
    EXPECT_EQ(writes, 16u);
    const auto json = mergeAsyncProfileTrace(R"({"traceEvents":[]})", events);
    // The exporter keeps request-owned logical intervals off physical CPU tracks.
    EXPECT_EQ(json.find("worker_task_total"), std::string::npos);
    EXPECT_EQ(json.find("KV background"), std::string::npos);
}

TEST_F(TransferWorkLoopTest, StageCapturePreservesMetricSampleCounts) {
    enableMetrics();
    initialize(8, 1, 1, 2);
    for (bool with_trace : {false, true}) {
        auto session = with_trace ? AsyncProfileSession::start(100000) : nullptr;
        for (size_t i = 0; i < 12; ++i) {
            auto context = engine->execute(writes());
            engine->waitForIdle();
            ASSERT_TRUE(context->success());
        }
        if (session) {
            const auto events = session->finish();
            checkWorkLoopTrace(events);
            EXPECT_EQ(events.size(), 36u);
        }
    }
    engine->shutdown();
    reporter->flushStageMetrics();
    checkStage("worker_task_total", 24, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("transfer_queue_wait", 24, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("disk_write", 24);
    checkQueueWait(24);
}

TEST_F(TransferWorkLoopTest, RejectedAdmissionAndTruncatedCaptureDoNotChangeCompletion) {
    TransferWorkLoop owner(
        {group}, nullptr, direct_disk_executor.get(), TransferDirection::EVICT, 2, 8, {}, nullptr, 3);
    owner.stopAdmission();
    auto session  = AsyncProfileSession::start(100000);
    auto rejected = owner.execute(writes());
    EXPECT_TRUE(rejected->done());
    EXPECT_FALSE(rejected->success());
    auto events = session->finish();
    ASSERT_EQ(events.size(), 2u);
    checkWorkLoopTrace(events);
    for (const auto& event : events)
        EXPECT_NE(event.record.kind, AsyncProfileKind::Execution);

    initialize();
    io->blocked  = true;
    session      = AsyncProfileSession::start(100000);
    auto context = engine->execute(writes());
    ASSERT_TRUE(io->waitWrites());
    events = session->finish();
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const auto& event) {
        return event.record.stage == "disk_write" && event.end_clipped;
    }));
    io->release();
    ASSERT_TRUE(await(context));
    engine->waitForIdle();
    session = AsyncProfileSession::start(1);
    context = engine->execute(writes());
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_TRUE(context->success());
    events = session->finish();
    EXPECT_TRUE(
        std::any_of(events.begin(), events.end(), [](const auto& event) { return event.record.stage == "dropped"; }));
}

TEST_F(TransferWorkLoopTest, InlineTimersDoNotAllocateSharedObserversOrRetainReporterReferences) {
    EXPECT_FALSE(AsyncProfileSession::enabled());
    enableMetrics();
    for (bool with_reporter : {false, true}) {
        TransferWorkLoop owner({group},
                               nullptr,
                               direct_disk_executor.get(),
                               TransferDirection::EVICT,
                               2,
                               8,
                               {},
                               with_reporter ? reporter : nullptr,
                               3);
        TransferStageTimer::allocation_probe_ =
            +[](const char*) { ADD_FAILURE() << "WorkLoop allocated a shared timer"; };
        block_tree_cache_detail::ScopeRollback restore([] { TransferStageTimer::allocation_probe_ = nullptr; });
        const auto                             reporter_owners = reporter.use_count();
        auto                                   context         = owner.execute(writes());
        owner.entries_.splice(owner.entries_.end(), owner.incoming_);
        auto& entry    = **owner.entries_.begin();
        entry.position = owner.entries_.begin();
        owner.tryStep(entry);
        EXPECT_EQ(bool(entry.task_timer), with_reporter);
        EXPECT_EQ(bool(entry.queue_timer), with_reporter);
        EXPECT_EQ(bool(entry.step_timer), with_reporter);
        EXPECT_EQ(reporter.use_count(), reporter_owners);
        restore.run();
        owner.start();
        ASSERT_TRUE(await(context));
        owner.shutdown();
        EXPECT_TRUE(context->success());
    }
    reporter->flushStageMetrics();
    checkStage("worker_task_total", 1, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("transfer_queue_wait", 1, true, Tier::HOST, Tier::DISK, "worker_task");
    checkStage("disk_write", 1);
    checkQueueWait(1);
}

TEST_F(TransferWorkLoopTest, DisabledStageMetricsSkipWorkLoopQueueAggregation) {
    enableMetrics(false);
    EXPECT_EQ(reporter->stage_metrics_, nullptr);
    initialize();
    auto context = engine->execute(writes());
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(reporter->stageMetricsEnabled());
    reporter->flushStageMetrics();
    EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 0u);
    EXPECT_EQ(seriesCount(metrics()->task_queue_waiting_tasks_metric), 0u);
    EXPECT_EQ(seriesCount(metrics()->transfer_task_queue_wait_latency_us_metric), 0u);
    // The existing direct reporter remains independent of the aggregation switch.
    reporter->reportQueueWaitMetric(false, "transfer", nullptr, Tier::HOST, Tier::DISK, 17);
    checkQueueWait(1);
}

class TransferWorkLoopCrcTest: public TransferWorkLoopTest {
protected:
    void SetUp() override {
        int count = 0;
        if (!CrcCopyWorkspace::available() || cudaGetDeviceCount(&count) != cudaSuccess || !count)
            GTEST_SKIP() << "requires a CUDA13 GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        direct_disk_executor = std::make_unique<DiskIOExecutor>(diskOptions());
        direct_disk_executor->start();
        host            = makeHostPool(384, 16, true);
        auto controlled = std::make_unique<ControlledDiskIO>();
        io              = controlled.get();
        disk            = makeDiskPool(384, 16, temp.path, std::move(controlled), "workloop_crc", true, true);
        device          = makeTestDevicePool({{384, 0}}, 32, "workloop_crc_device");
        group           = makeTestGroupSet(
            0,
            makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 384)}),
            {"group0"},
            {device},
            host,
            disk,
            true);
        for (size_t i = 0; i < 8; ++i) {
            hosts.push_back(poolMalloc(*host));
            disks.push_back(poolMalloc(*disk));
            sources.push_back(poolMalloc(*device));
            targets.push_back(poolMalloc(*device));
            const std::vector<uint8_t> bytes(384, uint8_t(i + 17));
            ASSERT_EQ(cudaMemcpy(buffer(sources.back()), bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
    }
    void TearDown() override {
        if (!io)
            return;
        TransferWorkLoopTest::TearDown();
        for (auto block : sources)
            releasePoolBlock(*device, block);
        for (auto block : targets)
            releasePoolBlock(*device, block);
    }
    void* buffer(BlockIdxType block) {
        return device->convertIndexToBuffer(0, block).front().addr;
    }
    void expectTarget(size_t i) {
        std::vector<uint8_t> bytes(384);
        ASSERT_EQ(cudaMemcpy(bytes.data(), buffer(targets[i]), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(bytes, std::vector<uint8_t>(384, uint8_t(i + 17)));
    }
    TransferTask evict(size_t i) {
        return makeTransferTask({TransferDescriptor::deviceToHost(0, {sources[i]}, hosts[i])});
    }
    void seedDisk(size_t i) {
        auto* record = io->data.data() + disk->blockOffset(disks[i]);
        std::memset(record, int(i + 17), 384);
        uint32_t crc = 0xffffffffu;
        for (size_t byte = 0; byte < 384; ++byte) {
            crc ^= record[byte];
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ ((crc & 1u) ? 0x82f63b78u : 0u);
        }
        crc = ~crc;
        std::memcpy(record + group->storageBytes() - 4, &crc, sizeof(crc));
    }
    DeviceBlockPoolPtr        device;
    DiskBlockPoolPtr          retained_controlled_disk;
    std::vector<BlockIdxType> sources, targets;
};

TEST_F(TransferWorkLoopCrcTest, ExpiredStagingHeadLetsSmallerBatchUseExistingFreeCapacity) {
    auto options              = diskOptions();
    options.read_thread_count = 2;
    engine                    = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group}, true, DeviceHostCopyOptions{}, 4, 2, 1, 8, nullptr, 0, 1, true, options);
    block_tree_cache_detail::ScopeRollback finish([&] {
        io->release();
        engine->shutdown();
    });
    seedDisk(0);
    seedDisk(3);
    io->blocked = true;
    auto active = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[0], {targets[0]})}));
    ASSERT_TRUE(io->waitReads());  // One of two staging blocks is occupied.
    auto task      = makeTransferTask({TransferDescriptor::diskToDevice(0, disks[1], {targets[1]}),
                                       TransferDescriptor::diskToDevice(0, disks[2], {targets[2]})});
    task.deadline_ = TransferTask::Clock::now() + 50ms;
    auto expired   = engine->execute(std::move(task));
    auto smaller   = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[3], {targets[3]})}));
    ASSERT_TRUE(await(expired));
    EXPECT_EQ(expired->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    ASSERT_TRUE(io->waitReads(2));  // The smaller waiter uses the remaining block before A releases its lease.
    EXPECT_FALSE(active->done());
    EXPECT_FALSE(smaller->done());
    finish.run();
    EXPECT_TRUE(active->success());
    EXPECT_TRUE(smaller->success());
    EXPECT_EQ(io->reads, 2u);
    expectTarget(0);
    expectTarget(3);
}

TEST_F(TransferWorkLoopCrcTest, DeadlineSkipsUnsubmittedCrcAndReleasesResources) {
    initialize();
    ASSERT_TRUE(await(engine->execute(evict(0))));
    ASSERT_TRUE(await(engine->execute(evict(1))));
    seedDisk(1);
    engine->shutdown();
    for (bool from_disk : {false, true}) {
        SCOPED_TRACE(from_disk);
        for (bool expire_while_waiting : {true, false}) {
            SCOPED_TRACE(expire_while_waiting);
            std::shared_ptr<HostStagingBlockPool> pool;
            if (from_disk)
                pool = std::make_shared<HostStagingBlockPool>(1, disk->strideBytes());
            TransferWorkLoop                       owner({group},
                                   std::make_unique<CrcCopyExecutor>(std::vector<GroupSetPtr>{group}, 1, 1),
                                   direct_disk_executor.get(),
                                   TransferDirection::LOAD,
                                   1,
                                   8,
                                   std::move(pool));
            block_tree_cache_detail::ScopeRollback finish([&] {
                owner.start();
                owner.shutdown();
            });
            const std::vector<uint8_t>             guard(384, 0xa5);
            ASSERT_EQ(cudaMemcpy(buffer(targets[1]), guard.data(), guard.size(), cudaMemcpyHostToDevice), cudaSuccess);
            auto active =
                owner.execute(makeTransferTask({TransferDescriptor::hostToDevice(0, hosts[0], {targets[0]})}));
            auto task      = makeTransferTask({from_disk ? TransferDescriptor::diskToDevice(0, disks[1], {targets[1]}) :
                                                           TransferDescriptor::hostToDevice(0, hosts[1], {targets[1]})});
            task.deadline_ = TransferTask::Clock::now() + 1h;
            auto       expired = owner.execute(std::move(task));
            const auto reads   = io->reads;
            // Keep A's CRC slot occupied until the owner consumes its result.
            owner.entries_.splice(owner.entries_.end(), owner.incoming_);
            for (auto position = owner.entries_.begin(); position != owner.entries_.end(); ++position) {
                auto& entry    = **position;
                entry.position = position;
                owner.tryStep(entry);
            }
            if (from_disk) {
                ASSERT_TRUE(io->waitReads(reads + 1));
                std::unique_lock<std::mutex> lock(owner.mutex_);
                ASSERT_TRUE(owner.cv_.wait_for(lock, 5s, [&] { return !owner.disk_completed_.empty(); }));
                auto completed = owner.disk_completed_.take();
                lock.unlock();
                auto* entry = completed.pop();
                ASSERT_EQ(entry, owner.entries_.back().get());
                ASSERT_TRUE(completed.empty());
                if (!expire_while_waiting)
                    entry->task.deadline_ = TransferTask::Clock::now() - 1s;
                // Consume only Disk completion, keeping A's CRC slot occupied.
                owner.consumeStage(*entry);
            }
            if (!from_disk || expire_while_waiting) {
                ASSERT_FALSE(expired->done());
                ASSERT_NE(owner.entries_.back()->waiting_on, nullptr);
                if (expire_while_waiting) {
                    auto& waiter          = *owner.entries_.back();
                    waiter.task.deadline_ = TransferTask::Clock::now() - 1s;
                    owner.wait_deadlines_.erase(*waiter.wait_deadline);
                    waiter.wait_deadline = owner.wait_deadlines_.emplace(waiter.task.deadline(), &waiter);
                    owner.expireResourceWaiters();
                    EXPECT_TRUE(expired->done());
                    EXPECT_FALSE(active->done());
                } else {
                    // A completion wakes B before the next deadline sweep. Retrying must
                    // recheck its deadline even though initial planning already ran.
                    owner.entries_.back()->task.deadline_ = TransferTask::Clock::now() - 1s;
                }
            }
            finish.run();
            EXPECT_TRUE(active->success());
            EXPECT_EQ(expired->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
            EXPECT_EQ(expired->result().blockError(0), CacheCopyError::NOT_EXECUTED);
            std::vector<uint8_t> actual(384);
            ASSERT_EQ(cudaMemcpy(actual.data(), buffer(targets[1]), actual.size(), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(actual, guard);
            EXPECT_TRUE(owner.wait_deadlines_.empty());
            if (from_disk) {
                EXPECT_EQ(owner.staging_.pool->free_id_list_.size(), 1u);
            }
        }
    }
}

TEST_F(TransferWorkLoopCrcTest, CombinedRoutesKeepOnlyBatchStageIntervals) {
    initialize(8, 2, 1, 2);
    for (size_t i = 0; i < 2; ++i)
        seedDisk(i);
    auto session = AsyncProfileSession::start(100000);
    for (size_t i = 0; i < 2; ++i) {
        auto context = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[i], {targets[i]})}));
        ASSERT_TRUE(await(context));
        EXPECT_TRUE(context->success());
        expectTarget(i);
    }
    engine->shutdown();
    const auto events = session->finish();
    ASSERT_EQ(events.size(), 12u);
    checkWorkLoopTrace(events);
    std::map<std::string, size_t> stages;
    for (const auto& event : events)
        ++stages[event.record.stage];
    EXPECT_EQ(stages,
              (std::map<std::string, size_t>{{"worker_task_total", 2},
                                             {"worker_batch_total", 2},
                                             {"staging_wait", 2},
                                             {"transfer_queue_wait", 2},
                                             {"h2d", 2},
                                             {"disk_read", 2}}));
}

TEST_F(TransferWorkLoopCrcTest, SlowWriteBackpressureKeepsOtherStagesAndDirectionProgressing) {
    enableMetrics();
    initialize(8, 8, 1);
    for (size_t i = 0; i < 2; ++i) {
        auto sealed = engine->execute(evict(i));
        ASSERT_TRUE(await(sealed));
        ASSERT_TRUE(sealed->success());
    }
    io->blocked = true;
    auto first  = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[0], disks[0])}));
    ASSERT_TRUE(io->waitWrites());
    auto waiting    = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[1], disks[1])}));
    auto host_evict = engine->execute(evict(2));
    ASSERT_TRUE(await(host_evict));
    EXPECT_TRUE(host_evict->success());
    auto load = engine->execute(makeTransferTask({TransferDescriptor::hostToDevice(0, hosts[2], {targets[2]})}));
    ASSERT_TRUE(await(load));
    EXPECT_TRUE(load->success());
    expectTarget(2);
    EXPECT_FALSE(first->done());
    EXPECT_FALSE(waiting->done());
    engine->stopAdmission();
    io->release();
    engine->shutdown();
    EXPECT_TRUE(first->success());
    EXPECT_TRUE(waiting->success());
    EXPECT_EQ(io->writes, 2u);
    reporter->flushStageMetrics();
    checkStage("d2h", 3, true, Tier::DEVICE, Tier::HOST);
    checkStage("disk_write", 2, true, Tier::HOST, Tier::DISK);
}

TEST_F(TransferWorkLoopCrcTest, ExpiredStagingWaiterIsSkippedWithoutBlockingTheNextTask) {
    for (size_t i = 0; i < 3; ++i)
        seedDisk(i);
    for (const auto direction : {TransferDirection::LOAD}) {
        enableMetrics();
        SCOPED_TRACE(direction == TransferDirection::EVICT ? "evict" : "load");
        size_t              callbacks = 0;
        std::atomic<size_t> submissions{0};
        auto                crc = std::make_unique<CrcCopyExecutor>(std::vector<GroupSetPtr>{group}, 2, 1);
        for (auto& [ordinal, slots] : crc->slots_)
            for (auto& slot : slots)
                slot.copy->setTestHook([&](const char* point, void*, void*) {
                    if (std::strcmp(point, "metadata") == 0)
                        ++submissions;
                    return int(cudaSuccess);
                });
        auto             pool = std::make_shared<HostStagingBlockPool>(1, disk->strideBytes());
        TransferWorkLoop owner(
            {group}, std::move(crc), direct_disk_executor.get(), direction, 1, 3, std::move(pool), reporter, 3);
        block_tree_cache_detail::ScopeRollback finish([&] {
            if (!owner.thread_.joinable())
                owner.start();
            io->release();
            owner.shutdown();
        });
        const std::vector<uint8_t>             guard(384, 0xa5);
        ASSERT_EQ(cudaMemcpy(buffer(targets[1]), guard.data(), guard.size(), cudaMemcpyHostToDevice), cudaSuccess);
        const auto reads = io->reads, writes = io->writes;
        io->blocked     = true;
        const auto task = [&](size_t i) {
            auto result      = makeTransferTask({TransferDescriptor::diskToDevice(0, disks[i], {targets[i]})});
            result.deadline_ = TransferTask::Clock::now() + 1h;
            return result;
        };
        auto active  = owner.execute(task(0));
        auto expired = owner.execute(task(1));
        auto next    = owner.execute(task(2));
        owner.entries_.splice(owner.entries_.end(), owner.incoming_);
        for (auto position = owner.entries_.begin(); position != owner.entries_.end();) {
            auto current         = position++;
            (*current)->position = current;
            owner.tryStep(**current);
        }
        ASSERT_FALSE(active->done());
        ASSERT_FALSE(expired->done());
        ASSERT_FALSE(next->done());
        auto  position = owner.entries_.begin();
        auto& first    = **position++;
        auto& waiter   = **position++;
        ASSERT_EQ(first.staging.size(), 1u);
        ASSERT_EQ(owner.staging_.waiting.head, &waiter);
        ASSERT_EQ(owner.staging_.waiting.tail, position->get());
        // Expire the waiter while the active I/O keeps its staging lease.
        waiter.task.deadline_ = TransferTask::Clock::now() - 1s;
        owner.wait_deadlines_.erase(*waiter.wait_deadline);
        waiter.wait_deadline = owner.wait_deadlines_.emplace(waiter.task.deadline(), &waiter);
        expired->onDone([&](TransferResult result) {
            EXPECT_EQ(result.status.code(), ErrorCode::DEADLINE_EXCEEDED);
            EXPECT_EQ(result.blockError(0), CacheCopyError::NOT_EXECUTED);
            ++callbacks;
        });
        owner.start();
        if (direction == TransferDirection::EVICT)
            ASSERT_TRUE(io->waitWrites(writes + 1));
        else
            ASSERT_TRUE(io->waitReads(reads + 1));
        ASSERT_TRUE(await(expired));
        EXPECT_FALSE(active->done());
        EXPECT_FALSE(next->done());
        finish.run();
        EXPECT_TRUE(active->success());
        EXPECT_EQ(expired->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_TRUE(next->success());
        EXPECT_EQ(callbacks, 1u);
        EXPECT_EQ(submissions.load(), 2u);
        EXPECT_EQ(io->writes, writes + (direction == TransferDirection::EVICT ? 2u : 0u));
        EXPECT_EQ(io->reads, reads + (direction == TransferDirection::LOAD ? 2u : 0u));
        EXPECT_EQ(owner.outstandingBatches(), 0u);
        EXPECT_EQ(owner.entry_count_, 0u);
        EXPECT_TRUE(owner.wait_deadlines_.empty());
        EXPECT_TRUE(owner.staging_.waiting.empty());
        EXPECT_EQ(owner.staging_.pool->free_id_list_.size(), 1u);
        reporter->flushStageMetrics();
        EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 9u);
        const auto source = direction == TransferDirection::LOAD ? Tier::DISK : Tier::DEVICE;
        const auto target = direction == TransferDirection::LOAD ? Tier::DEVICE : Tier::DISK;
        checkStage("transfer_queue_wait", 3, true, source, target, "worker_task");
        checkStage("worker_task_total", 2, true, source, target, "worker_task");
        checkStage("worker_task_total", 1, false, source, target, "worker_task");
        checkStage("worker_batch_total", 2, true, source, target);
        checkStage("worker_batch_total", 1, false, source, target);
        checkStage("staging_wait", 2, true, source, target);
        checkStage("staging_wait", 1, false, source, target);
        checkStage(direction == TransferDirection::LOAD ? "disk_read" : "disk_write", 2, true, source, target);
        checkStage(direction == TransferDirection::LOAD ? "h2d" : "d2h", 2, true, source, target);
        checkQueueWait(3, source, target);
        if (direction == TransferDirection::LOAD) {
            expectTarget(0);
            expectTarget(2);
            std::vector<uint8_t> actual(guard.size());
            ASSERT_EQ(cudaMemcpy(actual.data(), buffer(targets[1]), actual.size(), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(actual, guard);
        }
    }
}

TEST_F(TransferWorkLoopCrcTest, RequestStagingTimeoutPreservesCompletedBatchResults) {
    initialize(8, 8, 1);
    auto&               owner  = *engine->load_loops_.front();
    bool                expire = true;
    std::atomic<size_t> submissions{0};
    for (auto& [ordinal, slots] : owner.crc_->slots_)
        for (auto& slot : slots)
            slot.copy->setTestHook([&](const char* point, void*, void*) {
                if (std::strcmp(point, "metadata") == 0)
                    ++submissions;
                if (std::strcmp(point, "query") == 0 && std::exchange(expire, false)) {
                    for (auto& entry : owner.entries_)
                        entry->task.deadline_ = TransferTask::Clock::now() - 1s;
                    std::lock_guard<std::mutex> lock(owner.mutex_);
                    for (auto& entry : owner.incoming_)
                        entry->task.deadline_ = TransferTask::Clock::now() - 1s;
                }
                return int(cudaSuccess);
            });
    std::vector<TransferDescriptor> descriptors;
    for (size_t i = 0; i < 3; ++i) {
        seedDisk(i);
        descriptors.push_back(TransferDescriptor::diskToDevice(0, disks[i], {targets[i]}));
    }
    io->blocked  = true;
    auto context = executeRequest(makeTransferTask(descriptors));
    ASSERT_TRUE(io->waitReads());
    io->release();
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{
                  CacheCopyError::NONE, CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED}));
    EXPECT_EQ(submissions.load(), 1u);
    EXPECT_EQ(io->reads, 1u);
    EXPECT_EQ(io->writes, 0u);
    EXPECT_EQ(owner.entry_count_, 0u);
    EXPECT_EQ(owner.staging_.pool->free_id_list_.size(), 1u);
    expectTarget(0);
}

TEST_F(TransferWorkLoopCrcTest, ClosedAdmissionStillSubmitsCrcSuccessorsAndSmallStagingBatches) {
    enableMetrics();
    initialize(8, 8, 1);
    std::vector<TransferDescriptor> descriptors;
    for (size_t i = 0; i < 5; ++i) {
        seedDisk(i);
        descriptors.push_back(TransferDescriptor::diskToDevice(0, disks[i], {targets[i]}));
    }
    io->blocked  = true;
    auto context = executeRequest(makeTransferTask(descriptors));
    ASSERT_TRUE(io->waitReads());
    std::promise<TransferResult> completed;
    auto                         future = completed.get_future();
    context->onDone([&](TransferResult result) { completed.set_value(std::move(result)); });
    context.reset();
    engine->stopAdmission();
    io->release();
    engine->shutdown();
    ASSERT_EQ(future.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(future.get().ok());
    EXPECT_EQ(io->reads, 5u);
    for (size_t i = 0; i < 5; ++i)
        expectTarget(i);
    reporter->flushStageMetrics();
    EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 6u);
    checkStage("transfer_queue_wait", 5, true, Tier::DISK, Tier::DEVICE, "worker_task");
    checkStage("worker_task_total", 5, true, Tier::DISK, Tier::DEVICE, "worker_task");
    for (const char* stage : {"worker_batch_total", "staging_wait", "disk_read", "h2d"})
        checkStage(stage, 5, true, Tier::DISK, Tier::DEVICE);
    checkQueueWait(5, Tier::DISK, Tier::DEVICE);
}

TEST_F(TransferWorkLoopCrcTest, BurstWithOneSlotCompletesEveryWaitingEntryExactlyOnce) {
    enableMetrics();
    initialize(16, 2, 1);
    auto session = AsyncProfileSession::start(100000);
    io->blocked  = true;
    auto first   = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[0], disks[0])}));
    ASSERT_TRUE(io->waitWrites());
    std::promise<void>                     entered, release;
    auto                                   entered_future = entered.get_future();
    auto                                   released       = release.get_future();
    block_tree_cache_detail::ScopeRollback unblock([&] { release.set_value(); });
    first->onDone([&](TransferResult) {
        entered.set_value();
        EXPECT_EQ(released.wait_for(5s), std::future_status::ready);
    });
    io->release();
    ASSERT_EQ(entered_future.wait_for(5s), std::future_status::ready);
    std::vector<std::shared_ptr<TransferAsyncContext>> contexts;
    std::atomic<size_t>                                calls{0};
    for (size_t i = 0; i < 8; ++i) {
        contexts.push_back(engine->execute(evict(i)));
        contexts.back()->onDone([&](TransferResult result) {
            EXPECT_TRUE(result.ok());
            ++calls;
        });
    }
    std::promise<void>                     query_entered, release_query;
    auto                                   query_future   = query_entered.get_future();
    auto                                   query_released = release_query.get_future();
    bool                                   first_query    = true;
    block_tree_cache_detail::ScopeRollback unblock_query([&] { release_query.set_value(); });
    for (auto& [ordinal, slots] : engine->evict_loops_.front()->crc_->slots_)
        for (auto& slot : slots)
            slot.copy->setTestHook([&](const char* point, void*, void*) {
                if (std::strcmp(point, "query") == 0 && std::exchange(first_query, false)) {
                    query_entered.set_value();
                    EXPECT_EQ(query_released.wait_for(5s), std::future_status::ready);
                }
                return int(cudaSuccess);
            });
    unblock.run();
    ASSERT_EQ(query_future.wait_for(5s), std::future_status::ready);
    // One accepted CRC and seven Busy waiters now belong to the owner.
    engine->stopAdmission();
    unblock_query.run();
    for (const auto& context : contexts)
        ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(calls, 8u);
    reporter->flushStageMetrics();
    checkStage("d2h", 8, true, Tier::DEVICE, Tier::HOST);
    checkStage("worker_task_total", 8, true, Tier::DEVICE, Tier::HOST, "worker_task");
    checkStage("transfer_queue_wait", 8, true, Tier::DEVICE, Tier::HOST, "worker_task");
    checkStage("disk_write", 1, true, Tier::HOST, Tier::DISK);
    checkQueueWait(8, Tier::DEVICE, Tier::HOST);
    checkQueueWait(1, Tier::HOST, Tier::DISK);
    const auto events = session->finish();
    checkWorkLoopTrace(events);
    EXPECT_EQ(events.size(), 27u);  // Busy retries add no intervals: nine direct tasks.
}

TEST_F(TransferWorkLoopCrcTest, ClosingWithQueuedDiskWritesDrainsBothWrites) {
    engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group}, true, DeviceHostCopyOptions{}, 4, 8, 1, 8, nullptr, 0, 1, true, diskOptions());
    for (size_t i = 0; i < 2; ++i) {
        auto sealed = engine->execute(evict(i));
        ASSERT_TRUE(await(sealed));
        ASSERT_TRUE(sealed->success());
    }
    io->blocked = true;
    auto first  = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[0], disks[0])}));
    ASSERT_TRUE(io->waitWrites());
    auto waiting        = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[1], disks[1])}));
    auto pending_writes = std::make_shared<size_t>(0);
    auto marker         = engine->evict_loops_.front()->execute(
        evict(2), [loop = engine->evict_loops_.front().get(), pending_writes](const TransferResult&) {
            // The before_complete callback runs on the owner before publishing the marker context.
            for (const auto& entry : loop->entries_)
                if (entry->step == TransferWorkLoop::Step::DISK_WRITE)
                    ++*pending_writes;
        });
    ASSERT_TRUE(await(marker));
    EXPECT_TRUE(marker->success());  // B's CRC retired; its write is queued in DiskBlockIO.
    EXPECT_EQ(*pending_writes, 2u);  // Both writes were submitted despite CRC C=1.
    EXPECT_FALSE(waiting->done());
    engine->stopAdmission();
    io->release();
    engine->shutdown();
    EXPECT_TRUE(first->success());
    EXPECT_TRUE(waiting->success());
    EXPECT_EQ(io->writes, 2u);
    EXPECT_EQ(io->data[disk->blockOffset(disks[1])], 18u);
}

TEST_F(TransferWorkLoopCrcTest, DirectIoRoundTripPreservesNonzeroPaddingAndCrc) {
    retained_controlled_disk = disk;
    for (auto block : disks)
        releasePoolBlock(*disk, block);
    disks.clear();
    disk = makeDiskPool(384, 16, temp.path, nullptr, "workloop_direct", false, true);
    for (size_t i = 0; i < 8; ++i)
        disks.push_back(poolMalloc(*disk));
    group = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device}, host, disk, true);
    for (const bool async : {false, true}) {
        SCOPED_TRACE(async ? "workloop" : "legacy");
        initialize(8, 2, 1, 1, 2, async);
        auto sealed = engine->execute(evict(0));
        ASSERT_TRUE(await(sealed));
        ASSERT_TRUE(sealed->success());
        auto* source = static_cast<uint8_t*>(host->blockBuffer(hosts[0]).addr);
        std::memset(source + group->storageBytes(), 0xa5, disk->strideBytes() - group->storageBytes());
        auto evict = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[0], disks[0])}));
        ASSERT_TRUE(await(evict));
        ASSERT_TRUE(evict->success());
        auto load = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[0], {targets[0]})}));
        ASSERT_TRUE(await(load));
        EXPECT_TRUE(load->success());
        engine->shutdown();
        expectTarget(0);
        HostStagingBlockPool scratch(1, disk->strideBytes());
        auto                 leases = scratch.tryMallocBatch(1);
        ASSERT_TRUE(leases);
        const auto buffer = leases->front().blockBuffer(384);
        ASSERT_EQ(disk->read(disks[0], buffer.base, disk->strideBytes()), BlockIOStatus::OK);
        for (size_t i = group->storageBytes(); i < disk->strideBytes(); ++i) {
            EXPECT_EQ(static_cast<uint8_t*>(buffer.base)[i], 0xa5u);
            EXPECT_EQ(source[i], 0xa5u);
        }
    }
}

TEST_F(TransferWorkLoopCrcTest, CrcErrorsDoNotIssueDiskWrites) {
    for (const char* phase : {"metadata", "query"}) {
        SCOPED_TRACE(phase);
        auto session = AsyncProfileSession::start(100000);
        enableMetrics();
        initialize();
        auto& owner = *engine->evict_loops_.front();
        for (auto& [ordinal, slots] : owner.crc_->slots_)
            for (auto& slot : slots)
                slot.copy->setTestHook([phase](const char* point, void*, void*) {
                    return int(std::strcmp(point, phase) == 0 ? cudaErrorInvalidValue : cudaSuccess);
                });
        auto context = engine->execute(evict(0));
        ASSERT_TRUE(await(context));
        engine->shutdown();
        EXPECT_EQ(context->result().copy_error, CacheCopyError::COPY_FAILED);
        EXPECT_EQ(io->writes, 0u);
        EXPECT_TRUE(device->isAllocated(sources[0]));
        reporter->flushStageMetrics();
        checkStage("d2h", 1, false, Tier::DEVICE, Tier::HOST);
        checkStage("worker_task_total", 1, false, Tier::DEVICE, Tier::HOST, "worker_task");
        checkStage("transfer_queue_wait", 1, true, Tier::DEVICE, Tier::HOST, "worker_task");
        checkWorkLoopTrace(session->finish());
    }
}

TEST_F(TransferWorkLoopCrcTest, SmallStagingRequestCompletesHealthyBatchesAfterCorruption) {
    enableMetrics();
    initialize(8, 8, 1);
    std::vector<TransferDescriptor> descriptors;
    for (size_t i = 0; i < 3; ++i) {
        seedDisk(i);
        descriptors.push_back(TransferDescriptor::diskToDevice(0, disks[i], {targets[i]}));
    }
    io->data[disk->blockOffset(disks[1])] ^= 1;
    auto context = executeRequest(makeTransferTask(descriptors));
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NONE, CacheCopyError::CRC_MISMATCH, CacheCopyError::NONE}));
    EXPECT_EQ(io->reads, 3u);
    expectTarget(0);
    expectTarget(2);
    reporter->flushStageMetrics();
    EXPECT_EQ(seriesCount(metrics()->transfer_stage_qps_metric), 9u);
    checkStage("worker_batch_total", 2, true, Tier::DISK, Tier::DEVICE);
    checkStage("worker_batch_total", 1, false, Tier::DISK, Tier::DEVICE);
    checkStage("staging_wait", 3, true, Tier::DISK, Tier::DEVICE);
    checkStage("disk_read", 3, true, Tier::DISK, Tier::DEVICE);
    checkStage("h2d", 2, true, Tier::DISK, Tier::DEVICE);
    checkStage("h2d", 1, false, Tier::DISK, Tier::DEVICE);
    checkStage("worker_task_total", 2, true, Tier::DISK, Tier::DEVICE, "worker_task");
    checkStage("worker_task_total", 1, false, Tier::DISK, Tier::DEVICE, "worker_task");
}

TEST_F(TransferWorkLoopCrcTest, DiskWriteFailureAfterSuccessfulCrcEviction) {
    enableMetrics();
    initialize();
    auto sealed = engine->execute(evict(0));
    ASSERT_TRUE(await(sealed));
    ASSERT_TRUE(sealed->success());
    io->fail_write = 1;
    auto context   = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(0, hosts[0], disks[0])}));
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->result().copy_error, CacheCopyError::IO_FAILED);
    EXPECT_EQ(context->result().block_errors, (std::vector<CacheCopyError>{CacheCopyError::IO_FAILED}));
    EXPECT_EQ(io->writes, 1u);
    reporter->flushStageMetrics();
    checkStage("d2h", 1, true, Tier::DEVICE, Tier::HOST);
    checkStage("disk_write", 1, false, Tier::HOST, Tier::DISK);
    checkStage("worker_task_total", 1, false, Tier::HOST, Tier::DISK, "worker_task");
}

TEST_F(TransferWorkLoopCrcTest, HostDiskBatchesPreservePaddingAndLoadBackToDevice) {
    for (const bool async : {false, true}) {
        SCOPED_TRACE(async);
        enableMetrics();
        initialize(8, 3, 1, 1, 12, async);
        ASSERT_EQ(engine->batchLimit(Tier::DEVICE, Tier::HOST), 3u);
        const auto writes_before = io->writes;
        const auto reads_before  = io->reads;
        std::fill(io->data.begin(), io->data.end(), 0xa5);
        std::vector<TransferDescriptor> evictions, disk_evictions, loads;
        for (size_t i = sources.size(); i-- > 0;) {
            auto* host_data = static_cast<uint8_t*>(host->blockBuffer(hosts[i]).addr);
            std::memset(host_data + group->storageBytes(), 0xa5, disk->strideBytes() - group->storageBytes());
            evictions.push_back(TransferDescriptor::deviceToHost(0, {sources[i]}, hosts[i]));
            disk_evictions.push_back(TransferDescriptor::hostToDisk(0, hosts[i], disks[i]));
            loads.push_back(TransferDescriptor::diskToDevice(0, disks[i], {targets[i]}));
        }
        auto evicted = executeRequest(makeTransferTask(evictions));
        ASSERT_TRUE(await(evicted));
        ASSERT_TRUE(evicted->success());
        EXPECT_EQ(evicted->result().block_errors, std::vector<CacheCopyError>(8, CacheCopyError::NONE));
        auto disk_evicted = executeRequest(makeTransferTask(disk_evictions));
        ASSERT_TRUE(await(disk_evicted));
        ASSERT_TRUE(disk_evicted->success());
        auto loaded = executeRequest(makeTransferTask(loads));
        ASSERT_TRUE(await(loaded));
        ASSERT_TRUE(loaded->success());
        engine->shutdown();
        EXPECT_EQ(loaded->result().block_errors, std::vector<CacheCopyError>(8, CacheCopyError::NONE));
        EXPECT_EQ(io->writes, writes_before + 8);
        EXPECT_EQ(io->reads, reads_before + 8);
        for (size_t i = 0; i < sources.size(); ++i) {
            expectTarget(i);
            for (size_t byte = group->storageBytes(); byte < disk->strideBytes(); ++byte)
                ASSERT_EQ(io->data[disk->blockOffset(disks[i]) + byte], 0xa5u);
        }
        reporter->flushStageMetrics();
        for (const char* stage : {"worker_batch_total", "staging_wait", "disk_read", "h2d"})
            checkStage(stage, 3, true, Tier::DISK, Tier::DEVICE);
    }
}

TEST_F(TransferWorkLoopCrcTest, DiskDeviceBatchRetainsAllStagingUntilReadCompletes) {
    for (const bool async : {false, true}) {
        SCOPED_TRACE(async);
        initialize(8, 2, 1, 2, 4, async);
        for (size_t i = 0; i < 3; ++i)
            seedDisk(i);
        auto marker_data = engine->execute(evict(3));
        ASSERT_TRUE(await(marker_data));
        ASSERT_TRUE(marker_data->success());
        const auto reads_before = io->reads;
        io->blocked             = true;
        auto loaded = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[0], {targets[0]}),
                                                        TransferDescriptor::diskToDevice(0, disks[1], {targets[1]})}));
        ASSERT_TRUE(io->waitReads(reads_before + 1));
        auto waiting = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[2], {targets[2]})}));
        auto marker  = engine->execute(makeTransferTask({TransferDescriptor::hostToDevice(0, hosts[3], {targets[3]})}));
        ASSERT_TRUE(await(marker));
        ASSERT_TRUE(marker->success());
        auto* pool = async ? engine->load_loops_.front()->staging_.pool.get() :
                             engine->legacy_->device_disk_executor_->full_staging_pool_.get();
        EXPECT_FALSE(pool->tryMallocBatch(1));
        EXPECT_FALSE(loaded->done());
        EXPECT_FALSE(waiting->done());
        io->release();
        ASSERT_TRUE(await(loaded));
        ASSERT_TRUE(await(waiting));
        engine->shutdown();
        EXPECT_TRUE(loaded->success());
        EXPECT_TRUE(waiting->success());
        EXPECT_EQ(io->reads, reads_before + 3);
        auto leases = pool->tryMallocBatch(2);
        ASSERT_TRUE(leases);
        EXPECT_EQ(leases->size(), 2u);
    }
}

TEST_F(TransferWorkLoopCrcTest, PartialHostDiskBatchWriteFailsEveryDescriptorAndCanRetry) {
    for (const bool async : {false, true}) {
        SCOPED_TRACE(async);
        initialize(8, 3, 1, 1, 6, async);
        const auto                      writes_before = io->writes;
        std::vector<TransferDescriptor> evictions, loads;
        for (size_t i = 0; i < 3; ++i) {
            auto sealed = engine->execute(evict(i));
            ASSERT_TRUE(await(sealed));
            ASSERT_TRUE(sealed->success());
            evictions.push_back(TransferDescriptor::hostToDisk(0, hosts[i], disks[i]));
            loads.push_back(TransferDescriptor::diskToDevice(0, disks[i], {targets[i]}));
        }
        io->fail_write = writes_before + 2;
        auto failed    = engine->execute(makeTransferTask(evictions));
        ASSERT_TRUE(await(failed));
        EXPECT_FALSE(failed->success());
        EXPECT_EQ(failed->result().block_errors, std::vector<CacheCopyError>(3, CacheCopyError::IO_FAILED));
        engine->waitForIdle();
        EXPECT_EQ(io->writes, writes_before + 2);
        io->fail_write = 0;
        auto evicted   = engine->execute(makeTransferTask(evictions));
        ASSERT_TRUE(await(evicted));
        ASSERT_TRUE(evicted->success());
        auto loaded = engine->execute(makeTransferTask(loads));
        ASSERT_TRUE(await(loaded));
        ASSERT_TRUE(loaded->success());
        engine->shutdown();
        EXPECT_EQ(io->writes, writes_before + 5);
        for (size_t i = 0; i < 3; ++i)
            expectTarget(i);
    }
}

TEST_F(TransferWorkLoopCrcTest, EightDescriptorStagingBatchKeepsOneResultPerDescriptorOnCrcFailure) {
    engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                          true,
                                                          DeviceHostCopyOptions{},
                                                          32,
                                                          8,
                                                          1,
                                                          8,
                                                          nullptr,
                                                          0,
                                                          1,
                                                          true,
                                                          diskOptions());
    std::vector<TransferDescriptor> descriptors;
    for (size_t i = 0; i < 8; ++i) {
        seedDisk(i);
        descriptors.push_back(TransferDescriptor::diskToDevice(0, disks[i], {targets[i]}));
    }
    io->data[disk->blockOffset(disks[5])] ^= 1;
    auto context = engine->execute(makeTransferTask(descriptors));
    ASSERT_TRUE(await(context));
    engine->shutdown();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->result().copy_error, CacheCopyError::CRC_MISMATCH);
    ASSERT_EQ(context->result().block_errors.size(), 8u);
    for (size_t i = 0; i < 8; ++i)
        EXPECT_EQ(context->result().block_errors[i],
                  i == 5 ? CacheCopyError::CRC_MISMATCH : CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(io->reads, 8u);
}

TEST_F(TransferWorkLoopCrcTest, ClosingDuringDiagnosticsWaitsForPhysicalSafetyAndOwnerCompletion) {
    initialize();
    auto sealed = engine->execute(evict(0));
    ASSERT_TRUE(await(sealed));
    static_cast<uint8_t*>(host->blockBuffer(hosts[0]).addr)[0] ^= 1;
    std::promise<void>                     entered, release;
    auto                                   entered_future = entered.get_future();
    auto                                   released       = release.get_future();
    block_tree_cache_detail::ScopeRollback unblock([&] { release.set_value(); });
    for (auto& [ordinal, slots] : engine->load_loops_.front()->crc_->slots_)
        for (auto& slot : slots)
            slot.copy->setTestHook([&](const char* point, void*, void*) {
                if (std::strcmp(point, "snapshot") == 0) {
                    entered.set_value();
                    EXPECT_EQ(released.wait_for(5s), std::future_status::ready);
                }
                return int(cudaSuccess);
            });
    auto context = engine->execute(makeTransferTask({TransferDescriptor::hostToDevice(0, hosts[0], {targets[0]})}));
    ASSERT_EQ(entered_future.wait_for(5s), std::future_status::ready);
    // Diagnostics hold only the load owner; the independent evict service
    // can still complete healthy work before engine shutdown joins both owners.
    auto healthy = engine->execute(evict(1));
    ASSERT_TRUE(await(healthy));
    EXPECT_TRUE(healthy->success());
    auto drain = std::async(std::launch::async, [&] { engine->shutdown(); });
    EXPECT_EQ(drain.wait_for(20ms), std::future_status::timeout);
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(host->isAllocated(hosts[0]));
    EXPECT_TRUE(device->isAllocated(targets[0]));
    unblock.run();
    ASSERT_EQ(drain.wait_for(5s), std::future_status::ready);
    drain.get();
    EXPECT_EQ(context->result().copy_error, CacheCopyError::CRC_MISMATCH);
}

TEST_F(TransferWorkLoopTest, IndependentFullAndSwaDiskPoolsShareOneExecutorAcrossOwners) {
    auto swa_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size = 128;
    auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 384),
                                      makeTestGroupBase(swa_policy, {1}, 384)});
    group->initialize(0, topology, {"group0"}, makeTestBackingLayout(384, false));
    auto       swa_disk  = makeDiskPool(384, 16, temp.path, std::make_unique<ControlledDiskIO>(), "swa");
    auto       swa       = makeTestGroupSet(1, topology, {"group1"}, {}, host, swa_disk);
    const auto swa_block = poolMalloc(*swa_disk);
    for (bool async : {false, true}) {
        engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, swa},
                                                              true,
                                                              DeviceHostCopyOptions{},
                                                              128,
                                                              2,
                                                              2,
                                                              8,
                                                              nullptr,
                                                              0,
                                                              1,
                                                              async,
                                                              diskOptions());
        ASSERT_NE(engine->diskIOExecutor(), nullptr);
        EXPECT_EQ(engine->diskIOExecutor()->config().read_thread_count, diskOptions().read_thread_count);
        EXPECT_EQ(engine->diskIOExecutor()->config().write_thread_count, diskOptions().write_thread_count);
        for (const auto& owners : {&engine->load_loops_, &engine->evict_loops_})
            for (const auto& owner : *owners)
                EXPECT_EQ(owner->disk_io_executor_, engine->diskIOExecutor());
        auto full_context = engine->execute(writes());
        auto swa_context  = engine->execute(makeTransferTask({TransferDescriptor::hostToDisk(1, hosts[1], swa_block)}));
        ASSERT_TRUE(await(full_context));
        ASSERT_TRUE(await(swa_context));
        EXPECT_TRUE(full_context->success());
        EXPECT_TRUE(swa_context->success());
        engine->shutdown();
    }
    releasePoolBlock(*swa_disk, swa_block);
}

TEST_F(TransferWorkLoopCrcTest, FullAndSwaShareStagingWhileDirectionsKeepIndependentCapacity) {
    auto swa_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size = 128;
    auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 384),
                                      makeTestGroupBase(swa_policy, {1}, 384)});
    group->initialize(0, topology, {"group0"}, makeTestBackingLayout(384, true));
    auto swa = makeTestGroupSet(1, topology, {"group1"}, {device}, host, disk, true);
    engine   = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, swa},
                                                          true,
                                                          DeviceHostCopyOptions{},
                                                          4,
                                                          8,
                                                          1,
                                                          8,
                                                          nullptr,
                                                          0,
                                                          1,
                                                          true,
                                                          diskOptions());
    block_tree_cache_detail::ScopeRollback finish([&] {
        io->release();
        engine->shutdown();
    });
    auto&                                  load_staging    = engine->load_loops_.front()->staging_;
    auto&                                  evict_staging   = engine->evict_loops_.front()->staging_;
    ASSERT_EQ(load_staging.pool->capacity(), 2u);
    ASSERT_EQ(evict_staging.pool->capacity(), 2u);
    ASSERT_TRUE(await(engine->execute(evict(4))));
    seedDisk(0);
    seedDisk(1);
    seedDisk(2);
    io->blocked    = true;
    auto full_load = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[0], {targets[0]}),
                                                       TransferDescriptor::diskToDevice(0, disks[1], {targets[1]})}));
    ASSERT_TRUE(io->waitReads());
    auto swa_load = engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(1, disks[2], {targets[2]})}));
    // This later task proves the load owner has processed the SWA staging wait.
    auto marker = engine->execute(makeTransferTask({TransferDescriptor::hostToDevice(0, hosts[4], {targets[4]})}));
    ASSERT_TRUE(await(marker));
    ASSERT_TRUE(marker->success());
    ASSERT_NE(load_staging.waiting.head, nullptr);
    EXPECT_EQ(load_staging.waiting.head->task.descriptors().front().group_set_id, 1u);
    EXPECT_FALSE(load_staging.pool->tryMallocBatch(1));
    auto evict_leases = evict_staging.pool->tryMallocBatch(2);
    ASSERT_TRUE(evict_leases);
    EXPECT_FALSE(swa_load->done());
    evict_leases.reset();
    finish.run();
    EXPECT_TRUE(full_load->success());
    EXPECT_TRUE(swa_load->success());
    EXPECT_EQ(io->reads, 3u);
    expectTarget(0);
    expectTarget(1);
    expectTarget(2);
}

TEST_F(TransferWorkLoopCrcTest, DirectionPollScansAllDevicesWithIndependentSlots) {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count < 2)
        GTEST_SKIP() << "requires two locked GPUs";
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    auto other_device = makeTestDevicePool({{384, 0}}, 4, "workloop_second_device");
    auto other_host   = makeHostPool(384, 4, true);
    auto other       = makeTestGroupSet(1, group->topologyPtr(), {"group0"}, {other_device}, other_host, nullptr, true);
    auto source      = poolMalloc(*other_device);
    auto destination = poolMalloc(*other_host);
    const std::vector<uint8_t> expected(384, 0x65);
    ASSERT_EQ(cudaMemcpy(other_device->convertIndexToBuffer(0, source).front().addr,
                         expected.data(),
                         expected.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, other},
                                                          true,
                                                          DeviceHostCopyOptions{},
                                                          4,
                                                          2,
                                                          1,
                                                          8,
                                                          nullptr,
                                                          0,
                                                          1,
                                                          true,
                                                          diskOptions());
    auto first  = engine->execute(evict(0));
    auto second = engine->execute(makeTransferTask({TransferDescriptor::deviceToHost(1, {source}, destination)}));
    ASSERT_TRUE(await(first));
    ASSERT_TRUE(await(second));
    EXPECT_TRUE(first->success());
    EXPECT_TRUE(second->success());
    engine->shutdown();
    EXPECT_EQ(std::memcmp(other_host->blockBuffer(destination).addr, expected.data(), expected.size()), 0);
    releasePoolBlock(*other_device, source);
    releasePoolBlock(*other_host, destination);
}

TEST_F(TransferWorkLoopCrcTest, StagingBatchLimitDoesNotShrinkDeviceHostBatches) {
    engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group}, true, DeviceHostCopyOptions{}, 6, 8, 1, 8, nullptr, 0, 1, true, diskOptions());
    ASSERT_EQ(engine->load_loops_.front()->staging_.pool->capacity(), 3u);
    ASSERT_EQ(engine->evict_loops_.front()->staging_.pool->capacity(), 3u);
    EXPECT_EQ(engine->load_loops_.front()->batchLimit(), 8u);
    EXPECT_EQ(engine->evict_loops_.front()->batchLimit(), 8u);
    EXPECT_EQ(engine->batchLimit(Tier::DISK, Tier::DEVICE), 3u);
    EXPECT_EQ(engine->batchLimit(Tier::DEVICE, Tier::HOST), 8u);

    auto evictions = makeTransferTask({TransferDescriptor::deviceToHost(0, {sources[0]}, hosts[0]),
                                       TransferDescriptor::deviceToHost(0, {sources[1]}, hosts[1]),
                                       TransferDescriptor::deviceToHost(0, {sources[2]}, hosts[2]),
                                       TransferDescriptor::deviceToHost(0, {sources[3]}, hosts[3])});
    auto evicted   = engine->execute(std::move(evictions));
    ASSERT_TRUE(await(evicted));
    ASSERT_TRUE(evicted->success());
    auto loaded = executeRequest(makeTransferTask({TransferDescriptor::hostToDevice(0, hosts[0], {targets[0]}),
                                                   TransferDescriptor::hostToDevice(0, hosts[1], {targets[1]}),
                                                   TransferDescriptor::hostToDevice(0, hosts[2], {targets[2]}),
                                                   TransferDescriptor::hostToDevice(0, hosts[3], {targets[3]})}));
    ASSERT_TRUE(await(loaded));
    ASSERT_TRUE(loaded->success());
    engine->shutdown();
    expectTarget(0);
    expectTarget(1);
    expectTarget(2);
    expectTarget(3);
}

TEST_F(TransferWorkLoopCrcTest, SharedStagingUsesLargestDiskStrideAcrossGroups) {
    auto swa_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size = 128;
    auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 384),
                                      makeTestGroupBase(swa_policy, {1}, 384)});
    group->initialize(0, topology, {"group0"}, makeTestBackingLayout(384, true));
    auto config             = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = "swa_larger_stride";
    config->work_dir        = temp.path;
    config->payload_bytes   = 384;
    config->stride_bytes    = 8192;
    config->disk_size_bytes = 8192 * 17;
    auto swa_disk           = std::make_shared<DiskBlockPool>(config, std::make_unique<ControlledDiskIO>());
    ASSERT_TRUE(swa_disk->init());
    auto swa        = makeTestGroupSet(1, topology, {"group1"}, {device}, host, swa_disk, true);
    auto single_swa = makeTestGroupSet(
        0, makeTestTopology({makeTestGroupBase(swa_policy, {0}, 384)}), {"group0"}, {device}, host, swa_disk, true);
    for (const auto& groups : {std::vector<GroupSetPtr>{group, swa}, std::vector<GroupSetPtr>{single_swa}}) {
        SCOPED_TRACE(groups.size());
        PerRankBlockTransferEngine bounded(groups, true, {}, 12, 8, 1, 8, nullptr, 0, 1, true);
        ASSERT_TRUE(bounded.asyncEnabled());
        for (const auto* direction : {&bounded.load_loops_, &bounded.evict_loops_}) {
            const auto& pool = direction->front()->staging_.pool;
            ASSERT_NE(pool, nullptr);
            EXPECT_EQ(pool->capacity(), 6u);
            EXPECT_EQ(pool->stride_bytes_, 8192u);
            EXPECT_EQ(direction->front()->batchLimit(), 8u);
        }
        EXPECT_EQ(bounded.batchLimit(Tier::DEVICE, Tier::HOST), 8u);
        EXPECT_EQ(bounded.batchLimit(Tier::DISK, Tier::DEVICE), 6u);
    }
}

TEST_F(TransferWorkLoopCrcTest, DefaultStagingBudgetKeepsConfiguredBatchLimit) {
    PerRankBlockTransferEngine defaults({group}, true, {}, 128, 8, 1, 8, nullptr, 0, 1, true);
    EXPECT_EQ(defaults.load_loops_.front()->batchLimit(), 8u);
    EXPECT_EQ(defaults.evict_loops_.front()->batchLimit(), 8u);
    EXPECT_EQ(defaults.batchLimit(Tier::DISK, Tier::DEVICE), 8u);
}

TEST_F(TransferWorkLoopCrcTest, LoopPairsOwnIndependentCountBasedStagingAndCrcSlots) {
    auto swa_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size = 128;
    auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 384),
                                      makeTestGroupBase(swa_policy, {1}, 384)});
    group->initialize(0, topology, {"group0"}, makeTestBackingLayout(384, true));
    auto config             = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = "multi_loop_swa_stride";
    config->work_dir        = temp.path;
    config->payload_bytes   = 384;
    config->stride_bytes    = 8192;
    config->disk_size_bytes = 8192 * 17;
    auto swa_disk           = std::make_shared<DiskBlockPool>(config, std::make_unique<ControlledDiskIO>());
    ASSERT_TRUE(swa_disk->init());
    auto swa = makeTestGroupSet(1, topology, {"group1"}, {device}, host, swa_disk, true);
    engine   = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, swa},
                                                          true,
                                                          DeviceHostCopyOptions{},
                                                          8,
                                                          8,
                                                          2,
                                                          8,
                                                          nullptr,
                                                          3,
                                                          2,
                                                          true,
                                                          diskOptions());
    std::set<const HostStagingBlockPool*>  pools;
    std::set<const CrcCopyWorkspace*>      copies;
    std::array<std::atomic<size_t>, 4>     submissions{};
    block_tree_cache_detail::ScopeRollback cleanup([&] { engine->shutdown(); });
    size_t                                 owner_index   = 0;
    size_t                                 staging_bytes = 0;
    for (const auto* direction : {&engine->load_loops_, &engine->evict_loops_}) {
        ASSERT_EQ(direction->size(), 2u);
        for (size_t worker = 0; worker < direction->size(); ++worker) {
            const auto& loop = (*direction)[worker];
            EXPECT_EQ(loop->batchLimit(), 8u);
            const auto& pool = loop->staging_.pool;
            ASSERT_NE(pool, nullptr);
            EXPECT_EQ(pool->capacity(), 4u);
            EXPECT_EQ(pool->stride_bytes_, 8192u);
            staging_bytes += pool->capacity() * pool->stride_bytes_;
            EXPECT_TRUE(pools.insert(pool.get()).second);
            ASSERT_NE(loop->crc_, nullptr);
            const auto expected =
                (direction == &engine->load_loops_ ? "btc_load_" : "btc_evict_") + std::to_string(worker);
            for (auto& [device_index, slots] : loop->crc_->slots_) {
                EXPECT_EQ(slots.size(), 2u);
                for (auto& slot : slots) {
                    EXPECT_TRUE(copies.insert(slot.copy.get()).second);
                    slot.copy->setTestHook([&, index = owner_index, expected](const char* point, void*, void*) {
                        if (std::strcmp(point, "metadata") == 0) {
                            ++submissions[index];
                            char name[16]{};
                            EXPECT_EQ(pthread_getname_np(pthread_self(), name, sizeof(name)), 0);
                            EXPECT_EQ(std::string(name), expected);
                        }
                        return int(cudaSuccess);
                    });
                }
            }
            ++owner_index;
        }
    }
    EXPECT_EQ(pools.size(), 4u);
    EXPECT_EQ(copies.size(), 8u);  // Two directions * two owners * two slots.
    EXPECT_EQ(staging_bytes, 2u * 8u * 8192u);
    std::vector<std::shared_ptr<TransferAsyncContext>> contexts;
    for (size_t i = 0; i < 2; ++i)
        contexts.push_back(engine->execute(evict(i)));
    for (const auto& context : contexts) {
        ASSERT_TRUE(await(context));
        ASSERT_TRUE(context->success());
    }
    contexts.clear();
    for (size_t i = 0; i < 2; ++i) {
        seedDisk(i);
        contexts.push_back(
            engine->execute(makeTransferTask({TransferDescriptor::diskToDevice(0, disks[i], {targets[i]})})));
    }
    for (const auto& context : contexts) {
        ASSERT_TRUE(await(context));
        ASSERT_TRUE(context->success());
    }
    cleanup.run();
    for (const auto& count : submissions)
        EXPECT_EQ(count.load(), 1u);
    expectTarget(0);
    expectTarget(1);
}

TEST_F(TransferWorkLoopCrcTest, LargerPhysicalDiskStridePreservesConfiguredStagingBlockCounts) {
    auto config             = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = "larger_stride";
    config->work_dir        = temp.path;
    config->payload_bytes   = 384;
    config->stride_bytes    = 8192;
    config->disk_size_bytes = 8192 * 17;
    auto wider_disk         = std::make_shared<DiskBlockPool>(config, std::make_unique<ControlledDiskIO>());
    ASSERT_TRUE(wider_disk->init());
    auto wider_group = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device}, host, wider_disk, true);
    PerRankBlockTransferEngine minimum({wider_group}, true, {}, 2, 8, 1, 8, nullptr, 0, 1, true);
    EXPECT_EQ(minimum.load_loops_.front()->staging_.pool->capacity(), 1u);
    EXPECT_EQ(minimum.evict_loops_.front()->staging_.pool->capacity(), 1u);
    PerRankBlockTransferEngine bounded({wider_group}, true, {}, 8, 8, 1, 8, nullptr, 0, 1, true);
    EXPECT_EQ(bounded.load_loops_.front()->staging_.pool->capacity(), 4u);
    EXPECT_EQ(bounded.evict_loops_.front()->staging_.pool->capacity(), 4u);
    EXPECT_EQ(bounded.load_loops_.front()->staging_.pool->stride_bytes_, 8192u);
}

TEST_F(TransferWorkLoopCrcTest, InitializationRejectsInvalidStagingAndAcceptsSelectedLegacy) {
    for (const size_t blocks : {0u, 1u, 3u})
        EXPECT_THROW((PerRankBlockTransferEngine({group}, true, {}, blocks, 8, 1, 8, nullptr, 0, 1, true)),
                     std::invalid_argument);
    auto plain = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device}, host, disk, false);
    PerRankBlockTransferEngine fallback({plain}, true, {}, 4, 8, 1, 0, nullptr, 0, 1, false);
    EXPECT_FALSE(fallback.asyncEnabled());
}

}  // namespace
}  // namespace rtp_llm
