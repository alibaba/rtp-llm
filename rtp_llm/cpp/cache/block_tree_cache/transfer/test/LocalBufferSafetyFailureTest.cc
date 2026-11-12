#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>
#include <new>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/TransferExecutor.h"

// Fail one allocation on the calling thread, then allow error handling to proceed.
// A separate test binary keeps this interposition out of unrelated GPU tests.
namespace {
thread_local int allocation_countdown = -1;
std::atomic<size_t> injected_allocations{0};
}
[[gnu::noinline]] void* operator new(std::size_t size) {
    if (allocation_countdown >= 0 && allocation_countdown-- == 0) {
        ++injected_allocations;
        throw std::bad_alloc();
    }
    if (void* pointer = std::malloc(size ? size : 1)) return pointer;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* operator new[](std::size_t size) { return ::operator new(size); }
[[gnu::noinline]] void operator delete(void* pointer) noexcept { std::free(pointer); }
[[gnu::noinline]] void operator delete[](void* pointer) noexcept { std::free(pointer); }
[[gnu::noinline]] void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
[[gnu::noinline]] void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace rtp_llm {
namespace {
using State = TransferTaskRegistry::State;
using namespace std::chrono_literals;

void registerTask(TransferTaskRegistry& registry, TransferTask& task) {
    const auto trace = std::to_string(task.traceId());
    ASSERT_TRUE(registry.begin(trace));
    task.setCompletionProof(std::make_shared<TransferBufferSafety>([&registry, trace] {
        registry.finish(trace, true);
    }));
}
State state(const TransferTaskRegistry& registry, uint64_t trace) {
    return registry.queryLocal(std::vector<uint64_t>{trace}).front();
}

class SafeExecutor final: public TransferExecutor {
public:
    explicit SafeExecutor(BlockTreeTaskPool& pool): TransferExecutor(pool, 1, nullptr) {}
    bool fail_diagnostics{false};
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        TransferResult result(fail_diagnostics ? TransferStatus::DISK_IO_ERROR : TransferStatus::OK);
        result.buffer_safe = true;
        if (fail_diagnostics) allocation_countdown = 0;
        return result;
    }
};

TEST(LocalBufferSafetyFailureTest, SafeFailureSurvivesDiagnosticAllocationFailure) {
    BlockTreeTaskPool pool(1, 8, "safe_diagnostic");
    ASSERT_TRUE(pool.start());
    SafeExecutor executor(pool);
    executor.fail_diagnostics = true;
    TransferTaskRegistry registry;
    TransferTask task({TransferDescriptor::hostToDisk(0, 1, 1)}, 30s);
    const auto trace = task.traceId();
    registerTask(registry, task);
    const auto before = injected_allocations.load();
    auto context = executor.execute(std::move(task), {HostBufferView{}}, {nullptr});
    pool.waitForIdle();
    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->result().buffer_safe);
    EXPECT_GT(injected_allocations.load(), before);
    EXPECT_EQ(state(registry, trace), State::BUFFER_SAFE);
}

TEST(LocalBufferSafetyFailureTest, LegacySubmissionAllocationFailuresNeverStrandProof) {
    BlockTreeTaskPool pool(1, 8, "safe_submit");
    ASSERT_TRUE(pool.start());
    SafeExecutor executor(pool);
    size_t failures = 0;
    for (int position = 0; position < 80; ++position) {
        SCOPED_TRACE(position);
        TransferTaskRegistry registry;
        TransferTask task({TransferDescriptor::hostToDisk(0, 1, 1)}, 30s);
        if (position >= 40) task.deadline_ = TransferTask::Clock::now() - 1ms;
        const auto trace = task.traceId();
        registerTask(registry, task);
        std::vector<HostBufferView> hosts(1);
        std::vector<const GroupSet*> groups(1, nullptr);
        std::shared_ptr<TransferAsyncContext> context;
        allocation_countdown = position % 40;
        try {
            context = executor.execute(std::move(task), std::move(hosts), std::move(groups));
        } catch (const std::bad_alloc&) { ++failures; }
        allocation_countdown = -1;
        pool.waitForIdle();
        EXPECT_EQ(state(registry, trace), State::BUFFER_SAFE);
        if (context) { EXPECT_TRUE(context->done()); }
    }
    EXPECT_GT(failures, 0u);
}

TEST(LocalBufferSafetyFailureTest, ExpiredDispatcherAllocationFailuresCloseRegisteredExecution) {
    auto engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{}, false, DeviceHostCopyOptions{}, 128, 1, 1, 8);
    BlockTransferDispatcher dispatcher(engine);
    for (int position = 0; position < 40; ++position) {
        SCOPED_TRACE(position);
        TransferTask task({TransferDescriptor::hostToDisk(0, 1, 1)}, 30s);
        task.deadline_ = TransferTask::Clock::now() - 1ms;
        const auto trace = task.traceId();
        allocation_countdown = position;
        try { dispatcher.executePerRank(std::move(task)); } catch (const std::bad_alloc&) {}
        allocation_countdown = -1;
        // Failure before registration is UNKNOWN; admitted, unsubmitted work must finish.
        EXPECT_NE(state(dispatcher.taskRegistry(), trace), State::RUNNING);
    }
}

TEST(LocalBufferSafetyFailureTest, RealEnginePreparationFailurePublishesUnsubmittedProof) {
    PerRankBlockTransferEngine engine({}, false, {}, 128, 1, 1, 8);
    TransferTaskRegistry registry;
    TransferTask task({TransferDescriptor::hostToDisk(0, 1, 1)}, 30s);
    const auto trace = task.traceId();
    registerTask(registry, task);
    bool threw = false;
    allocation_countdown = 0;  // group_sets.reserve(), before any producer is entered.
    try { engine.execute(std::move(task)); } catch (const std::bad_alloc&) { threw = true; }
    allocation_countdown = -1;
    EXPECT_TRUE(threw);
    EXPECT_EQ(state(registry, trace), State::BUFFER_SAFE);
}

template<typename Base>
class SafeStage final: public Base {
public:
    explicit SafeStage(BlockTreeTaskPool& pool): Base(pool, 1) {}
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        return TransferResult::emptyAggregate();
    }
};

GroupSetPtr stagingGroup() {
    auto spec = std::make_shared<OpaqueKVCacheSpec>("allocation_safety");
    spec->setLayout(4096, 1, 4096, 4096);
    GroupBase group;
    group.tag = spec->tag;
    group.spec = spec;
    group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology = CacheTopology::create({group}, {{0, {group.tag}}});
    auto result = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    result->initialize(0, std::move(topology), {group.tag});
    return result;
}

TEST(LocalBufferSafetyFailureTest, PartialDeviceDiskSubmissionStillJoinsAllAcceptedBatches) {
    BlockTreeTaskPool pool(1, 8, "safe_partial");
    ASSERT_TRUE(pool.start());
    SafeStage<DeviceHostTransferExecutor> device(pool);
    SafeStage<HostDiskTransferExecutor> disk(pool);
    auto group = stagingGroup();
    DeviceDiskTransferExecutor executor(device, disk, {group}, 8, pool, 1, nullptr);
    const auto before = injected_allocations.load();
    for (int position = 0; position < 100; ++position) {
        SCOPED_TRACE(position);
        TransferTaskRegistry registry;
        TransferTask task({TransferDescriptor::diskToDevice(0, 1, {1}),
                           TransferDescriptor::diskToDevice(0, 2, {2}),
                           TransferDescriptor::diskToDevice(0, 3, {3})}, 30s);
        const auto trace = task.traceId();
        registerTask(registry, task);
        std::vector<const GroupSet*> groups(3, group.get());
        std::shared_ptr<TransferAsyncContext> context;
        allocation_countdown = position;
        try { context = executor.execute(std::move(task), groups); } catch (const std::bad_alloc&) {}
        allocation_countdown = -1;
        pool.waitForIdle();
        EXPECT_EQ(state(registry, trace), State::BUFFER_SAFE);
        if (context) { EXPECT_TRUE(context->done()); }
    }
    EXPECT_GT(injected_allocations.load(), before);
}

TEST(LocalBufferSafetyFailureTest, CancelledStagingStillFinishesWhenErrorFormattingThrows) {
    BlockTreeTaskPool pool(1, 8, "safe_cancel");
    SafeStage<DeviceHostTransferExecutor> device(pool);
    SafeStage<HostDiskTransferExecutor> disk(pool);
    auto group = stagingGroup();
    DeviceDiskTransferExecutor executor(device, disk, {group}, 2, pool, 1, nullptr);
    auto held = executor.full_staging_pool_->tryMallocBatch(1);
    ASSERT_TRUE(held);
    TransferTaskRegistry registry;
    TransferTask task({TransferDescriptor::diskToDevice(0, 1, {1})}, 30s);
    const auto trace = task.traceId();
    registerTask(registry, task);
    auto context = executor.execute(std::move(task), {group.get()});
    ASSERT_FALSE(context->done());
    auto& waiter = executor.full_staging_pool_->batch_waiters_.front();
    auto ready = std::move(waiter.callback);
    waiter.callback = [ready = std::move(ready)](auto leases) mutable {
        allocation_countdown = 0;
        ready(std::move(leases));
        allocation_countdown = -1;
    };
    const auto before = injected_allocations.load();
    executor.cancelPendingTransfers();
    EXPECT_GT(injected_allocations.load(), before);
    EXPECT_TRUE(context->done());
    EXPECT_EQ(state(registry, trace), State::BUFFER_SAFE);
}

TEST(LocalBufferSafetyFailureTest, CrcTerminalProofSurvivesResultAllocationFailure) {
    const std::vector<TransferDescriptor> descriptors{TransferDescriptor::hostToDevice(0, 1, {1})};
    for (auto terminal : {CrcCopyState::OK, CrcCopyState::CRC_MISMATCH, CrcCopyState::DEVICE_ERROR}) {
        TransferTaskRegistry registry;
        TransferTask task(descriptors, 30s);
        const auto trace = task.traceId();
        registerTask(registry, task);
        TransferBatchAsyncContext context(1, nullptr, task.completionProof());
        const auto before = injected_allocations.load();
        allocation_countdown = 0;
        auto result = CrcCopyExecutor::mapResult(terminal, CrcCopyDirection::LOAD, descriptors, {});
        allocation_countdown = -1;
        EXPECT_GT(injected_allocations.load(), before);
        EXPECT_TRUE(result.buffer_safe);
        EXPECT_EQ(result.ok(), terminal == CrcCopyState::OK);
        context.complete(std::move(result));
        EXPECT_EQ(state(registry, trace), State::BUFFER_SAFE);
    }
}

TEST(LocalBufferSafetyFailureTest, DiagnosticFallbackPreservesSafetyAndUnknownErrorPriority) {
    for (bool safe : {false, true}) {
        allocation_countdown = 0;
        auto fallback = TransferResult::executionException(safe, "diagnostic allocation is deliberately unavailable");
        allocation_countdown = -1;
        EXPECT_EQ(fallback.buffer_safe, safe);
        EXPECT_EQ(fallback.status.code(), ErrorCode::EXECUTION_EXCEPTION);

        TransferResult first(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "original unknown completion"));
        first.buffer_safe = safe;
        TransferResult second(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "additional diagnostic failure"));
        second.buffer_safe = true;
        allocation_countdown = 0;
        first.merge(std::move(second));
        allocation_countdown = -1;
        EXPECT_EQ(first.buffer_safe, safe);
        EXPECT_EQ(first.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    }
}
}  // namespace
}  // namespace rtp_llm
