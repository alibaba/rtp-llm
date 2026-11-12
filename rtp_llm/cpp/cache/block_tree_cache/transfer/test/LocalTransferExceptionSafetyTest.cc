#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include <atomic>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/TransferExecutor.h"

// Fail one allocation on the calling thread, then allow error handling to proceed.
// A separate test binary keeps this interposition out of unrelated GPU tests.
namespace {
thread_local int    allocation_countdown = -1;
std::atomic<size_t> injected_allocations{0};
}  // namespace
[[gnu::noinline]] void* operator new(std::size_t size) {
    if (allocation_countdown >= 0 && allocation_countdown-- == 0) {
        ++injected_allocations;
        throw std::bad_alloc();
    }
    if (void* pointer = std::malloc(size ? size : 1))
        return pointer;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* operator new[](std::size_t size) {
    return ::operator new(size);
}
[[gnu::noinline]] void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;
std::atomic<size_t> executed_stages{0};

class SafeExecutor final: public TransferExecutor {
public:
    explicit SafeExecutor(BlockTreeTaskPool& pool): TransferExecutor(pool, 1, nullptr) {}
    bool           fail_diagnostics{false};
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        TransferResult result(fail_diagnostics ? TransferStatus::DISK_IO_ERROR : TransferStatus::OK);
        if (fail_diagnostics)
            allocation_countdown = 0;
        return result;
    }
};

TEST(LocalBufferSafetyFailureTest, SafeFailureSurvivesDiagnosticAllocationFailure) {
    BlockTreeTaskPool pool(1, 8, "safe_diagnostic");
    ASSERT_TRUE(pool.start());
    SafeExecutor executor(pool);
    executor.fail_diagnostics = true;
    TransferTask task({TransferDescriptor::hostToDisk(0, 1, 1)}, 30s);
    const auto   before  = injected_allocations.load();
    auto         context = executor.execute(std::move(task), {HostBufferView{}}, {nullptr});
    pool.waitForIdle();
    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_GT(injected_allocations.load(), before);
}

TEST(LocalBufferSafetyFailureTest, LegacySubmissionAllocationFailuresNeverStrandContext) {
    BlockTreeTaskPool pool(1, 8, "safe_submit");
    ASSERT_TRUE(pool.start());
    SafeExecutor executor(pool);
    size_t       failures = 0;
    for (int position = 0; position < 80; ++position) {
        SCOPED_TRACE(position);
        TransferTask task({TransferDescriptor::hostToDisk(0, 1, 1)}, 30s);
        if (position >= 40)
            task.deadline_ = TransferTask::Clock::now() - 1ms;
        std::vector<HostBufferView>           hosts(1);
        std::vector<const GroupSet*>          groups(1, nullptr);
        std::shared_ptr<TransferAsyncContext> context;
        allocation_countdown = position % 40;
        try {
            context = executor.execute(std::move(task), std::move(hosts), std::move(groups));
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        allocation_countdown = -1;
        pool.waitForIdle();
        if (context) {
            EXPECT_TRUE(context->done());
        }
    }
    EXPECT_GT(failures, 0u);
}

template<typename Base>
class SafeStage final: public Base {
public:
    explicit SafeStage(BlockTreeTaskPool& pool): Base(pool, 1) {}
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        ++executed_stages;
        return TransferResult{};
    }
};

GroupSetPtr stagingGroup() {
    auto spec = std::make_shared<OpaqueKVCacheSpec>("allocation_safety");
    spec->setLayout(4096, 1, 4096, 4096);
    GroupBase group;
    group.tag     = spec->tag;
    group.spec    = spec;
    group.policy  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology = CacheTopology::create({group}, {{0, {group.tag}}});
    auto result   = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    result->initialize(0, std::move(topology), {group.tag});
    return result;
}

TEST(LocalBufferSafetyFailureTest, PartialDeviceDiskSubmissionStillJoinsAllAcceptedBatches) {
    BlockTreeTaskPool pool(1, 8, "safe_partial");
    ASSERT_TRUE(pool.start());
    SafeStage<DeviceHostTransferExecutor> device(pool);
    SafeStage<HostDiskTransferExecutor>   disk(pool);
    auto                                  group = stagingGroup();
    DeviceDiskTransferExecutor            executor(device, disk, {group}, 8, pool, 1, nullptr);
    const auto                            before = injected_allocations.load();
    for (int position = 0; position < 100; ++position) {
        SCOPED_TRACE(position);
        TransferTask                          task({TransferDescriptor::diskToDevice(0, 1, {1}),
                                                    TransferDescriptor::diskToDevice(0, 2, {2}),
                                                    TransferDescriptor::diskToDevice(0, 3, {3})},
                          30s);
        std::vector<const GroupSet*>          groups(3, group.get());
        std::shared_ptr<TransferAsyncContext> context;
        executed_stages      = 0;
        allocation_countdown = position;
        try {
            context = executor.execute(std::move(task), groups);
        } catch (const std::bad_alloc&) {}
        allocation_countdown = -1;
        pool.waitForIdle();
        if (context) {
            EXPECT_TRUE(context->done());
        } else {
            EXPECT_EQ(executed_stages.load(), 0u) << "I/O was accepted before the context escaped by exception";
        }
    }
    EXPECT_GT(injected_allocations.load(), before);
}

TEST(LocalBufferSafetyFailureTest, CancelledStagingStillFinishesWhenErrorFormattingThrows) {
    BlockTreeTaskPool                     pool(1, 8, "safe_cancel");
    SafeStage<DeviceHostTransferExecutor> device(pool);
    SafeStage<HostDiskTransferExecutor>   disk(pool);
    auto                                  group = stagingGroup();
    DeviceDiskTransferExecutor            executor(device, disk, {group}, 2, pool, 1, nullptr);
    auto                                  held = executor.full_staging_pool_->tryMallocBatch(1);
    ASSERT_TRUE(held);
    TransferTask task({TransferDescriptor::diskToDevice(0, 1, {1})}, 30s);
    auto         context = executor.execute(std::move(task), {group.get()});
    ASSERT_FALSE(context->done());
    auto& waiter    = executor.full_staging_pool_->batch_waiters_.front();
    auto  ready     = std::move(waiter.callback);
    waiter.callback = [ready = std::move(ready)](auto leases) mutable {
        allocation_countdown = 0;
        ready(std::move(leases));
        allocation_countdown = -1;
    };
    const auto before = injected_allocations.load();
    executor.cancelPendingTransfers();
    EXPECT_GT(injected_allocations.load(), before);
    EXPECT_TRUE(context->done());
}

TEST(LocalBufferSafetyFailureTest, CrcTerminalResultSurvivesDiagnosticAllocationFailure) {
    const std::vector<TransferDescriptor> descriptors{TransferDescriptor::hostToDevice(0, 1, {1})};
    for (auto terminal : {CrcCopyState::OK, CrcCopyState::CRC_MISMATCH, CrcCopyState::DEVICE_ERROR}) {
        TransferBatchAsyncContext context(1);
        const auto                before = injected_allocations.load();
        allocation_countdown             = 0;
        auto result          = CrcCopyExecutor::mapResult(terminal, CrcCopyDirection::LOAD, descriptors, {});
        allocation_countdown = -1;
        EXPECT_GT(injected_allocations.load(), before);
        EXPECT_EQ(result.ok(), terminal == CrcCopyState::OK);
        context.complete(std::move(result));
    }
}

TEST(LocalBufferSafetyFailureTest, MissingFenceAllocationFailuresKeepAdmissionAndRetentionConsistent) {
    using Registry                = TransferTaskRegistry;
    using State                   = Registry::State;
    size_t     failures           = 0;
    size_t     successful_queries = 0;
    const auto before             = injected_allocations.load();
    for (int position = 0; position < 96; ++position) {
        SCOPED_TRACE(position);
        auto                           now = Registry::Clock::time_point{};
        Registry                       registry([&] { return now; });
        const std::vector<std::string> traces{"allocation_fence_first_with_nontrivial_key",
                                              "allocation_fence_second_with_nontrivial_key",
                                              "allocation_fence_third_with_nontrivial_key"};
        allocation_countdown = position;
        try {
            const auto response  = registry.queryAndBlockMissing(traces);
            allocation_countdown = -1;
            EXPECT_EQ(response, (std::vector<State>(traces.size(), State::REJECTED)));
            ++successful_queries;
        } catch (const std::bad_alloc&) {
            allocation_countdown = -1;
            ++failures;
        }
        const auto retained = registry.queryLocal(traces);
        for (size_t index = 0; index < traces.size(); ++index) {
            if (retained[index] == State::REJECTED) {
                // A committed fence must own valid metadata and reject/consume
                // exactly the late arrival, even after a later trace failed allocation.
                EXPECT_FALSE(registry.begin(traces[index]));
                EXPECT_EQ(registry.queryLocal(std::vector<std::string>{traces[index]}).front(), State::UNKNOWN);
            } else {
                EXPECT_EQ(retained[index], State::UNKNOWN);
                EXPECT_TRUE(registry.begin(traces[index]));
                EXPECT_TRUE(registry.finish(traces[index], true));
                EXPECT_EQ(registry.forget(traces[index]), Registry::ForgetResult::DONE);
            }
        }
        EXPECT_EQ(registry.size(), 0u);
        now += std::chrono::hours(2);
        // No orphan list node may survive either rollback or late-arrival consume.
        EXPECT_EQ(registry.cleanupExpiredMissing(128), 0u);
        EXPECT_EQ(registry.size(), 0u);
    }
    EXPECT_GT(failures, 0u);
    EXPECT_GT(successful_queries, 0u);
    EXPECT_GT(injected_allocations.load(), before);
}

TEST(LocalBufferSafetyFailureTest, DiagnosticFallbackPreservesUnknownErrorPriority) {
    allocation_countdown = 0;
    auto fallback = TransferResult::executionException(false, "diagnostic allocation is deliberately unavailable");
    allocation_countdown = -1;
    EXPECT_EQ(fallback.status.code(), ErrorCode::EXECUTION_EXCEPTION);

    TransferResult first(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "original unknown completion"));
    TransferResult second(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "additional diagnostic failure"));
    allocation_countdown = 0;
    first.merge(std::move(second));
    allocation_countdown = -1;
    EXPECT_EQ(first.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
}

TEST(LocalBufferSafetyFailureTest, StageFinalPublicationMovesWithoutDiagnosticAllocation) {
    size_t             callbacks = 0;
    ErrorCode          code      = ErrorCode::NONE_ERROR;
    TransferStageState stage(
        [&](TransferResult result) {
            ++callbacks;
            code = result.status.code();
        },
        1);
    stage.addBatch();
    stage.completeBatch(TransferResult(ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                                 "the completed stage has a nontrivial allocated diagnostic")),
                        {0});
    bool threw           = false;
    allocation_countdown = 0;
    try {
        stage.finishSubmitting();
    } catch (...) {
        threw = true;
    }
    allocation_countdown = -1;
    EXPECT_FALSE(threw);
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(code, ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
}

class LateCompletionContext final: public TransferAsyncContext {
public:
    explicit LateCompletionContext(bool fail_subscription): fail_subscription_(fail_subscription) {}
    void waitDone() override {
        result_.waitDone();
    }
    bool done() const override {
        return result_.done();
    }
    bool success() const override {
        return result_.success();
    }
    TransferResult result() const override {
        return result_.result();
    }
    void onDone(DoneCallback callback) override {
        result_.onDone(std::move(callback));
        if (fail_subscription_) {
            throw std::runtime_error("callback registered but subscription failed after submission");
        }
    }
    void complete() {
        result_.complete(TransferStatus::OK);
    }

private:
    bool                      fail_subscription_;
    TransferBatchAsyncContext result_{1};
};

class SubscriptionFailureEngine final: public PerRankBlockTransferEngine {
public:
    SubscriptionFailureEngine(): PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}) {}
    std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask) override {
        auto child = std::make_shared<LateCompletionContext>(children.empty());
        children.push_back(child);
        return child;
    }
    std::vector<std::shared_ptr<LateCompletionContext>> children;
};

TEST(LocalBufferSafetyFailureTest, SubmittedChildSubscriptionFailurePreservesUnknownAndCannotCompleteTwice) {
    SubscriptionFailureEngine engine;
    auto                      context = engine.execute(
        TransferTask({TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::hostToDisk(1, 2, 2)}, 30s));
    ASSERT_EQ(engine.children.size(), 2u);
    size_t callbacks = 0;
    context->onDone([&](TransferResult) { ++callbacks; });
    EXPECT_FALSE(context->done());
    // This first child already settled through the exception path. Its real
    // completion must not consume the still-pending second child's counter.
    engine.children[0]->complete();
    EXPECT_FALSE(context->done());
    EXPECT_EQ(callbacks, 0u);
    engine.children[1]->complete();
    ASSERT_TRUE(context->done());
    EXPECT_EQ(context->result().status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    EXPECT_EQ(callbacks, 1u);
}
TEST(LocalBufferSafetyFailureTest, ExpansionAllocationFailureDeliversUniformUnknown) {
    TransferBatchAsyncContext context(3);
    ErrorCode                 delivered = ErrorCode::NONE_ERROR;
    size_t                    calls     = 0;
    context.onDone([&](TransferResult result) {
        ++calls;
        delivered = result.status.code();
        for (size_t index = 0; index < 3; ++index)
            EXPECT_EQ(result.blockError(index), CacheCopyError::RPC_FAILED);
    });
    allocation_countdown = 0;
    context.complete(TransferStatus::OK);
    allocation_countdown = -1;
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(delivered, ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.result().block_errors.empty());
}

TEST(LocalBufferSafetyFailureTest, CallbackCopyAllocationFailureStillDeliversUnknown) {
    TransferBatchAsyncContext context(2);
    ErrorCode                 delivered = ErrorCode::NONE_ERROR;
    size_t                    calls     = 0;
    context.onDone([&](TransferResult result) {
        ++calls;
        delivered = result.status.code();
    });
    TransferResult original(
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "this terminal result requires an allocated diagnostic when copied"));
    original.expandBlockErrors(2);
    allocation_countdown = 0;
    context.complete(std::move(original));
    allocation_countdown = -1;
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(delivered, ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    EXPECT_TRUE(context.done());
}

TEST(LocalBufferSafetyFailureTest, LateCallbackAndResultCopyAllocationFailureRemainConservative) {
    TransferBatchAsyncContext context(2);
    context.complete(TransferResult(
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "this terminal result needs allocation to copy its diagnostic")));
    ErrorCode                          delivered = ErrorCode::NONE_ERROR;
    TransferAsyncContext::DoneCallback callback  = [&](TransferResult result) { delivered = result.status.code(); };
    allocation_countdown                         = 0;
    context.onDone(std::move(callback));
    allocation_countdown = -1;
    EXPECT_EQ(delivered, ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    allocation_countdown = 0;
    auto result          = context.result();
    allocation_countdown = -1;
    EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
}

TEST(LocalBufferSafetyFailureTest, UniformUnknownKeepsTargetCustodyUntilExplicitRelease) {
    auto config                            = std::make_shared<HostBlockPoolConfig>();
    config->pool_type                      = BlockPoolType::HOST;
    config->pool_name                      = "unknown_completion_delivery";
    config->physical_block_count           = 4;
    config->frozen_block_abort_lower_ratio = 0.0;
    config->payload_bytes                  = 64;
    config->stride_bytes                   = 4096;
    config->alignment                      = 4096;
    auto pool                              = std::make_shared<HostBlockPool>(config);
    ASSERT_TRUE(pool->init());
    auto allocated = pool->malloc(3);
    ASSERT_TRUE(allocated);
    pool->incTreeRef(*allocated, BlockTreeRefType::LOAD);
    auto                      release = pool->prepareFrozenReferenceRelease(*allocated);
    TransferBatchAsyncContext context(3);
    size_t                    callbacks = 0;
    context.onDone([&](TransferResult result) {
        ++callbacks;
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        EXPECT_TRUE(result.block_errors.empty());
        for (size_t index = 0; index < allocated->size(); ++index) {
            EXPECT_EQ(result.blockError(index), CacheCopyError::RPC_FAILED);
            pool->freezeBlock((*allocated)[index]);
        }
    });
    auto original =
        TransferResult::executionException(false, "unknown completion with an allocated diagnostic for every target");
    original.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
    original.copy_error = CacheCopyError::RPC_FAILED;
    original.expandBlockErrors(3);
    allocation_countdown = 0;  // Fail the callback's diagnostic copy after publication.
    context.complete(std::move(original));
    allocation_countdown = -1;
    ASSERT_EQ(callbacks, 1u);
    pool->decTreeRef(*allocated, BlockTreeRefType::LOAD);
    context.complete(TransferStatus::OK);  // A late physical completion cannot replace UNKNOWN.
    context.onDone([&](TransferResult result) {
        EXPECT_EQ(result.status.code(), ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        for (size_t index = 0; index < allocated->size(); ++index) {
            EXPECT_EQ(result.blockError(index), CacheCopyError::RPC_FAILED);
            EXPECT_TRUE(pool->isFrozen((*allocated)[index]));
            EXPECT_TRUE(pool->isAllocated((*allocated)[index]));
        }
    });
    EXPECT_FALSE(pool->malloc().has_value());
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 3u);
    release();
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
}

}  // namespace
}  // namespace rtp_llm
