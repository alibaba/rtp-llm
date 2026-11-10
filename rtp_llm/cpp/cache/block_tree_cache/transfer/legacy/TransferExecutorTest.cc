#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include <stdexcept>
#include <vector>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/TransferExecutor.h"

namespace rtp_llm {
namespace {
TransferDescriptor descriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}
class RecordingTransferExecutor final: public TransferExecutor {
public:
    explicit RecordingTransferExecutor(BlockTreeTaskPool& pool, size_t batch_size = 8):
        TransferExecutor(pool, batch_size, nullptr) {}
    size_t         calls{0};
    TransferStatus status{TransferStatus::OK};
    bool           throw_on_execute{false};
    std::vector<TransferResult> results;

private:
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        ++calls;
        if (throw_on_execute) {
            throw std::runtime_error("injected executing copy exception");
        }
        return results.empty() ? TransferResult(status) : results.at(calls - 1);
    }
};

TEST(TransferExecutorTest, WorkerQueueTimeoutDoesNotExecuteBlocks) {
    BlockTreeTaskPool pool(1, 8, "TransferExpiryTest");
    ASSERT_TRUE(pool.start());
    RecordingTransferExecutor executor(pool);
    TransferTask              task({descriptor(0), descriptor(0)}, std::chrono::seconds(1));
    task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);
    auto context   = executor.execute(std::move(task), {HostBufferView{}, HostBufferView{}}, {nullptr, nullptr});
    context->waitDone();
    EXPECT_EQ(executor.calls, 0u);
    EXPECT_TRUE(context->result().buffer_safe);
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(context->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED}));
}

TEST(TransferExecutorTest, FullQueueDoesNotExecuteRejectedBlocks) {
    BlockTreeTaskPool pool(1, 1, "TransferQueueFullTest");
    ASSERT_TRUE(pool.start());
    RecordingTransferExecutor              executor(pool);
    std::promise<void>                     entered;
    std::promise<void>                     release;
    auto                                   released = release.get_future().share();
    block_tree_cache_detail::ScopeRollback unblock([&] {
        release.set_value();
        pool.waitForIdle();
    });
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::BACKGROUND, [&] {
        entered.set_value();
        released.wait();
    }));
    ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_TRUE(pool.submit(BlockTreeTaskClass::BACKGROUND, [] {}));
    auto context = executor.execute(TransferTask({descriptor(0), descriptor(0)}, std::chrono::seconds(30)),
                                    {HostBufferView{}, HostBufferView{}},
                                    {nullptr, nullptr});
    context->waitDone();
    EXPECT_EQ(executor.calls, 0u);
    EXPECT_TRUE(context->result().buffer_safe);
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(context->errorInfo().ToString().find("queue is full or stopped"), std::string::npos);
    EXPECT_EQ(context->result().copy_error, CacheCopyError::NOT_EXECUTED);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED}));
}

TEST(TransferExecutorTest, ExecutedCopyFailureKeepsCopyFailedClassification) {
    BlockTreeTaskPool pool(1, 8, "TransferFailureTest");
    ASSERT_TRUE(pool.start());
    RecordingTransferExecutor executor(pool);
    executor.status = TransferStatus::DEVICE_IO_ERROR;
    auto context =
        executor.execute(TransferTask({descriptor(0)}, std::chrono::seconds(30)), {HostBufferView{}}, {nullptr});
    context->waitDone();
    EXPECT_EQ(executor.calls, 1u);
    EXPECT_FALSE(context->result().buffer_safe);
    EXPECT_EQ(context->result().copy_error, CacheCopyError::COPY_FAILED);
    EXPECT_EQ(context->result().block_errors, (std::vector<CacheCopyError>{CacheCopyError::COPY_FAILED}));
}

TEST(TransferExecutorTest, BatchSafetyRequiresEveryExecutedBatchToProveCompletion) {
    BlockTreeTaskPool pool(1, 8, "TransferBatchSafetyTest");
    ASSERT_TRUE(pool.start());
    for (const bool first_safe : {false, true}) {
        for (const bool second_safe : {false, true}) {
            for (const bool failed : {false, true}) {
                RecordingTransferExecutor executor(pool, 1);
                TransferResult first(TransferStatus::OK);
                first.buffer_safe = first_safe;
                TransferResult second(failed ? TransferStatus::DISK_IO_ERROR : TransferStatus::OK);
                second.buffer_safe = second_safe;
                executor.results = {first, second};
                auto context = executor.execute(
                    TransferTask({descriptor(0), descriptor(0)}, std::chrono::seconds(30)),
                    {HostBufferView{}, HostBufferView{}},
                    {nullptr, nullptr});
                context->waitDone();
                pool.waitForIdle();
                EXPECT_EQ(executor.calls, 2u);
                EXPECT_EQ(context->success(), !failed);
                EXPECT_EQ(context->result().buffer_safe, first_safe && second_safe);
            }
        }
    }
}

TEST(TransferExecutorTest, ExecutingExceptionDoesNotClaimBufferSafety) {
    BlockTreeTaskPool pool(1, 8, "TransferExceptionSafetyTest");
    ASSERT_TRUE(pool.start());
    RecordingTransferExecutor executor(pool);
    executor.throw_on_execute = true;
    auto context =
        executor.execute(TransferTask({descriptor(0)}, std::chrono::seconds(30)), {HostBufferView{}}, {nullptr});
    context->waitDone();
    pool.waitForIdle();
    EXPECT_EQ(executor.calls, 1u);
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_FALSE(context->result().buffer_safe);
}

TEST(TransferExecutorTest, PerRankValidationRejectsBeforeBufferAccess) {
    for (const bool async : {false, true}) {
        PerRankBlockTransferEngine engine({}, false, {}, 128, 8, 1, 8, nullptr, 0, 4, async);
        for (const auto& descriptors : std::vector<std::vector<TransferDescriptor>>{{}, {descriptor(0)}}) {
            auto context = engine.execute(TransferTask(descriptors, std::chrono::seconds(30)));
            EXPECT_TRUE(context->done());
            EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(context->result().copy_error, CacheCopyError::INVALID_REQUEST);
            EXPECT_TRUE(context->result().buffer_safe);
        }
    }
}

template<typename Executor>
class ScriptedStageExecutor final: public Executor {
public:
    using Executor::Executor;
    TransferResult result{TransferStatus::OK};
    bool           throw_on_execute{false};
    size_t         calls{0};

private:
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        ++calls;
        if (throw_on_execute) {
            throw std::runtime_error("injected executing stage exception");
        }
        return result;
    }
};

GroupSetPtr stagingTestGroup() {
    // Scripted stages never access physical block IDs; only staging geometry is needed.
    auto spec = std::make_shared<OpaqueKVCacheSpec>("staging_safety");
    spec->setLayout(4096, 1, 4096, 4096);
    GroupBase group;
    group.tag    = spec->tag;
    group.spec   = spec;
    group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology = CacheTopology::create({group}, {{0, {group.tag}}});
    auto result = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    result->initialize(0, std::move(topology), {group.tag});
    return result;
}

TEST(TransferExecutorTest, DeviceDiskRejectsAndCancelsBeforeEitherStageStarts) {
    BlockTreeTaskPool pool(1, 8, "DeviceDiskAdmissionSafetyTest");
    ScriptedStageExecutor<DeviceHostTransferExecutor> device(pool, 8);
    ScriptedStageExecutor<HostDiskTransferExecutor> disk(pool, 8);
    const auto group = stagingTestGroup();
    DeviceDiskTransferExecutor executor(device, disk, {group}, 2, pool, 8, nullptr);
    auto invalid = executor.execute(TransferTask({}, std::chrono::seconds(30)), {});
    EXPECT_EQ(invalid->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(invalid->result().buffer_safe);

    for (const bool load : {false, true}) {
        const auto desc = load ? TransferDescriptor::diskToDevice(0, 1, {1}) :
                                 TransferDescriptor::deviceToDisk(0, {1}, 1);
        TransferTask expired({desc}, std::chrono::seconds(30));
        expired.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);
        auto timed_out = executor.execute(std::move(expired), {group.get()});
        EXPECT_TRUE(timed_out->done());
        EXPECT_EQ(timed_out->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_EQ(timed_out->result().copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_TRUE(timed_out->result().buffer_safe);

        auto held = executor.full_staging_pool_->tryMallocBatch(1);
        ASSERT_TRUE(held.has_value());
        auto cancelled = executor.execute(TransferTask({desc}, std::chrono::seconds(30)), {group.get()});
        EXPECT_FALSE(cancelled->done());
        executor.cancelPendingTransfers();
        EXPECT_TRUE(cancelled->done());
        EXPECT_EQ(cancelled->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(cancelled->result().copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_TRUE(cancelled->result().buffer_safe);
        held.reset();

        // The staging lease succeeds, but this worker queue was never started.
        auto rejected = executor.execute(TransferTask({desc}, std::chrono::seconds(30)), {group.get()});
        EXPECT_TRUE(rejected->done());
        EXPECT_EQ(rejected->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(rejected->result().copy_error, CacheCopyError::NOT_EXECUTED);
        EXPECT_TRUE(rejected->result().buffer_safe);
    }
    EXPECT_EQ(device.calls, 0u);
    EXPECT_EQ(disk.calls, 0u);
}

TEST(TransferExecutorTest, DeviceDiskCombinesStageProofsAndKeepsExecutingExceptionsUnsafe) {
    BlockTreeTaskPool pool(1, 8, "DeviceDiskStageSafetyTest");
    ASSERT_TRUE(pool.start());
    ScriptedStageExecutor<DeviceHostTransferExecutor> device(pool, 8);
    ScriptedStageExecutor<HostDiskTransferExecutor> disk(pool, 8);
    const auto group = stagingTestGroup();
    DeviceDiskTransferExecutor executor(device, disk, {group}, 2, pool, 8, nullptr);
    for (const bool load : {false, true}) {
        for (const bool device_safe : {false, true}) {
            for (const bool disk_safe : {false, true}) {
                device.result.buffer_safe = device_safe;
                disk.result.buffer_safe = disk_safe;
                const auto desc = load ? TransferDescriptor::diskToDevice(0, 1, {1}) :
                                         TransferDescriptor::deviceToDisk(0, {1}, 1);
                auto context = executor.execute(TransferTask({desc}, std::chrono::seconds(30)), {group.get()});
                context->waitDone();
                pool.waitForIdle();
                EXPECT_TRUE(context->success());
                EXPECT_EQ(context->result().buffer_safe, device_safe && disk_safe);
            }
        }
    }
    device.result.buffer_safe = true;
    disk.throw_on_execute = true;
    auto failed = executor.execute(
        TransferTask({TransferDescriptor::deviceToDisk(0, {1}, 1)}, std::chrono::seconds(30)), {group.get()});
    failed->waitDone();
    pool.waitForIdle();
    EXPECT_EQ(failed->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_FALSE(failed->result().buffer_safe);
}

}  // namespace
}  // namespace rtp_llm
