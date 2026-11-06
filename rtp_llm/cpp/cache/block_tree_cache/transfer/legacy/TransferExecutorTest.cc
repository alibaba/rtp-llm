#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/TransferExecutor.h"

namespace rtp_llm {
namespace {
TransferDescriptor descriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}
class RecordingTransferExecutor final: public TransferExecutor {
public:
    explicit RecordingTransferExecutor(BlockTreeTaskPool& pool): TransferExecutor(pool, 8, nullptr) {}
    size_t         calls{0};
    TransferStatus status{TransferStatus::OK};

private:
    TransferResult executeBatch(const std::vector<HostBufferView>&,
                                const std::vector<TransferDescriptor>&,
                                const std::vector<const GroupSet*>&) override {
        ++calls;
        return status;
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
    EXPECT_EQ(context->result().copy_error, CacheCopyError::COPY_FAILED);
    EXPECT_EQ(context->result().block_errors, (std::vector<CacheCopyError>{CacheCopyError::COPY_FAILED}));
}

}  // namespace
}  // namespace rtp_llm
