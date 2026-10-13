#include <gtest/gtest.h>

#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {
namespace {

std::shared_ptr<AsyncContext> okContext() {
    return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
}

class ScriptedPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankEngine(std::deque<std::shared_ptr<AsyncContext>> contexts = {}):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), contexts_(std::move(contexts)) {}

    std::shared_ptr<AsyncContext> execute(TransferTask task) override {
        const auto& descriptors = task.descriptors();
        ++submit_count_;
        submitted_batches_.push_back(descriptors);
        if (contexts_.empty()) {
            return okContext();
        }
        std::shared_ptr<AsyncContext> context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    size_t submittedBatchCount() const {
        return submit_count_;
    }

    const std::vector<std::vector<TransferDescriptor>>& submittedBatches() const {
        return submitted_batches_;
    }

private:
    std::deque<std::shared_ptr<AsyncContext>>    contexts_;
    size_t                                       submit_count_{0};
    std::vector<std::vector<TransferDescriptor>> submitted_batches_;
};

TransferDescriptor descriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

TransferDescriptor deviceHostDescriptor(size_t group_id) {
    return TransferDescriptor::hostToDevice(group_id, 1, {1});
}

TransferDescriptor hostDiskDescriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

TEST(BlockTransferDispatcherTest, DescriptorVectorUsesPerRankEntry) {
    std::shared_ptr<ScriptedPerRankEngine> engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher                dispatcher(engine);

    auto context =
        dispatcher.executePerRank(TransferTask({TransferDescriptor::hostToDisk(0, 1, 1)}, std::chrono::seconds(30)));
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, ExpiredTaskFailsBeforePerRankExecution) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);
    TransferTask            task({descriptor(0)}, std::chrono::seconds(1));
    task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);

    auto context = dispatcher.executePerRank(std::move(task));
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, EmptyTaskFailsBeforeDispatch) {
    BlockTransferDispatcher dispatcher(nullptr);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo::OkStatus();
    dispatcher.runTransfer(TransferTask({}, std::chrono::milliseconds(100)), [&](ErrorInfo error) {
        ++callback_count;
        final_error = std::move(error);
    });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(final_error.code(), ErrorCode::INVALID_PARAMS);
}

TEST(BlockTransferDispatcherTest, PerRankBatchUsesOneSubmit) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "callback not invoked");
    dispatcher.runTransfer(TransferTask({descriptor(0), descriptor(0), descriptor(0)}, std::chrono::milliseconds(100)),
                           [&](ErrorInfo error) {
                               ++callback_count;
                               final_error = std::move(error);
                           });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_TRUE(final_error.ok());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, MultiRankFailureDoesNotFallbackToPerRank) {
    auto per_rank_engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{okContext()});
    auto group_set = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    auto multi_rank_engine =
        std::make_shared<MultiRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_set}, nullptr);
    BlockTransferDispatcher dispatcher(per_rank_engine, multi_rank_engine);

    size_t                   callback_count = 0;
    ErrorInfo                final_error    = ErrorInfo::OkStatus();
    const TransferDescriptor unsupported;
    dispatcher.runTransfer(TransferTask({unsupported}, std::chrono::milliseconds(100)), [&](ErrorInfo error) {
        ++callback_count;
        final_error = std::move(error);
    });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(final_error.ok());
    EXPECT_EQ(per_rank_engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, RunTransferDoesNotWaitForPendingPerRankContext) {
    auto pending = std::make_shared<TransferBatchAsyncContext>();
    auto engine  = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{pending});
    BlockTransferDispatcher dispatcher(engine);

    size_t callback_count = 0;
    dispatcher.runTransfer(TransferTask({descriptor(0)}, std::chrono::milliseconds(100)), [&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++callback_count;
    });

    EXPECT_FALSE(pending->done());
    EXPECT_EQ(callback_count, 0u);
    pending->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(callback_count, 1u);
}

TEST(BlockTransferDispatcherTest, AsynchronousRunTransferGroupsAndWaitsForEveryBatch) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto third  = std::make_shared<TransferBatchAsyncContext>();
    auto engine =
        std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{first, second, third});
    BlockTransferDispatcher dispatcher(engine, nullptr, 2);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo::OkStatus();
    dispatcher.runTransfer(
        TransferTask({descriptor(0), descriptor(1), descriptor(0), descriptor(0)}, std::chrono::milliseconds(100)),
        [&](ErrorInfo error) {
            ++callback_count;
            final_error = std::move(error);
        });

    ASSERT_EQ(engine->submittedBatches().size(), 3u);
    EXPECT_EQ(engine->submittedBatches()[0].size(), 2u);
    EXPECT_EQ(engine->submittedBatches()[0][0].group_set_id, 0u);
    EXPECT_EQ(engine->submittedBatches()[0][1].group_set_id, 0u);
    EXPECT_EQ(engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(engine->submittedBatches()[1][0].group_set_id, 0u);
    EXPECT_EQ(engine->submittedBatches()[2].size(), 1u);
    EXPECT_EQ(engine->submittedBatches()[2][0].group_set_id, 1u);

    second->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second failed"));
    first->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(callback_count, 0u);
    third->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(final_error.ok());
}

TEST(BlockTransferDispatcherTest, DrainWaitsForCallbackReturnNotOnlyContextReadiness) {
    auto pending = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{pending});
    BlockTransferDispatcher dispatcher(engine);
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    std::promise<void> release;
    auto released = release.get_future().share();
    dispatcher.runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(5)), [&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        entered.set_value();
        released.wait();
    });
    auto complete = std::async(std::launch::async, [&] { pending->complete(ErrorInfo::OkStatus()); });
    const auto entered_status = entered_future.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(entered_status, std::future_status::ready);
    EXPECT_TRUE(pending->done());
    auto drain = std::async(std::launch::async, [&] { dispatcher.drainTransfers(); });
    EXPECT_EQ(drain.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    release.set_value();
    EXPECT_EQ(complete.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(drain.wait_for(std::chrono::seconds(5)), std::future_status::ready);
}

TEST(BlockTransferDispatcherTest, DrainIncludesNextStageSubmittedByCallback) {
    auto first = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(std::deque<std::shared_ptr<AsyncContext>>{first, second});
    BlockTransferDispatcher dispatcher(engine);
    bool settled = false;
    dispatcher.runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(5)), [&](ErrorInfo) {
        dispatcher.runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(5)), [&](ErrorInfo) {
            settled = true;
        });
    });
    auto drain = std::async(std::launch::async, [&] { dispatcher.drainTransfers(); });
    first->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(drain.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    second->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(drain.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(settled);
}

TEST(BlockTransferDispatcherTest, ThrowingCallbackDoesNotLeaveDrainPending) {
    auto engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);
    EXPECT_THROW(dispatcher.runTransfer(TransferTask({descriptor(0)}, std::chrono::seconds(5)),
                                        [](ErrorInfo) { throw std::runtime_error("callback"); }),
                 std::runtime_error);
    dispatcher.drainTransfers();
}

TEST(BlockTransferDispatcherTest, ExpiredTaskFailsWithoutSubmittingABatch) {
    auto                    engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);
    TransferTask            task({descriptor(0)}, std::chrono::seconds(1));
    task.deadline_ = TransferTask::Clock::now() - std::chrono::milliseconds(1);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo::OkStatus();
    dispatcher.runTransfer(std::move(task), [&](ErrorInfo error) {
        ++callback_count;
        final_error = std::move(error);
    });

    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(final_error.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, AllDirectionsUseTheConfiguredBatchLimit) {
    const std::vector<TransferDescriptor> directions{
        TransferDescriptor::hostToDevice(0, 1, {1}),
        TransferDescriptor::deviceToHost(0, {1}, 1),
        TransferDescriptor::hostToDisk(0, 1, 1),
        TransferDescriptor::diskToHost(0, 1, 1),
        TransferDescriptor::diskToDevice(0, 1, {1}),
        TransferDescriptor::deviceToDisk(0, {1}, 1)};
    for (const auto& direction : directions) {
        for (const size_t limit : {2u, 8u, 16u}) {
            auto engine = std::make_shared<ScriptedPerRankEngine>();
            BlockTransferDispatcher dispatcher(engine, nullptr, limit);
            size_t callback_count = 0;
            dispatcher.runTransfer(
                TransferTask(std::vector<TransferDescriptor>(100, direction), std::chrono::seconds(30)),
                [&](ErrorInfo error) {
                    EXPECT_TRUE(error.ok());
                    ++callback_count;
                });
            const auto& batches = engine->submittedBatches();
            const size_t expected_count = limit == 2 ? 50 : (limit == 8 ? 13 : 7);
            ASSERT_EQ(batches.size(), expected_count);
            size_t total = 0;
            for (size_t index = 0; index < batches.size(); ++index) {
                EXPECT_EQ(batches[index].size(), index + 1 == batches.size() ? (limit == 2 ? 2u : 4u) : limit);
                total += batches[index].size();
            }
            EXPECT_EQ(total, 100u);
            EXPECT_EQ(callback_count, 1u);
        }
    }
}

TEST(BlockTransferDispatcherTest, DefaultBatchLimitIsSharedByDirections) {
    auto                            device_host_engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher         device_host_dispatcher(device_host_engine);
    std::vector<TransferDescriptor> device_host_descriptors;
    for (size_t index = 0; index < 9; ++index) {
        device_host_descriptors.push_back(deviceHostDescriptor(0));
    }
    size_t device_host_callback_count = 0;
    device_host_dispatcher.runTransfer(TransferTask(device_host_descriptors, std::chrono::milliseconds(100)),
                                       [&](ErrorInfo error) {
                                           EXPECT_TRUE(error.ok());
                                           ++device_host_callback_count;
                                       });
    ASSERT_EQ(device_host_engine->submittedBatches().size(), 2u);
    EXPECT_EQ(device_host_engine->submittedBatches()[0].size(), 8u);
    EXPECT_EQ(device_host_engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(device_host_callback_count, 1u);

    auto                            host_disk_engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher         host_disk_dispatcher(host_disk_engine);
    std::vector<TransferDescriptor> host_disk_descriptors;
    for (size_t index = 0; index < 9; ++index) {
        host_disk_descriptors.push_back(hostDiskDescriptor(0));
    }
    size_t host_disk_callback_count = 0;
    host_disk_dispatcher.runTransfer(TransferTask(host_disk_descriptors, std::chrono::milliseconds(100)),
                                     [&](ErrorInfo error) {
                                         EXPECT_TRUE(error.ok());
                                         ++host_disk_callback_count;
                                     });
    ASSERT_EQ(host_disk_engine->submittedBatches().size(), 2u);
    EXPECT_EQ(host_disk_engine->submittedBatches()[0].size(), 8u);
    EXPECT_EQ(host_disk_engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(host_disk_callback_count, 1u);
}

}  // namespace
}  // namespace rtp_llm
